# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# This file is part of ByteQC.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https: // www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""K-point CDERI utilities for future KRMP2 work.

This module keeps the KRMP2 k-point CDERI flow independent from the existing
ByteQC gamma-point CDERI path. The k-point CDERI builder is derived from the
`gpu4pyscf` RSDF implementation but is localized here so the storage policy can
be changed without modifying `gpu4pyscf` itself.

Current policy:
- full KRMP2 k-point CDERI only
- no gamma-only path
- no MF/SCF object input
- compressed CDERI is written slice-by-slice to `FileMp`
- writes use `WRITE_PROCESSSES` FileMp workers for current CDERI throughput tests
"""

from __future__ import annotations

import gc
import os
import tempfile
from multiprocessing import Pool
import warnings

import cupy as cp
import cupyx
import numpy as np
from pyscf import df as mol_df
from pyscf.lib import param
from pyscf.pbc.df import incore as pbc_df_incore
from pyscf.pbc.tools.k2gamma import (
    double_translation_indices,
    translation_vectors_for_kmesh,
)

from byteqc import lib
from byteqc.lib.utils import is_pinned
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import asarray, contract, ndarray
from gpu4pyscf.pbc.df import ft_ao, rsdf_builder
from gpu4pyscf.pbc.df.int3c2e import SRInt3c2eOpt
from gpu4pyscf.pbc.df.rsdf_builder import (
    LINEAR_DEP_THR,
    OMEGA_MIN,
    _precontract_j2c_aux_coeff,
)
from gpu4pyscf.pbc.lib.kpts_helper import conj_images_in_bvk_cell, kk_adapted_iter
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.pbc.tools.pbc import _Gv_wrap_around, get_coulG

WRITE_PROCESSSES = 2
READ_PROCESSSES = 8

def _to_numpy_array(arr, dtype=None) -> np.ndarray:
    if isinstance(arr, cp.ndarray):
        arr = cp.asnumpy(arr)
    return np.asarray(arr, dtype=dtype)


def _get_kk_conserv(record) -> np.ndarray:
    cache = record["_cache"]
    if "kk_conserv" not in cache:
        cache["kk_conserv"] = double_translation_indices(record["kmesh"])
    return cache["kk_conserv"]


def _tree_nbytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def unpack_krmp2_cderi_kaux(record, k_aux: int, negative: bool = False, axis: int = 0) -> np.ndarray:
    source = record["cderip"] if negative else record["cderi"]
    if not source:
        raise KeyError(f"Compressed CDERI block for k_aux={k_aux} is not available")

    kk_conserv = _get_kk_conserv(record)
    _, kj_idx = np.where(kk_conserv == k_aux)
    cache = record["_cache"]

    if "conj_mapping" not in cache:
        cache["conj_mapping"] = _to_numpy_array(
            conj_images_in_bvk_cell(record["kmesh"]), dtype=np.int64
        )

    if k_aux not in source:
        k_aux_conj = int(cache["conj_mapping"][k_aux])
        if k_aux_conj == k_aux or k_aux_conj not in source:
            raise KeyError(f"Compressed CDERI block for k_aux={k_aux} is not available")

        unpacked_conj = unpack_krmp2_cderi_kaux(
            record, k_aux_conj, negative=negative, axis=axis
        )
        unpacked = np.empty_like(unpacked_conj)
        ki_idx, kj_idx = np.where(kk_conserv == k_aux)
        for ki, kj in zip(ki_idx, kj_idx):
            unpacked[ki] = unpacked_conj[kj].conj().transpose(0, 2, 1)
        return unpacked

    int3c_cell = record.get("int3c_cell", record["cell"])
    phases = translation_vectors_for_kmesh(int3c_cell, record["kmesh"], True)
    phases = _to_numpy_array(phases, dtype=float).dot(record["kpts"].T)
    expLk = cp.exp(1j * cp.asarray(phases))
    dataset_name = record["cderi_full"].get(int(k_aux))
    if dataset_name is None:
        raise KeyError(f"Compressed CDERI block for k_aux={k_aux} is not available")
    pool_read = None
    try:
        file = lib.FileMp(record["file"], "r")
        dataset = file[dataset_name]
        cderi_full = cupyx.empty_pinned(dataset.shape, dtype=dataset.dtype, order="C")
        if not is_pinned(cderi_full):
            raise RuntimeError("KRMP2 CDERI loader requires pinned host memory")
        pool_read = Pool(processes=max(1, int(record.get("file_processes", READ_PROCESSSES))))
        future = dataset.getitem(np.s_[:, :], pool=pool_read, buf=cderi_full)
        future.wait()
        cderi_full = future.view(np.ndarray)
        dataset = None
        file.close()
    finally:
        if pool_read is not None:
            pool_read.close()
            pool_read.join()
    naux_negative = int(record["naoaux_negative_by_kpt"].get(int(k_aux), 0))
    if negative:
        if naux_negative <= 0:
            raise KeyError(f"Negative-metric CDERI block for k_aux={k_aux} is not available")
        cderi_compressed = cderi_full[-naux_negative:]
    elif naux_negative > 0:
        cderi_compressed = cderi_full[:-naux_negative]
    else:
        cderi_compressed = cderi_full

    unpacked = rsdf_builder._unpack_cderi_v2(
        cderi_compressed,
        record["cderi_idx"][0],
        kj_idx,
        cache["conj_mapping"],
        expLk,
        record["nao"],
        axis=axis,
    )
    if isinstance(unpacked, cp.ndarray):
        out = cp.asnumpy(unpacked)
    else:
        out = _to_numpy_array(unpacked)
    del unpacked, expLk, cderi_full, cderi_compressed, phases, dataset_name
    cp.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()
    return out


def unpack_krmp2_cderi_kpair(record, ki: int, kj: int, negative: bool = False, axis: int = 0) -> np.ndarray:
    k_aux = int(_get_kk_conserv(record)[ki, kj])
    unpacked = unpack_krmp2_cderi_kaux(
        record, k_aux, negative=negative, axis=axis
    )
    return _to_numpy_array(unpacked[ki])


def build_krmp2_cderi(
    cell,
    kpts,
    auxbasis=None,
    auxcell=None,
    kmesh=None,
    path=None,
    file_processes: int = WRITE_PROCESSSES,
    pair_blksize=None,
    omega=None,
    linear_dep_threshold=LINEAR_DEP_THR,
) -> dict:
    """Build compressed k-point CDERI and store it in `FileMp`.

    Current scope is intentionally narrow:
    - 3D periodic cells only
    - no gamma-only calculation path
    - full k-point CDERI (`j_only=False`)
    - FileMp worker count follows `WRITE_PROCESSSES`
    """

    if cell.dimension != 3:
        raise NotImplementedError("The current KRMP2 CDERI wrapper only supports 3D cells")
    # if int(file_processes) != int(WRITE_PROCESSSES):
    #     raise NotImplementedError(
    #         "The current KRMP2 CDERI disk path expects file_processes to match WRITE_PROCESSSES"
    #     )

    log = logger.new_logger(cell, cell.verbose)
    if cell.omega == 0:
        with_long_range = True
        if omega is None:
            omega = OMEGA_MIN
        omega = abs(float(omega))
    else:
        with_long_range = False
        if omega is not None and abs(float(omega)) != abs(cell.omega):
            raise ValueError("Custom omega is not supported when cell.omega != 0")
        omega = abs(float(cell.omega))

    if hasattr(kpts, "kpts"):
        kpts = kpts.kpts
    if kpts is None:
        raise ValueError("kpts must be provided explicitly for KRMP2 CDERI")
    kpts = _to_numpy_array(kpts, dtype=float)
    if kpts.shape == (3,):
        kpts = kpts.reshape(1, 3)
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise ValueError(f"Invalid k-point shape: {kpts.shape}")
    if len(kpts) == 1:
        raise NotImplementedError("Gamma-only CDERI is intentionally out of scope for this KRMP2 path")

    if kmesh is None:
        kmesh = kpts_to_kmesh(
            cell,
            kpts,
            rcut=cell.rcut * 10,
            bound_by_supmol=False,
        )
    else:
        kmesh = _to_numpy_array(kmesh, dtype=int)
        if int(np.prod(kmesh)) != len(kpts):
            raise ValueError(
                f"kmesh {tuple(kmesh)} is incompatible with {len(kpts)} k-points"
            )
    if int(np.prod(kmesh)) == 1:
        raise NotImplementedError("Gamma-only kmesh is intentionally out of scope for this KRMP2 path")

    if auxcell is None:
        if auxbasis is None:
            auxbasis = mol_df.addons.make_auxbasis(cell, mp2fit=True)
        auxcell = pbc_df_incore.make_auxcell(cell, auxbasis)

    if path is None:
        path = tempfile.mkdtemp(prefix="byteqc_krmp2_cderi_", dir=param.TMPDIR)
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "cderi.h5")

    kpt_iters = list(kk_adapted_iter(kmesh))
    uniq_kpts = kpts[[x[0] for x in kpt_iters]]
    nkpts = len(uniq_kpts)
    bvk_ncells = int(np.prod(kmesh))
    if len(kpts) != bvk_ncells:
        raise ValueError(
            f"kmesh {tuple(kmesh)} implies {bvk_ncells} k-points, got {len(kpts)}"
        )

    # Build the same RSDF ingredients as gpu4pyscf's compressed_cderi_kk, but
    # keep the final compressed blocks out-of-core in FileMp instead of mapped
    # host arrays.
    ao_cell = cell
    int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega=-omega, bvk_kmesh=kmesh).build()
    int3c_cell = int3c2e_opt.cell
    cell = int3c_cell
    auxcell = int3c2e_opt.auxcell
    ao_cell = getattr(int3c_cell, "mol", ao_cell)

    cd_j2c_cache, negative_metric_size = _precontract_j2c_aux_coeff(
        auxcell,
        kpts,
        omega,
        with_long_range,
        linear_dep_threshold,
        kmesh,
    )
    naux_cart = int(cd_j2c_cache[0].shape[0])
    naux_max = max(int(x.shape[1]) for x in cd_j2c_cache)

    cderi_idx = int3c2e_opt.pair_and_diag_indices()
    nao_pairs = int(len(cderi_idx[0]))

    mesh = int3c2e_opt.mesh
    Gv, _Gvbase, kws = cell.get_Gv_weights(mesh)
    ngrids = len(Gv)
    Gk = (Gv + uniq_kpts[:, None]).reshape(-1, 3)
    Gk = _Gv_wrap_around(cell, Gk, cp.zeros(3), mesh)
    coulG = get_coulG(cell, Gv=Gk, omega=omega).reshape(nkpts, ngrids)
    coulG *= kws
    coulG[0, 0] -= np.pi / omega**2 / cell.vol

    mem_free = cp.cuda.runtime.memGetInfo()[0]
    mem_free -= sum(x.nbytes for x in cd_j2c_cache)
    mem_free -= ngrids * naux_max * 16 * nkpts
    batch_size = int(
        min(nao_pairs, max(mem_free // max(nkpts * naux_cart * 16 * 4, 1) + 225, 1))
    )

    cderi_full = {}
    cderi = {}
    cderip = {}
    naoaux_total_by_kpt = {}
    naoaux_negative_by_kpt = {}

    nsp_per_block = ft_ao.ft_ao_scheme()[0]
    bas_ij_aggregated = cell.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block
    )
    eval_j3c, aux_sorting, ao_pair_offsets = int3c2e_opt.int3c2e_evaluator(
        ao_pair_batch_size=batch_size,
        bas_ij_aggregated=bas_ij_aggregated,
    )[:3]
    shl_pair_batches = len(ao_pair_offsets) - 1
    if pair_blksize is None:
        # Align FileMp's second-axis blocks with the exact AO-pair batches that
        # the RSDF builder will generate. This makes each write hit one block.
        pair_blksizes = tuple(
            int(x) for x in np.diff(np.asarray(ao_pair_offsets, dtype=np.int64)) if int(x) > 0
        )
        if not pair_blksizes or sum(pair_blksizes) != nao_pairs:
            raise RuntimeError("Invalid AO-pair offsets for FileMp block construction")
    else:
        pair_blksizes = max(1, min(int(pair_blksize), nao_pairs))

    aux_coeffs = []
    for coeff in cd_j2c_cache:
        aux_coeff = cp.empty_like(coeff)
        aux_coeff[aux_sorting] = cp.asarray(coeff)
        aux_coeffs.append(aux_coeff)

    expLk = cp.exp(1j * cp.asarray(int3c2e_opt.bvkmesh_Ls.dot(uniq_kpts.T)))
    expLk_conjz = expLk.conj().view(np.float64).reshape(bvk_ncells, nkpts, 2)

    if with_long_range:
        ft_opt = ft_ao.FTOpt.from_intopt(int3c2e_opt)
        eval_ft, _ao_pair_offsets = ft_opt.ft_evaluator(
            batch_size, bas_ij_aggregated=bas_ij_aggregated
        )
        if not np.array_equal(ao_pair_offsets, _ao_pair_offsets):
            raise RuntimeError("AO-pair offsets mismatch between SR and LR evaluators")

        auxG = ft_ao.ft_ao(auxcell, Gk).T
        auxG = auxG.reshape(naux_cart, nkpts, ngrids)
        for k in range(nkpts):
            auxG[aux_sorting, k] = auxG[:, k].conj()
        auxG_conj = auxG
        auxG_conj *= cp.asarray(coulG)

        avail_mem = mem_free - nkpts * naux_cart * batch_size * 16 * 2
        Gblksize = int(avail_mem // max(16 * batch_size, 1)) // 32 * 32
        if Gblksize <= 0:
            raise RuntimeError("Insufficient GPU memory")
        Gblksize = min(Gblksize, ngrids)
        buf2 = cp.empty(batch_size * Gblksize, dtype=np.complex128)
    else:
        eval_ft = None
        auxG_conj = None
        Gblksize = 0
        buf2 = None

    buf0 = cp.empty(nkpts * batch_size * naux_cart, dtype=np.complex128)
    buf1 = cp.empty(naux_max * batch_size * bvk_ncells, dtype=np.complex128)
    host_buffers = {}
    file = lib.FileMp(file_path, "w")
    pool_write = None
    try:
        datasets = {}
        for j2c_idx, (kp, _kp_conj, _ki_idx, _kj_idx) in enumerate(kpt_iters):
            dataset_name = f"cderi_full_{int(kp)}"
            naux_total = int(cd_j2c_cache[j2c_idx].shape[1])
            naux_negative = int(negative_metric_size.get(j2c_idx, 0))
            cderi_full[int(kp)] = dataset_name
            cderi[int(kp)] = dataset_name
            if naux_negative > 0:
                cderip[int(kp)] = dataset_name
            naoaux_total_by_kpt[int(kp)] = naux_total
            naoaux_negative_by_kpt[int(kp)] = naux_negative
            datasets[int(kp)] = file.create_dataset(
                dataset_name,
                (naux_total, nao_pairs),
                np.complex128,
                blksizes=(naux_total, pair_blksizes),
            )

        cderi_nbytes_by_kpt = {}
        cderi_negative_nbytes_by_kpt = {}
        cderi_full_nbytes_by_kpt = {}
        cderi_disk_nbytes_by_kpt = {}
        for kp, dataset_name in cderi_full.items():
            kp = int(kp)
            naux_pos = int(naoaux_total_by_kpt[kp] - naoaux_negative_by_kpt[kp])
            naux_neg = int(naoaux_negative_by_kpt[kp])
            naux_full = int(naoaux_total_by_kpt[kp])
            cderi_nbytes_by_kpt[kp] = naux_pos * nao_pairs * np.dtype(np.complex128).itemsize
            cderi_negative_nbytes_by_kpt[kp] = naux_neg * nao_pairs * np.dtype(np.complex128).itemsize
            cderi_full_nbytes_by_kpt[kp] = naux_full * nao_pairs * np.dtype(np.complex128).itemsize
            dataset_dir = os.path.join(
                os.path.splitext(file_path)[0] + "_Mp",
                dataset_name,
            )
            cderi_disk_nbytes_by_kpt[kp] = _tree_nbytes(dataset_dir) if os.path.isdir(dataset_dir) else 0

        cderi_nbytes = sum(cderi_nbytes_by_kpt.values())
        cderi_negative_nbytes = sum(cderi_negative_nbytes_by_kpt.values())
        cderi_full_nbytes = sum(cderi_full_nbytes_by_kpt.values())
        cderi_disk_nbytes = _tree_nbytes(path)
        log.info(
            "KRMP2 CDERI size before build: naoaux=%d negative_aux=%d logic=%.3f GB negative=%.3f GB full=%.3f GB disk=%.3f GB",
            max(int(naoaux_total_by_kpt[kp] - naoaux_negative_by_kpt[kp]) for kp in cderi_full),
            max(int(x) for x in naoaux_negative_by_kpt.values()),
            cderi_nbytes / 1024**3,
            cderi_negative_nbytes / 1024**3,
            cderi_full_nbytes / 1024**3,
            cderi_disk_nbytes / 1024**3,
        )
        top_kpts = sorted(cderi_full_nbytes_by_kpt.items(), key=lambda kv: kv[1], reverse=True)
        if top_kpts:
            log.info(
                "KRMP2 CDERI largest q-blocks before build: %s",
                ", ".join(
                    f"q={kp}: full={nbytes/1024**2:.1f} MiB disk={cderi_disk_nbytes_by_kpt[kp]/1024**2:.1f} MiB"
                    for kp, nbytes in top_kpts[: min(4, len(top_kpts))]
                ),
            )

        host_buffers = {
            kp: [
                cupyx.empty_pinned(
                    (naoaux_total_by_kpt[kp], int(max(pair_blksizes))),
                    dtype=np.complex128,
                    order="C",
                ),
                cupyx.empty_pinned(
                    (naoaux_total_by_kpt[kp], int(max(pair_blksizes))),
                    dtype=np.complex128,
                    order="C",
                ),
            ]
            for kp in cderi_full
        }
        pending_writes = {
            kp: [None, None] for kp in cderi_full
        }
        for kp in cderi_full:
            for buf in host_buffers[kp]:
                if not is_pinned(buf):
                    raise RuntimeError("KRMP2 CDERI FileMp write buffers must be pinned")
        # FileMp's ioMp uses raw pointer passing (arr2ptr/ptr2arr), so this
        # path must inherit the parent's address space via fork. Python 3.12
        # warns whenever fork happens after helper threads have been created.
        # We keep the required fork semantics here and suppress only this
        # noisy deprecation warning locally.

        pool_write = Pool(processes=int(file_processes))

        for batch_id in range(shl_pair_batches):
            log.debug1("KRMP2 CDERI batch %d/%d", batch_id, shl_pair_batches)

            j3c = eval_j3c(shl_pair_batch_id=batch_id, out=buf1)
            if j3c.size == 0:
                continue

            pair_size = int(j3c.shape[0])
            j3c_buf = ndarray((nkpts, naux_cart, pair_size, 2), buffer=buf0)
            j3c = contract("prL,LKz->Krpz", j3c, expLk_conjz, out=j3c_buf)
            j3c = j3c.view(np.complex128)[:, :, :, 0]

            if with_long_range:
                for j2c_idx, (kp, _kp_conj, _ki_idx, _kj_idx) in enumerate(kpt_iters):
                    for g0 in range(0, ngrids, Gblksize):
                        g1 = min(ngrids, g0 + Gblksize)
                        auxG_c = auxG_conj[:, j2c_idx, g0:g1]
                        pqG = eval_ft(Gv[g0:g1] + kpts[kp], batch_id, out=buf2)
                        contract("rG,pG->rp", auxG_c, pqG, beta=1.0, out=j3c[j2c_idx])

            p0 = int(ao_pair_offsets[batch_id])
            p1 = int(ao_pair_offsets[batch_id + 1])

            slot = batch_id & 1
            # Keep writes asynchronous until this double-buffer slot is about to
            # be reused, so disk IO overlaps with the next batch's GPU work.
            for kp in cderi_full:
                waits = pending_writes[kp][slot]
                if waits is not None:
                    for wait in waits:
                        wait.wait()
                    pending_writes[kp][slot] = None

            for j2c_idx, (kp, _kp_conj, _ki_idx, _kj_idx) in enumerate(kpt_iters):
                aux_coeff = aux_coeffs[j2c_idx]
                naux_total = int(aux_coeff.shape[1])
                cderi_k = ndarray((naux_total, pair_size), dtype=np.complex128, buffer=buf1)
                cderi_k = aux_coeff.T.dot(j3c[j2c_idx], out=cderi_k)

                kp = int(kp)
                out_host = lib.empty_from_buf(
                    host_buffers[kp][slot],
                    (naux_total, pair_size),
                    dtype=np.complex128,
                    order="C",
                    type=lib.MemoryTypeHost,
                )
                if not is_pinned(out_host):
                    raise RuntimeError("KRMP2 CDERI FileMp write slices must stay in pinned host memory")
                cderi_k.get(out=out_host, blocking=True)
                pending_writes[kp][slot] = datasets[kp].setitem(
                    np.s_[:, p0:p1], out_host, pool=pool_write
                )

        for kp in cderi_full:
            for waits in pending_writes[kp]:
                if waits is not None:
                    for wait in waits:
                        wait.wait()
    except Exception:
        if pool_write is not None:
            pool_write.terminate()
            pool_write.join()
        file.close()
        raise
    else:
        if pool_write is not None:
            pool_write.close()
            pool_write.join()
        file.close()

    record = {
        "cell": ao_cell,
        "ao_cell": ao_cell,
        "int3c_cell": int3c_cell,
        "auxcell": auxcell,
        "kpts": kpts,
        "nkpts": int(len(kpts)),
        "kmesh": _to_numpy_array(kmesh, dtype=int),
        "nao": int(ao_cell.nao),
        "int3c_nao": int(int3c_cell.nao),
        "naoaux": max(
            int(naoaux_total_by_kpt[kp] - naoaux_negative_by_kpt[kp]) for kp in cderi_full
        ),
        "naoaux_negative": max(int(x) for x in naoaux_negative_by_kpt.values()),
        "naoaux_by_kpt": {
            int(kp): int(naoaux_total_by_kpt[kp] - naoaux_negative_by_kpt[kp])
            for kp in cderi_full
        },
        "naoaux_negative_by_kpt": {
            int(kp): int(naoaux_negative_by_kpt[kp]) for kp in cderi_full
        },
        "cderi_full": cderi_full,
        "cderi": cderi,
        "cderip": cderip,
        "cderi_idx": (
            _to_numpy_array(cderi_idx[0], dtype=np.int64),
            _to_numpy_array(cderi_idx[1], dtype=np.int64),
        ),
        "omega": omega,
        "linear_dep_threshold": linear_dep_threshold,
        "ao_pair_offsets": _to_numpy_array(ao_pair_offsets, dtype=np.int64),
        "pair_blksizes": _to_numpy_array(pair_blksizes, dtype=np.int64),
        "path": path,
        "file": file_path,
        "_cache": {},
    }
    del aux_coeffs, cd_j2c_cache, buf0, buf1, host_buffers, expLk, expLk_conjz
    del Gv, Gk, coulG, int3c2e_opt, eval_j3c, ao_pair_offsets, aux_sorting
    del bas_ij_aggregated, kpt_iters, uniq_kpts
    if with_long_range:
        del auxG_conj, eval_ft, buf2
    cp.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()
    return record
