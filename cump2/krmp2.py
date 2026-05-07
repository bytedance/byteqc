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

"""Standalone KRMP2 GPU kernel built around occupied-sliced ovL reads.

The current kernel assumes neither the full ovL tensor nor the full t2 tensor can
fit on device. It therefore works on occupied-space slices:

- ovL is consumed slice-by-slice along the occupied axis
- t2 is formed only for the current `(si, sj, ka)` block
- direct/exchange energies are accumulated immediately on GPU
- no KRMP2-stage async IO is introduced yet; the first out-of-core version stays
  synchronous on purpose
"""

from __future__ import annotations

from multiprocessing import Pool
from typing import Optional

import cupy as cp
import numpy as np
from pyscf.lib import logger
from pyscf.pbc.lib import kpts_helper

from byteqc import lib
from byteqc.cump2.dfmp2_2 import div_t2 as _div_t2_real
from byteqc.cump2.krmp2_ovl import (
    build_krmp2_ovl as _build_krmp2_ovl_record,
    read_krmp2_ovl_kpair,
    remove_krmp2_file_record,
)
from byteqc.lib.utils import is_pinned


_DIV_T2_COMPLEX128 = cp.ElementwiseKernel(
    "raw float64 e_i, raw float64 e_a, raw float64 e_j, raw float64 e_b,"
    " int64 nb, int64 nj_nb, int64 na_nj_nb",
    "complex128 out",
    """
    size_t iocc = i / na_nj_nb;
    size_t rem0 = i % na_nj_nb;
    size_t avir = rem0 / nj_nb;
    size_t rem1 = rem0 % nj_nb;
    size_t jocc = rem1 / nb;
    size_t bvir = rem1 % nb;

    double denom = e_i[iocc] - e_a[avir] + e_j[jocc] - e_b[bvir];
    if (fabs(denom) > 1e-14) {
        out /= denom;
    } else {
        out = 0;
    }
    """,
    "byteqc_krmp2_div_t2_complex128",
)


def _to_numpy_block(block):
    if isinstance(block, cp.ndarray):
        return cp.asnumpy(block)
    return np.asarray(block)


def _normalize_host_mo_coeff(mo_coeff, nkpts: int):
    if isinstance(mo_coeff, np.ndarray):
        if mo_coeff.ndim != 3 or mo_coeff.shape[0] != nkpts:
            raise ValueError(
                f"mo_coeff must have shape (nkpts, nao, nmo); got {mo_coeff.shape}"
            )
        return [_to_numpy_block(mo_coeff[k]) for k in range(nkpts)]

    mo_coeff = [_to_numpy_block(block) for block in mo_coeff]
    if len(mo_coeff) != nkpts:
        raise ValueError(f"mo_coeff must contain {nkpts} k-point blocks; got {len(mo_coeff)}")
    return mo_coeff


def _normalize_mo_energy_blocks(mo_energy, nkpts: int):
    if isinstance(mo_energy, np.ndarray):
        if mo_energy.ndim != 2 or mo_energy.shape[0] != nkpts:
            raise ValueError(
                f"mo_energy must have shape (nkpts, nmo); got {mo_energy.shape}"
            )
        return [np.asarray(mo_energy[k], dtype=np.float64) for k in range(nkpts)]

    mo_energy = [np.asarray(_to_numpy_block(block), dtype=np.float64) for block in mo_energy]
    if len(mo_energy) != nkpts:
        raise ValueError(f"mo_energy must contain {nkpts} k-point blocks; got {len(mo_energy)}")
    return mo_energy


def _resolve_occ_vir_indices(mo_energy, mo_occ, nocc, ovl=None):
    nkpts = len(mo_energy)
    occ_idx_by_kpt = []
    vir_idx_by_kpt = []

    if mo_occ is not None:
        if isinstance(mo_occ, np.ndarray):
            mo_occ = [_to_numpy_block(mo_occ[k]) for k in range(nkpts)]
        else:
            mo_occ = [_to_numpy_block(block) for block in mo_occ]

        if len(mo_occ) != nkpts:
            raise ValueError(f"mo_occ must contain {nkpts} k-point blocks; got {len(mo_occ)}")

        for k in range(nkpts):
            occ_k = np.asarray(mo_occ[k])
            if occ_k.shape != mo_energy[k].shape:
                raise ValueError(
                    f"mo_occ[{k}] must have shape {mo_energy[k].shape}; got {occ_k.shape}"
                )
            occ_idx_by_kpt.append(np.where(occ_k > 0)[0].astype(np.int32))
            vir_idx_by_kpt.append(np.where(occ_k == 0)[0].astype(np.int32))
        return occ_idx_by_kpt, vir_idx_by_kpt

    if nocc is not None:
        if np.isscalar(nocc):
            nocc_by_kpt = [int(nocc)] * nkpts
        else:
            nocc_by_kpt = [int(x) for x in nocc]
            if len(nocc_by_kpt) != nkpts:
                raise ValueError(f"nocc must contain {nkpts} entries; got {len(nocc_by_kpt)}")

        for k in range(nkpts):
            nmo_k = int(mo_energy[k].shape[0])
            nocc_k = int(nocc_by_kpt[k])
            if not 0 < nocc_k < nmo_k + 1:
                raise ValueError(f"Invalid nocc[{k}]={nocc_k} for nmo={nmo_k}")
            occ_idx_by_kpt.append(np.arange(nocc_k, dtype=np.int32))
            vir_idx_by_kpt.append(np.arange(nocc_k, nmo_k, dtype=np.int32))
        return occ_idx_by_kpt, vir_idx_by_kpt

    if ovl is None:
        raise ValueError("Either mo_occ, nocc, or ovl must be provided")

    for ki in range(nkpts):
        occ_dim = None
        for ka in range(nkpts):
            block = ovl[ki, ka]
            if block is None:
                continue
            occ_dim = int(block.shape[0])
            break
        if occ_dim is None:
            raise ValueError(f"Unable to infer occupied dimension for k-point {ki}")
        occ_idx_by_kpt.append(np.arange(occ_dim, dtype=np.int32))

    for ka in range(nkpts):
        vir_dim = None
        for ki in range(nkpts):
            block = ovl[ki, ka]
            if block is None:
                continue
            vir_dim = int(block.shape[1])
            break
        if vir_dim is None:
            raise ValueError(f"Unable to infer virtual dimension for k-point {ka}")
        vir_idx_by_kpt.append(
            np.arange(occ_idx_by_kpt[ka].size, occ_idx_by_kpt[ka].size + vir_dim, dtype=np.int32)
        )

    return occ_idx_by_kpt, vir_idx_by_kpt


def _divide_t2_inplace(t2, e_i, e_a, e_j, e_b):
    if np.issubdtype(t2.dtype, np.complexfloating):
        e_i = cp.asarray(e_i, dtype=np.float64)
        e_a = cp.asarray(e_a, dtype=np.float64)
        e_j = cp.asarray(e_j, dtype=np.float64)
        e_b = cp.asarray(e_b, dtype=np.float64)
        _DIV_T2_COMPLEX128(
            e_i,
            e_a,
            e_j,
            e_b,
            int(t2.shape[3]),
            int(t2.shape[2] * t2.shape[3]),
            int(t2.shape[1] * t2.shape[2] * t2.shape[3]),
            t2,
        )
    else:
        _div_t2_real(
            t2,
            cp.asarray(e_i, dtype=t2.dtype),
            cp.asarray(e_a, dtype=t2.dtype),
            cp.asarray(e_j, dtype=t2.dtype),
            cp.asarray(e_b, dtype=t2.dtype),
        )
    return t2


def load_krmp2_ovl(ovl_record, device: str = "gpu"):
    """Load every `(ki, ka)` ovL block from the FileMp record.

    Args:
        ovl_record: Result of `build_krmp2_ovl(...)`.
        device: `"gpu"` returns a 2D object array of `cupy.ndarray`; `"cpu"`
            returns `numpy.ndarray`.
    """

    if device not in {"gpu", "cpu"}:
        raise ValueError(f"Unsupported device={device!r}")

    nkpts = int(ovl_record["nkpts"])
    blocks = np.empty((nkpts, nkpts), dtype=object)
    for ki in range(nkpts):
        for ka in range(nkpts):
            block = read_krmp2_ovl_kpair(ovl_record, ki, ka)
            block = np.asarray(block, order="C")

            if device == "gpu":
                blocks[ki, ka] = cp.asarray(block)
            else:
                blocks[ki, ka] = block
    return blocks


def build_and_load_krmp2_ovl(
    cderi_record,
    mo_coeff,
    mo_occ=None,
    nocc: Optional[int] = None,
    device: str = "gpu",
    aux_blksize: Optional[int] = None,
    path: Optional[str] = None,
    file_processes=None,
    read_processes=None,
    occ_blksize: Optional[int] = None,
    remove_cderi_after_ovl: bool = True,
    remove_ovl_after_load: bool = True,
) -> dict:
    """Build disk-backed ovL through the existing path, then load it."""

    nkpts = int(cderi_record["nkpts"])
    mo_coeff = _normalize_host_mo_coeff(mo_coeff, nkpts)
    if mo_occ is not None:
        if isinstance(mo_occ, np.ndarray):
            mo_occ = [_to_numpy_block(mo_occ[k]) for k in range(nkpts)]
        else:
            mo_occ = [_to_numpy_block(block) for block in mo_occ]

    kwargs = {
        "aux_blksize": aux_blksize,
        "path": path,
        "occ_blksize": occ_blksize,
        "remove_cderi_after_ovl": bool(remove_cderi_after_ovl),
    }
    if file_processes is not None:
        kwargs["file_processes"] = int(file_processes)
    if read_processes is not None:
        kwargs["read_processes"] = int(read_processes)

    ovl_record = _build_krmp2_ovl_record(
        cderi_record,
        mo_coeff,
        mo_occ=mo_occ,
        nocc=nocc,
        **kwargs,
    )
    ovl = load_krmp2_ovl(ovl_record, device=device)
    if remove_ovl_after_load:
        remove_krmp2_file_record(ovl_record, label="KRMP2 ovL record")
    return {
        "ovl": ovl,
        "ovl_record": ovl_record,
        "nkpts": int(ovl_record["nkpts"]),
        "nocc_by_kpt": np.asarray(ovl_record["nocc_by_kpt"], dtype=np.int32),
        "nvir_by_kpt": np.asarray(ovl_record["nvir_by_kpt"], dtype=np.int32),
    }


def kernel(
    cell,
    kpts,
    mo_energy,
    mo_coeff=None,
    mo_occ=None,
    nocc: Optional[int] = None,
    ovl=None,
    cderi_record=None,
    ovl_record=None,
    verbose=logger.NOTE,
    ovl_aux_blksize: Optional[int] = None,
    ovl_path: Optional[str] = None,
    ovl_file_processes=None,
    ovl_read_processes=None,
    ovl_occ_blksize: Optional[int] = None,
    remove_cderi_after_ovl: bool = True,
    remove_ovl_after_kernel: bool = True,
):
    """Compute KRMP2 correlation energy on GPU from occupied-sliced ovL blocks.

    Args:
        cell: Periodic cell.
        kpts: Explicit k-point array.
        mo_energy: Per-k-point orbital energies.
        mo_coeff: Per-k-point MO coefficients. Required only if `ovl` is not
            provided and the helper has to build ovL from `cderi_record`.
        mo_occ / nocc: Occupation information used to split occupied and virtual
            spaces.
        ovl: Optional 2D object array with `ovl[ki, ka]` shaped as
            `(nocc_ki, nvir_ka, naux_q)`. Blocks may live on host or device.
        cderi_record / ovl_record: Convenience inputs for building/loading the
            disk-backed ovL record consumed by the occupied-sliced kernel.
        remove_cderi_after_ovl: If `True`, remove `cderi_record["file"]` and
            its FileMp side-car directory immediately after a new ovL record is
            built from `cderi_record`.
        remove_ovl_after_kernel: If `True`, remove `ovl_record["file"]` and
            its FileMp side-car directory after the disk-backed KRMP2 kernel
            finishes successfully.
    """

    kpts = np.asarray(kpts, dtype=float)
    nkpts = int(len(kpts))
    log = logger.new_logger(cell, verbose)

    if ovl is None:
        if ovl_record is None:
            if cderi_record is None:
                raise ValueError("ovl, ovl_record, or cderi_record must be provided")
            if mo_coeff is None:
                raise ValueError("mo_coeff is required when ovl has to be built from cderi_record")
            ovl_build_kwargs = {
                "aux_blksize": ovl_aux_blksize,
                "path": ovl_path,
                "occ_blksize": ovl_occ_blksize,
                "remove_cderi_after_ovl": bool(remove_cderi_after_ovl),
            }
            if ovl_file_processes is not None:
                ovl_build_kwargs["file_processes"] = int(ovl_file_processes)
            if ovl_read_processes is not None:
                ovl_build_kwargs["read_processes"] = int(ovl_read_processes)
            ovl_record = _build_krmp2_ovl_record(
                cderi_record,
                _normalize_host_mo_coeff(mo_coeff, nkpts),
                mo_occ=mo_occ,
                nocc=nocc,
                **ovl_build_kwargs,
            )
    else:
        normalized_ovl = np.empty((nkpts, nkpts), dtype=object)
        for ki in range(nkpts):
            for ka in range(nkpts):
                block = ovl[ki, ka]
                if block is None:
                    raise ValueError(f"ovl[{ki}, {ka}] is missing")
                if isinstance(block, (np.ndarray, cp.ndarray)):
                    normalized_ovl[ki, ka] = block
                else:
                    normalized_ovl[ki, ka] = np.asarray(block)
        ovl = normalized_ovl

    mo_energy = _normalize_mo_energy_blocks(mo_energy, nkpts)
    if ovl is None and mo_occ is None and nocc is None:
        nocc_by_kpt = np.asarray(ovl_record["nocc_by_kpt"], dtype=np.int32)
        nvir_by_kpt = np.asarray(ovl_record["nvir_by_kpt"], dtype=np.int32)
        occ_idx_by_kpt = [
            np.arange(int(nocc_by_kpt[k]), dtype=np.int32) for k in range(nkpts)
        ]
        vir_idx_by_kpt = [
            np.arange(
                int(nocc_by_kpt[k]),
                int(nocc_by_kpt[k]) + int(nvir_by_kpt[k]),
                dtype=np.int32,
            )
            for k in range(nkpts)
        ]
    else:
        occ_idx_by_kpt, vir_idx_by_kpt = _resolve_occ_vir_indices(
            mo_energy, mo_occ, nocc, ovl=ovl
        )
        nocc_by_kpt = np.asarray([len(idx) for idx in occ_idx_by_kpt], dtype=np.int32)
        nvir_by_kpt = np.asarray([len(idx) for idx in vir_idx_by_kpt], dtype=np.int32)

    if ovl_record is not None:
        record_nocc = np.asarray(ovl_record["nocc_by_kpt"], dtype=np.int32)
        record_nvir = np.asarray(ovl_record["nvir_by_kpt"], dtype=np.int32)
        if not np.array_equal(nocc_by_kpt, record_nocc):
            raise ValueError(
                f"Occupied partition mismatch between inputs and ovl_record: "
                f"{nocc_by_kpt.tolist()} vs {record_nocc.tolist()}"
            )
        if not np.array_equal(nvir_by_kpt, record_nvir):
            raise ValueError(
                f"Virtual partition mismatch between inputs and ovl_record: "
                f"{nvir_by_kpt.tolist()} vs {record_nvir.tolist()}"
            )

    mo_e_occ = [
        cp.asarray(np.asarray(mo_energy[k])[occ_idx_by_kpt[k]], dtype=np.float64)
        for k in range(nkpts)
    ]
    mo_e_vir = [
        cp.asarray(np.asarray(mo_energy[k])[vir_idx_by_kpt[k]], dtype=np.float64)
        for k in range(nkpts)
    ]

    kconserv = np.asarray(kpts_helper.get_kconserv(cell, kpts), dtype=np.int32)
    emp2 = cp.zeros((), dtype=np.complex128)

    if ovl is None:
        naux_by_pair = np.zeros((nkpts, nkpts), dtype=np.int32)
        for (ki, ka), shape in ovl_record["ovL_shapes"].items():
            naux_by_pair[int(ki), int(ka)] = int(shape[2])
    else:
        naux_by_pair = np.zeros((nkpts, nkpts), dtype=np.int32)
        for ki in range(nkpts):
            for ka in range(nkpts):
                naux_by_pair[ki, ka] = int(ovl[ki, ka].shape[2])

    max_nocc = int(nocc_by_kpt.max())
    max_nvir = int(nvir_by_kpt.max())
    max_naux = int(naux_by_pair.max())

    if ovl_occ_blksize is None:
        lib.free_all_blocks()
        mem_free = int(cp.cuda.runtime.memGetInfo()[0])
        reserve_bytes = max(1024**3, int(0.15 * mem_free))
        usable_bytes = max(0, mem_free - reserve_bytes)

        def _fits_occ_blksize(blksize):
            blksize = int(blksize)
            tensor_bytes = np.dtype(np.complex128).itemsize * (
                blksize * blksize * max_nvir * max_nvir
                + 3 * blksize * max_nvir * max_naux
            )
            return tensor_bytes <= usable_bytes

        occ_blksize = 1
        lo = 1
        hi = max(1, max_nocc)
        while lo <= hi:
            mid = (lo + hi) // 2
            if _fits_occ_blksize(mid):
                occ_blksize = mid
                lo = mid + 1
            else:
                hi = mid - 1

        log.info(
            "KRMP2 auto occ sizing: free=%.3f GB reserve=%.3f GB usable=%.3f GB "
            "max_nocc=%d max_nvir=%d max_naux=%d occ_blksize=%d approx_peak=%.3f MiB",
            mem_free / 1024**3,
            reserve_bytes / 1024**3,
            usable_bytes / 1024**3,
            max_nocc,
            max_nvir,
            max_naux,
            occ_blksize,
            (
                np.dtype(np.complex128).itemsize
                * (
                    occ_blksize * occ_blksize * max_nvir * max_nvir
                    + 3 * occ_blksize * max_nvir * max_naux
                )
            )
            / 1024**2,
        )
    else:
        occ_blksize = int(ovl_occ_blksize)

    occ_blksize = max(1, min(occ_blksize, max_nocc))
    occ_block_counts = np.asarray(
        [
            (int(nocc_by_kpt[k]) + occ_blksize - 1) // occ_blksize
            if int(nocc_by_kpt[k]) > 0
            else 0
            for k in range(nkpts)
        ],
        dtype=np.int32,
    )
    total_kpairs_logical = nkpts * nkpts
    total_kpairs_unique = nkpts * (nkpts + 1) // 2
    total_occ_pair_blocks_logical = int(
        sum(
            int(occ_block_counts[ki]) * int(occ_block_counts[kj])
            for ki in range(nkpts)
            for kj in range(nkpts)
        )
    )
    total_occ_pair_blocks_unique = int(
        sum(
            int(occ_block_counts[ki]) * int(occ_block_counts[kj])
            for ki in range(nkpts)
            for kj in range(ki, nkpts)
        )
    )
    total_ka_tasks_logical = total_occ_pair_blocks_logical * nkpts
    total_ka_tasks_unique = total_occ_pair_blocks_unique * nkpts

    read_host_buffers = None
    pool_read = None
    ovl_file = None
    ovl_datasets = None
    if ovl is None:
        read_host_buffers = [
            [
                lib.empty(
                    (occ_blksize, max_nvir, max_naux),
                    dtype=np.complex128,
                    type=lib.MemoryTypeHost,
                )
                for _ in range(4)
            ]
            for _ in range(2)
        ]
        for slot_buffers in read_host_buffers:
            for host_buf in slot_buffers:
                if not is_pinned(host_buf):
                    raise RuntimeError("KRMP2 ovL slice read buffers must be pinned")

        ovl_file = lib.FileMp(ovl_record["file"], "r")
        ovl_datasets = np.empty((nkpts, nkpts), dtype=object)
        for ki in range(nkpts):
            for ka in range(nkpts):
                ovl_datasets[ki, ka] = ovl_file[ovl_record["ovL"][ki, ka]]
        pool_read = Pool(processes=max(1, int(ovl_record.get("read_processes", 1))))

        def queue_read_slot(slot_id, task):
            ki, kj, si, si_len, sj, sj_len, ka, kb, na, nb, naux_direct, naux_exchange = task
            if naux_direct != int(naux_by_pair[kj, kb]):
                raise ValueError(
                    f"ovl[{ki},{ka}] and ovl[{kj},{kb}] have inconsistent naux: "
                    f"{naux_direct} vs {int(naux_by_pair[kj, kb])}"
                )
            if naux_exchange != int(naux_by_pair[kj, ka]):
                raise ValueError(
                    f"ovl[{ki},{kb}] and ovl[{kj},{ka}] have inconsistent naux: "
                    f"{naux_exchange} vs {int(naux_by_pair[kj, ka])}"
                )
            slot_buffers = read_host_buffers[slot_id]
            ia_h = lib.empty_from_buf(
                slot_buffers[0], (si_len, na, naux_direct), np.complex128
            )
            jb_h = lib.empty_from_buf(
                slot_buffers[1], (sj_len, nb, naux_direct), np.complex128
            )
            ib_h = lib.empty_from_buf(
                slot_buffers[2], (si_len, nb, naux_exchange), np.complex128
            )
            ja_h = lib.empty_from_buf(
                slot_buffers[3], (sj_len, na, naux_exchange), np.complex128
            )

            futures = (
                ovl_datasets[ki, ka].getitem(np.s_[si, :, :], pool=pool_read, buf=ia_h),
                ovl_datasets[kj, kb].getitem(np.s_[sj, :, :], pool=pool_read, buf=jb_h),
                ovl_datasets[ki, kb].getitem(np.s_[si, :, :], pool=pool_read, buf=ib_h),
                ovl_datasets[kj, ka].getitem(np.s_[sj, :, :], pool=pool_read, buf=ja_h),
            )
            return {
                "task": task,
                "hosts": (ia_h, jb_h, ib_h, ja_h),
                "futures": futures,
            }

    def build_t2_and_accumulate_direct(task, iaL, jbL):
        ki, kj, si, _si_len, sj, _sj_len, ka, kb, _na, _nb, naux_direct, _naux_exchange = task
        pair_weight = 2.0 if ki != kj else 1.0
        t2_iajb = lib.gemm(
            iaL.reshape(-1, naux_direct),
            jbL.reshape(-1, naux_direct),
            transb="T",
            buf=t2_buffer,
        ).reshape(iaL.shape[0], iaL.shape[1], jbL.shape[0], jbL.shape[1])
        t2_iajb /= nkpts
        cp.conjugate(t2_iajb, out=t2_iajb)
        _divide_t2_inplace(
            t2_iajb,
            mo_e_occ[ki][si],
            mo_e_vir[ka],
            mo_e_occ[kj][sj],
            mo_e_vir[kb],
        )

        tmp = lib.contraction(
            "iajb",
            t2_iajb,
            "jbL",
            jbL,
            "iaL",
            buf=tmp_ovl_buffer,
            alpha=(2.0 * pair_weight) / nkpts,
        )
        emp2[...] += tmp.ravel().dot(iaL.ravel())
        return t2_iajb

    def accumulate_exchange(task, t2_iajb, ibL, jaL):
        ki = int(task[0])
        kj = int(task[1])
        pair_weight = 2.0 if ki != kj else 1.0
        tmp = lib.contraction(
            "iajb",
            t2_iajb,
            "jaL",
            jaL,
            "ibL",
            buf=tmp_ovl_buffer,
            alpha=(-1.0 * pair_weight) / nkpts,
        )
        emp2[...] += tmp.ravel().dot(ibL.ravel())

    gpu_ovl_buffers = [
        cp.empty((occ_blksize, max_nvir, max_naux), dtype=np.complex128)
        for _ in range(2)
    ]
    tmp_ovl_buffer = cp.empty((occ_blksize, max_nvir, max_naux), dtype=np.complex128)
    t2_buffer = cp.empty(
        (max(1, occ_blksize * max_nvir * occ_blksize * max_nvir),),
        dtype=np.complex128,
    )

    log.info(
        "KRMP2 GPU kernel start: nkpts=%d ovl_source=%s occ_sliced=True occ_blksize=%d tensor_layout=ovov",
        nkpts,
        "disk" if ovl is None else "memory",
        occ_blksize,
    )
    log.info(
        "KRMP2 loop topology: logical_kpairs=%d unique_kpairs=%d "
        "logical_occ_pair_blocks=%d unique_occ_pair_blocks=%d "
        "logical_ka_tasks=%d unique_ka_tasks=%d "
        "occ_blocks_by_kpt=%s nocc_by_kpt=%s nvir_by_kpt=%s",
        total_kpairs_logical,
        total_kpairs_unique,
        total_occ_pair_blocks_logical,
        total_occ_pair_blocks_unique,
        total_ka_tasks_logical,
        total_ka_tasks_unique,
        occ_block_counts.tolist(),
        nocc_by_kpt.tolist(),
        nvir_by_kpt.tolist(),
    )

    progress = {
        "current_kpair": None,
        "current_occ_block": None,
        "kpair_count": 0,
        "occ_block_count": 0,
        "ka_task_count": 0,
    }

    def log_loop_progress(task, path):
        ki, kj, si, si_len, sj, sj_len, ka, kb, na, nb, naux_direct, naux_exchange = task
        kpair_key = (ki, kj)
        occ_block_key = (ki, kj, si.start, si.stop, sj.start, sj.stop)
        pair_weight = 2 if ki != kj else 1

        if progress["current_kpair"] != kpair_key:
            progress["current_kpair"] = kpair_key
            progress["current_occ_block"] = None
            progress["kpair_count"] += 1
            log.info(
                "KRMP2 k-pair %d/%d path=%s ki=%d kj=%d pair_weight=%d "
                "nocc_i=%d nocc_j=%d occ_i_blocks=%d occ_j_blocks=%d",
                progress["kpair_count"],
                total_kpairs_unique,
                path,
                ki,
                kj,
                pair_weight,
                int(nocc_by_kpt[ki]),
                int(nocc_by_kpt[kj]),
                int(occ_block_counts[ki]),
                int(occ_block_counts[kj]),
            )

        if progress["current_occ_block"] != occ_block_key:
            progress["current_occ_block"] = occ_block_key
            progress["occ_block_count"] += 1
            log.debug(
                "KRMP2 occ-block %d/%d path=%s ki=%d kj=%d occ_i=[%d:%d) occ_j=[%d:%d) "
                "si_len=%d sj_len=%d",
                progress["occ_block_count"],
                total_occ_pair_blocks_unique,
                path,
                ki,
                kj,
                si.start,
                si.stop,
                sj.start,
                sj.stop,
                si_len,
                sj_len,
            )

        progress["ka_task_count"] += 1
        log.debug1(
            "KRMP2 ka-task %d/%d path=%s ki=%d kj=%d ka=%d kb=%d occ_i=[%d:%d) occ_j=[%d:%d) "
            "pair_weight=%d nvir_ka=%d nvir_kb=%d naux_direct=%d naux_exchange=%d",
            progress["ka_task_count"],
            total_ka_tasks_unique,
            path,
            ki,
            kj,
            ka,
            kb,
            si.start,
            si.stop,
            sj.start,
            sj.stop,
            pair_weight,
            na,
            nb,
            naux_direct,
            naux_exchange,
        )

    try:
        if ovl is None:
            tasks = (
                (
                    ki,
                    kj,
                    slice(i0, min(i0 + occ_blksize, int(nocc_by_kpt[ki]))),
                    min(i0 + occ_blksize, int(nocc_by_kpt[ki])) - i0,
                    slice(j0, min(j0 + occ_blksize, int(nocc_by_kpt[kj]))),
                    min(j0 + occ_blksize, int(nocc_by_kpt[kj])) - j0,
                    ka,
                    int(kconserv[ki, ka, kj]),
                    int(nvir_by_kpt[ka]),
                    int(nvir_by_kpt[int(kconserv[ki, ka, kj])]),
                    int(naux_by_pair[ki, ka]),
                    int(naux_by_pair[ki, int(kconserv[ki, ka, kj])]),
                )
                for ki in range(nkpts)
                for kj in range(ki, nkpts)
                for i0 in range(0, int(nocc_by_kpt[ki]), occ_blksize)
                for j0 in range(0, int(nocc_by_kpt[kj]), occ_blksize)
                for ka in range(nkpts)
            )
            prefetched = [None, None]
            next_task = next(tasks, None)
            active_slot = 0
            idle_slot = 1
            host_slot_events = [None, None]

            def wait_host_slot(slot_id):
                event = host_slot_events[slot_id]
                if event is not None:
                    event.synchronize()
                    host_slot_events[slot_id] = None

            def mark_host_slot_done(slot_id):
                event = cp.cuda.Event()
                event.record()
                host_slot_events[slot_id] = event

            if next_task is not None:
                wait_host_slot(active_slot)
                prefetched[active_slot] = queue_read_slot(active_slot, next_task)
                next_task = next(tasks, None)

            while prefetched[active_slot] is not None:
                if next_task is not None:
                    wait_host_slot(idle_slot)
                    prefetched[idle_slot] = queue_read_slot(idle_slot, next_task)
                    next_task = next(tasks, None)

                active = prefetched[active_slot]
                task = active["task"]
                log_loop_progress(task, "disk")
                for future in active["futures"]:
                    future.wait()

                ia_h, jb_h, ib_h, ja_h = active["hosts"]
                iaL = lib.empty_from_buf(gpu_ovl_buffers[0], ia_h.shape, np.complex128)
                jbL = lib.empty_from_buf(gpu_ovl_buffers[1], jb_h.shape, np.complex128)
                iaL.set(ia_h)
                jbL.set(jb_h)
                t2_iajb = build_t2_and_accumulate_direct(task, iaL, jbL)

                ibL = lib.empty_from_buf(gpu_ovl_buffers[0], ib_h.shape, np.complex128)
                jaL = lib.empty_from_buf(gpu_ovl_buffers[1], ja_h.shape, np.complex128)
                ibL.set(ib_h)
                jaL.set(ja_h)
                accumulate_exchange(task, t2_iajb, ibL, jaL)
                mark_host_slot_done(active_slot)

                prefetched[active_slot] = None
                active_slot, idle_slot = idle_slot, active_slot

            for event in host_slot_events:
                if event is not None:
                    event.synchronize()
        else:
            for ki in range(nkpts):
                ni = int(nocc_by_kpt[ki])
                for kj in range(ki, nkpts):
                    nj = int(nocc_by_kpt[kj])
                    for i0 in range(0, ni, occ_blksize):
                        i1 = min(i0 + occ_blksize, ni)
                        si = slice(i0, i1)
                        si_len = i1 - i0
                        for j0 in range(0, nj, occ_blksize):
                            j1 = min(j0 + occ_blksize, nj)
                            sj = slice(j0, j1)
                            sj_len = j1 - j0
                            for ka in range(nkpts):
                                kb = int(kconserv[ki, ka, kj])
                                ia_src = ovl[ki, ka][si, :, :]
                                jb_src = ovl[kj, kb][sj, :, :]
                                ib_src = ovl[ki, kb][si, :, :]
                                ja_src = ovl[kj, ka][sj, :, :]

                                if ia_src.shape[2] != jb_src.shape[2]:
                                    raise ValueError(
                                        f"ovl[{ki},{ka}] and ovl[{kj},{kb}] have inconsistent naux: "
                                        f"{ia_src.shape[2]} vs {jb_src.shape[2]}"
                                    )
                                if ib_src.shape[2] != ja_src.shape[2]:
                                    raise ValueError(
                                        f"ovl[{ki},{kb}] and ovl[{kj},{ka}] have inconsistent naux: "
                                        f"{ib_src.shape[2]} vs {ja_src.shape[2]}"
                                    )

                                if isinstance(ia_src, cp.ndarray):
                                    iaL = ia_src
                                else:
                                    iaL = lib.empty_from_buf(
                                        gpu_ovl_buffers[0], ia_src.shape, np.complex128
                                    )
                                    iaL.set(np.asarray(ia_src, dtype=np.complex128, order="C"))
                                if isinstance(jb_src, cp.ndarray):
                                    jbL = jb_src
                                else:
                                    jbL = lib.empty_from_buf(
                                        gpu_ovl_buffers[1], jb_src.shape, np.complex128
                                    )
                                    jbL.set(np.asarray(jb_src, dtype=np.complex128, order="C"))

                                task = (
                                    ki,
                                    kj,
                                    si,
                                    si_len,
                                    sj,
                                    sj_len,
                                    ka,
                                    kb,
                                    int(ia_src.shape[1]),
                                    int(jb_src.shape[1]),
                                    int(ia_src.shape[2]),
                                    int(ib_src.shape[2]),
                                )
                                log_loop_progress(task, "memory")
                                t2_iajb = build_t2_and_accumulate_direct(task, iaL, jbL)

                                if isinstance(ib_src, cp.ndarray):
                                    ibL = ib_src
                                else:
                                    ibL = lib.empty_from_buf(
                                        gpu_ovl_buffers[0], ib_src.shape, np.complex128
                                    )
                                    ibL.set(np.asarray(ib_src, dtype=np.complex128, order="C"))
                                if isinstance(ja_src, cp.ndarray):
                                    jaL = ja_src
                                else:
                                    jaL = lib.empty_from_buf(
                                        gpu_ovl_buffers[1], ja_src.shape, np.complex128
                                    )
                                    jaL.set(np.asarray(ja_src, dtype=np.complex128, order="C"))

                                accumulate_exchange(task, t2_iajb, ibL, jaL)
    finally:
        if pool_read is not None:
            pool_read.close()
            pool_read.join()
        if ovl_file is not None:
            ovl_file.close()
        lib.free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    emp2_per_unitcell = float(emp2.real.get()) / nkpts
    log.info(
        "KRMP2 GPU kernel done: processed_unique_kpairs=%d/%d "
        "processed_unique_occ_pair_blocks=%d/%d processed_unique_ka_tasks=%d/%d "
        "logical_kpairs=%d logical_occ_pair_blocks=%d logical_ka_tasks=%d "
        "emp2_per_unitcell=%.12f",
        progress["kpair_count"],
        total_kpairs_unique,
        progress["occ_block_count"],
        total_occ_pair_blocks_unique,
        progress["ka_task_count"],
        total_ka_tasks_unique,
        total_kpairs_logical,
        total_occ_pair_blocks_logical,
        total_ka_tasks_logical,
        emp2_per_unitcell,
    )

    if remove_ovl_after_kernel and ovl is None and ovl_record is not None:
        remove_krmp2_file_record(ovl_record, log=log, label="KRMP2 ovL record")

    return emp2_per_unitcell
