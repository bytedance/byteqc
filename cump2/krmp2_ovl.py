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

"""KRMP2 ovL utilities built on top of the out-of-core k-point CDERI record."""

from __future__ import annotations

import gc
import os
import tempfile
from multiprocessing import Pool
from typing import Optional

import cupy as cp
import numpy as np
from pyscf.pbc.tools.k2gamma import (
    double_translation_indices,
    translation_vectors_for_kmesh,
)

from byteqc import lib
from byteqc.lib.utils import is_pinned
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.df import rsdf_builder
from gpu4pyscf.pbc.lib.kpts_helper import conj_images_in_bvk_cell

WRITE_PROCESSSES = 4
READ_PROCESSSES = 4


def build_krmp2_ovl(
    cderi_record,
    mo_coeff,
    mo_occ=None,
    nocc: Optional[int] = None,
    aux_blksize: Optional[int] = None,
    path: Optional[str] = None,
    file_processes: int = WRITE_PROCESSSES,
    read_processes: int = READ_PROCESSSES,
    occ_blksize: Optional[int] = None,
) -> dict:
    """Transform the current KRMP2 CDERI record to disk-backed ovL datasets.

    The returned record stores one FileMp dataset per `(ki, ka)` pair, with
    logical shape `(nocc_ki, nvir_ka, naux_q)`.
    """

    ao_cell = cderi_record.get("ao_cell", cderi_record["cell"])
    int3c_cell = cderi_record.get("int3c_cell", cderi_record["cell"])
    nkpts = int(cderi_record["nkpts"])
    nao = int(cderi_record["nao"])
    log = logger.new_logger(ao_cell, ao_cell.verbose)
    file_processes = max(1, int(file_processes))
    read_processes = max(1, int(read_processes))

    if path is None:
        path = cderi_record.get("path")
    if path is None:
        path = tempfile.mkdtemp(dir=os.environ.get("PYSCF_TMPDIR"))
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "ovl.h5")

    if isinstance(mo_coeff, np.ndarray):
        if mo_coeff.ndim != 3 or mo_coeff.shape[0] != nkpts:
            raise ValueError(
                f"mo_coeff must have shape (nkpts, nao, nmo); got {mo_coeff.shape}"
            )
        mo_coeff = [np.asarray(mo_coeff[k]) for k in range(nkpts)]
    else:
        mo_coeff = [np.asarray(x) for x in mo_coeff]
        if len(mo_coeff) != nkpts:
            raise ValueError(
                f"mo_coeff must contain {nkpts} k-point blocks; got {len(mo_coeff)}"
            )

    occ_idx_by_kpt = []
    vir_idx_by_kpt = []
    nocc_by_kpt = []
    nvir_by_kpt = []
    for k, coeff_k in enumerate(mo_coeff):
        if coeff_k.ndim != 2 or coeff_k.shape[0] != nao:
            raise ValueError(
                f"mo_coeff[{k}] must have shape ({nao}, nmo); got {coeff_k.shape}"
            )

        nmo_k = int(coeff_k.shape[1])
        if mo_occ is not None:
            occ_k = np.asarray(mo_occ[k])
            if occ_k.shape != (nmo_k,):
                raise ValueError(
                    f"mo_occ[{k}] must have shape ({nmo_k},); got {occ_k.shape}"
                )
            occ_idx = np.where(occ_k > 0)[0]
            vir_idx = np.where(occ_k == 0)[0]
        elif nocc is not None:
            if not 0 < int(nocc) < nmo_k + 1:
                raise ValueError(
                    f"Invalid nocc={nocc} for mo_coeff[{k}] with nmo={nmo_k}"
                )
            occ_idx = np.arange(int(nocc))
            vir_idx = np.arange(int(nocc), nmo_k)
        else:
            raise ValueError("Either mo_occ or nocc must be provided")

        occ_idx_by_kpt.append(np.asarray(occ_idx, dtype=np.int32))
        vir_idx_by_kpt.append(np.asarray(vir_idx, dtype=np.int32))
        nocc_by_kpt.append(int(len(occ_idx)))
        nvir_by_kpt.append(int(len(vir_idx)))

    kk_conserv = double_translation_indices(
        np.asarray(cderi_record["kmesh"], dtype=np.int32)
    )
    conj_mapping = np.asarray(
        conj_images_in_bvk_cell(np.asarray(cderi_record["kmesh"], dtype=np.int32)),
        dtype=np.int32,
    )
    phases = np.asarray(
        translation_vectors_for_kmesh(
            int3c_cell, np.asarray(cderi_record["kmesh"], dtype=np.int32), True
        ),
        dtype=float,
    ).dot(np.asarray(cderi_record["kpts"]).T)
    expLk = cp.exp(1j * cp.asarray(phases))
    pair_address = np.asarray(cderi_record["cderi_idx"][0], dtype=np.int32)
    nL = int(expLk.shape[0])
    max_nocc = max(nocc_by_kpt) if nocc_by_kpt else 1
    max_nvir = max(nvir_by_kpt) if nvir_by_kpt else 1
    write_slots = 2
    read_slots = 2
    d2h_slots = 2

    coeff_occ_gpu = [
        cp.asarray(
            np.asarray(mo_coeff[k][:, occ_idx_by_kpt[k]], dtype=np.complex128, order="C").conj()
        )
        for k in range(nkpts)
    ]
    coeff_vir_gpu = [
        cp.asarray(
            np.asarray(mo_coeff[k][:, vir_idx_by_kpt[k]], dtype=np.complex128, order="C")
        )
        for k in range(nkpts)
    ]

    q_infos = []
    ovl_names = {}
    ovl_shapes = {}
    ovl_aux_blksizes = {}
    max_aux_blksize = 0
    max_pair_cols = 0
    max_pairs_per_q = 0
    total_aux_batches = 0
    auto_aux_blksize = None
    aux_bytes_per_slice = None
    gpu_free_bytes = None
    gpu_reserve_bytes = None
    gpu_usable_bytes = None

    if aux_blksize is None:
        complex128_nbytes = np.dtype(np.complex128).itemsize
        aux_bytes_per_slice = complex128_nbytes * (
            nL * nao * nao
            + nkpts * nao * nao
            + nao * nao
            + nao * max(max_nocc, max_nvir)
            + d2h_slots * max_nocc * max_nvir
        )
        gpu_free_bytes = int(cp.cuda.runtime.memGetInfo()[0])
        gpu_reserve_bytes = max(1024**3, int(0.10 * gpu_free_bytes))
        gpu_usable_bytes = max(0, gpu_free_bytes - gpu_reserve_bytes)
        auto_aux_blksize = max(1, gpu_usable_bytes // max(1, aux_bytes_per_slice))
        log.info(
            "KRMP2 ovL auto aux sizing: free=%.3f GB reserve=%.3f GB usable=%.3f GB bytes_per_aux=%.3f MiB auto_aux_blksize=%d",
            gpu_free_bytes / 1024**3,
            gpu_reserve_bytes / 1024**3,
            gpu_usable_bytes / 1024**3,
            aux_bytes_per_slice / 1024**2,
            auto_aux_blksize,
        )

    cderi_file = lib.FileMp(cderi_record["file"], "r")
    try:
        for k_aux in range(nkpts):
            stored_k_aux = k_aux if k_aux in cderi_record["cderi"] else int(conj_mapping[k_aux])
            dataset_name = cderi_record["cderi"][stored_k_aux]
            dataset = cderi_file[dataset_name]
            naux_negative = int(cderi_record["naoaux_negative_by_kpt"].get(stored_k_aux, 0))
            naux_positive = int(dataset.shape[0] - naux_negative)
            stored_kj_idx = np.where(kk_conserv == stored_k_aux)[1]
            ki_idx, ka_idx = np.where(kk_conserv == k_aux)

            if aux_blksize is None:
                q_aux_blksize = max(1, min(naux_positive, int(auto_aux_blksize)))
            else:
                q_aux_blksize = max(1, min(int(aux_blksize), naux_positive))

            q_infos.append(
                (
                    int(k_aux),
                    int(stored_k_aux),
                    dataset_name,
                    naux_positive,
                    stored_kj_idx,
                    ki_idx,
                    ka_idx,
                    q_aux_blksize,
                    int(dataset.shape[1]),
                )
            )
            total_aux_batches += (naux_positive + q_aux_blksize - 1) // q_aux_blksize
            max_aux_blksize = max(max_aux_blksize, int(q_aux_blksize))
            max_pair_cols = max(max_pair_cols, int(dataset.shape[1]))
            max_pairs_per_q = max(max_pairs_per_q, int(len(ki_idx)))

            for ki, ka in zip(ki_idx, ka_idx):
                nocc_ki = nocc_by_kpt[ki]
                nvir_ka = nvir_by_kpt[ka]
                dataset_name = f"ovL_{int(ki)}_{int(ka)}"
                ovl_names[int(ki), int(ka)] = dataset_name
                ovl_shapes[int(ki), int(ka)] = (nocc_ki, nvir_ka, naux_positive)
                ovl_aux_blksizes[int(ki), int(ka)] = int(q_aux_blksize)
    finally:
        cderi_file.close()

    read_host_buffers = [
        lib.empty(
            (max_aux_blksize, max_pair_cols),
            dtype=np.complex128,
            type=lib.MemoryTypeHost,
        )
        for _ in range(read_slots)
    ]
    for host_buf in read_host_buffers:
        if not is_pinned(host_buf):
            raise RuntimeError("KRMP2 ovL FileMp read buffers must be pinned")

    write_host_buffers = [
        [
            lib.empty(
                (max_nocc, max_nvir, max_aux_blksize),
                dtype=np.complex128,
                type=lib.MemoryTypeHost,
            )
            for _ in range(max_pairs_per_q)
        ]
        for _ in range(write_slots)
    ]
    for slot_buffers in write_host_buffers:
        for host_buf in slot_buffers:
            if not is_pinned(host_buf):
                raise RuntimeError("KRMP2 ovL FileMp write buffers must be pinned")

    unpack_buf_d = cp.empty(
        (max(1, nL * max_aux_blksize * nao * nao),), dtype=np.complex128
    )
    unpack_out_d = cp.empty(
        (max(1, nkpts * max_aux_blksize * nao * nao),), dtype=np.complex128
    )
    transpose_pair_d = cp.empty((max_aux_blksize, nao, nao), dtype=np.complex128)
    stage1_buf_d = cp.empty(
        (max(1, max_aux_blksize * nao * max(max_nocc, max_nvir)),),
        dtype=np.complex128,
    )
    ovl_buf_slots_d = [
        cp.empty(
            (max(1, max_nocc * max_nvir * max_aux_blksize),),
            dtype=np.complex128,
        )
        for _ in range(d2h_slots)
    ]
    d2h_stream = cp.cuda.Stream(non_blocking=True)

    cderi_file = None
    ovl_file = None
    pool_read = None
    pool_write = None
    pending_writes = [None] * write_slots
    aux_batch_id = 0
    try:
        cderi_file = lib.FileMp(cderi_record["file"], "r")
        ovl_file = lib.FileMp(file_path, "w")
        ovl_datasets = {}
        for ki, ka in sorted(ovl_names):
            nocc_ki, nvir_ka, naux_q = ovl_shapes[ki, ka]
            local_occ_blksize = int(occ_blksize or nocc_ki)
            local_occ_blksize = max(1, min(local_occ_blksize, nocc_ki))
            occ_blocks = [local_occ_blksize] * (nocc_ki // local_occ_blksize)
            if sum(occ_blocks) < nocc_ki:
                occ_blocks.append(nocc_ki - sum(occ_blocks))

            q_aux_blksize = int(ovl_aux_blksizes[ki, ka])
            aux_blocks = [q_aux_blksize] * (naux_q // q_aux_blksize)
            if sum(aux_blocks) < naux_q:
                aux_blocks.append(naux_q - sum(aux_blocks))

            dataset = ovl_file.create_dataset(
                ovl_names[ki, ka],
                (nocc_ki, nvir_ka, naux_q),
                dtype=np.complex128,
                blksizes=(tuple(occ_blocks), nvir_ka, tuple(aux_blocks)),
            )
            ovl_datasets[ki, ka] = dataset

        pool_read = Pool(processes=read_processes)
        pool_write = Pool(processes=file_processes)
        total_q_blocks = len(q_infos)
        finished_aux_batches = 0
        log.info(
            "KRMP2 ovL build start: q_blocks=%d aux_batches=%d write_processes=%d read_processes=%d",
            total_q_blocks,
            total_aux_batches,
            file_processes,
            read_processes,
        )

        for q_id, (
            k_aux,
            stored_k_aux,
            dataset_name,
            naux_positive,
            stored_kj_idx,
            ki_idx,
            ka_idx,
            q_aux_blksize,
            pair_cols,
        ) in enumerate(q_infos, start=1):
            dataset = cderi_file[dataset_name]
            q_batch_total = (naux_positive + q_aux_blksize - 1) // q_aux_blksize

            log.info(
                "KRMP2 ovL q-block %d/%d: q=%d stored_q=%d naux=%d aux_blksize=%d aux_batches=%d adapted_pairs=%d remaining_q=%d remaining_aux_batches=%d",
                q_id,
                total_q_blocks,
                k_aux,
                stored_k_aux,
                naux_positive,
                q_aux_blksize,
                q_batch_total,
                len(ki_idx),
                total_q_blocks - q_id,
                total_aux_batches - finished_aux_batches,
            )

            slice_ranges = [
                (a0, min(naux_positive, a0 + q_aux_blksize))
                for a0 in range(0, naux_positive, q_aux_blksize)
            ]
            read_futures = [None] * read_slots
            read_slot_events = [None] * read_slots

            def wait_read_slot(slot_id):
                event = read_slot_events[slot_id]
                if event is not None:
                    event.synchronize()
                    read_slot_events[slot_id] = None

            if slice_ranges:
                a0, a1 = slice_ranges[0]
                aux_len = int(a1 - a0)
                wait_read_slot(0)
                cderi_slice = lib.empty_from_buf(
                    read_host_buffers[0],
                    (aux_len, pair_cols),
                    dtype=dataset.dtype,
                    order="C",
                    type=lib.MemoryTypeHost,
                )
                if not is_pinned(cderi_slice):
                    raise RuntimeError("KRMP2 ovL slice loader requires pinned host memory")
                read_futures[0] = dataset.getitem(np.s_[a0:a1, :], pool=pool_read, buf=cderi_slice)

            for q_batch_id, (a0, a1) in enumerate(slice_ranges):
                aux_len = int(a1 - a0)
                slot = aux_batch_id & 1
                aux_batch_id += 1

                read_slot = q_batch_id & 1
                future = read_futures[read_slot]
                if future is None:
                    raise RuntimeError("KRMP2 ovL read pipeline lost the current aux slice")
                future.wait()
                cderi_slice = future.view(np.ndarray)

                if q_batch_id + 1 < len(slice_ranges):
                    na0, na1 = slice_ranges[q_batch_id + 1]
                    next_aux_len = int(na1 - na0)
                    next_read_slot = read_slot ^ 1
                    wait_read_slot(next_read_slot)
                    next_cderi_slice = lib.empty_from_buf(
                        read_host_buffers[next_read_slot],
                        (next_aux_len, pair_cols),
                        dtype=dataset.dtype,
                        order="C",
                        type=lib.MemoryTypeHost,
                    )
                    if not is_pinned(next_cderi_slice):
                        raise RuntimeError("KRMP2 ovL slice loader requires pinned host memory")
                    read_futures[next_read_slot] = dataset.getitem(
                        np.s_[na0:na1, :], pool=pool_read, buf=next_cderi_slice
                    )
                else:
                    read_futures[read_slot] = None

                unpack_buf = lib.empty_from_buf(
                    unpack_buf_d,
                    (max(1, nL * aux_len * nao * nao),),
                    dtype=np.complex128,
                )
                unpack_out = lib.empty_from_buf(
                    unpack_out_d,
                    (max(1, nkpts * aux_len * nao * nao),),
                    dtype=np.complex128,
                )
                unpacked = rsdf_builder._unpack_cderi_v2(
                    cderi_slice,
                    pair_address,
                    stored_kj_idx,
                    conj_mapping,
                    expLk,
                    nao,
                    axis=0,
                    buf=unpack_buf,
                    out=unpack_out,
                )
                read_done = cp.cuda.Event()
                read_done.record()
                read_slot_events[read_slot] = read_done

                if pending_writes[slot] is not None:
                    for waits in pending_writes[slot]:
                        for wait in waits:
                            wait.wait()
                    pending_writes[slot] = None

                slice_write_args = []
                d2h_events = [None] * d2h_slots
                for pair_id, (ki, ka) in enumerate(zip(ki_idx, ka_idx)):
                    nocc_ki = nocc_by_kpt[ki]
                    nvir_ka = nvir_by_kpt[ka]
                    coeff_occ_d = coeff_occ_gpu[ki]
                    coeff_vir_d = coeff_vir_gpu[ka]
                    d2h_slot = pair_id % d2h_slots
                    if d2h_events[d2h_slot] is not None:
                        d2h_events[d2h_slot].synchronize()
                        d2h_events[d2h_slot] = None

                    if k_aux == stored_k_aux:
                        ao_pair_d = unpacked[ki]
                    else:
                        ao_pair_d = lib.empty_from_buf(
                            transpose_pair_d,
                            (aux_len, nao, nao),
                            dtype=np.complex128,
                        )
                        ao_pair_d[:] = unpacked[ka].transpose(0, 2, 1)
                        cp.conjugate(ao_pair_d, out=ao_pair_d)

                    if nocc_ki <= nvir_ka:
                        tmp_d = lib.contraction(
                            "Lij",
                            ao_pair_d,
                            "io",
                            coeff_occ_d,
                            "Loj",
                            buf=stage1_buf_d,
                        )
                        ovl_d = lib.empty_from_buf(
                            ovl_buf_slots_d[d2h_slot],
                            (nocc_ki, nvir_ka, aux_len),
                            dtype=np.complex128,
                        )
                        lib.contraction("Loj", tmp_d, "jv", coeff_vir_d, "ovL", ovl_d)
                    else:
                        tmp_d = lib.contraction(
                            "Lij",
                            ao_pair_d,
                            "jv",
                            coeff_vir_d,
                            "vLi",
                            buf=stage1_buf_d,
                        )
                        ovl_d = lib.empty_from_buf(
                            ovl_buf_slots_d[d2h_slot],
                            (nocc_ki, nvir_ka, aux_len),
                            dtype=np.complex128,
                        )
                        lib.contraction("vLi", tmp_d, "io", coeff_occ_d, "ovL", ovl_d)

                    ovl_h = lib.empty_from_buf(
                        write_host_buffers[slot][pair_id],
                        (nocc_ki, nvir_ka, aux_len),
                        dtype=np.complex128,
                        type=lib.MemoryTypeHost,
                    )
                    if not is_pinned(ovl_h):
                        raise RuntimeError("KRMP2 ovL FileMp write slices must stay in pinned host memory")

                    compute_done = cp.cuda.Event()
                    compute_done.record()
                    with d2h_stream:
                        d2h_stream.wait_event(compute_done)
                        ovl_d.get(out=ovl_h, blocking=False)
                        d2h_done = cp.cuda.Event()
                        d2h_done.record()
                    d2h_events[d2h_slot] = d2h_done
                    slice_write_args.append((ovl_datasets[ki, ka], ovl_h))
                    tmp_d = None
                    ovl_d = None

                for d2h_done in d2h_events:
                    if d2h_done is not None:
                        d2h_done.synchronize()

                pending_writes[slot] = [
                    dataset.setitem(np.s_[:, :, a0:a1], ovl_h, pool=pool_write)
                    for dataset, ovl_h in slice_write_args
                ]

                del cderi_slice, unpacked
            for event in read_slot_events:
                if event is not None:
                    event.synchronize()
            finished_aux_batches += q_batch_total

        for waits in pending_writes:
            if waits is not None:
                for wait_list in waits:
                    for wait in wait_list:
                        wait.wait()
    except Exception:
        if pool_write is not None:
            pool_write.terminate()
            pool_write.join()
        if pool_read is not None:
            pool_read.terminate()
            pool_read.join()
        if ovl_file is not None:
            ovl_file.close()
        if cderi_file is not None:
            cderi_file.close()
        raise
    else:
        if pool_write is not None:
            pool_write.close()
            pool_write.join()
        if pool_read is not None:
            pool_read.close()
            pool_read.join()
        ovl_file.close()
        cderi_file.close()
    finally:
        cp.cuda.get_current_stream().synchronize()
        lib.free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()

    return {
        "ovL": ovl_names,
        "ovL_shapes": ovl_shapes,
        "file": file_path,
        "path": path,
        "nkpts": nkpts,
        "kmesh": np.asarray(cderi_record["kmesh"], dtype=np.int32),
        "nocc_by_kpt": np.asarray(nocc_by_kpt, dtype=np.int32),
        "nvir_by_kpt": np.asarray(nvir_by_kpt, dtype=np.int32),
        "file_processes": int(file_processes),
        "read_processes": int(read_processes),
    }


def read_krmp2_ovl_kpair(ovl_record, ki: int, ka: int, occ_slice=None) -> np.ndarray:
    """Read one `(ki, ka)` ovL block from the disk-backed record."""

    dataset_name = ovl_record["ovL"][int(ki), int(ka)]
    nocc_ki, nvir_ka, naux_q = ovl_record["ovL_shapes"][int(ki), int(ka)]
    if occ_slice is None:
        read_key = np.s_[:, :, :]
        nocc_read = nocc_ki
    elif isinstance(occ_slice, slice):
        start, stop, step = occ_slice.indices(nocc_ki)
        nocc_read = len(range(start, stop, step))
        read_key = np.s_[occ_slice, :, :]
    else:
        occ_idx = np.arange(nocc_ki)[occ_slice]
        nocc_read = int(np.asarray(occ_idx).size)
        read_key = np.s_[occ_slice, :, :]

    file = lib.FileMp(ovl_record["file"], "r")
    pool_read = None
    try:
        dataset = file[dataset_name]
        out_host = lib.empty(
            (max(1, nocc_read), nvir_ka, naux_q),
            dtype=dataset.dtype,
            type=lib.MemoryTypeHost,
        )
        if not is_pinned(out_host):
            raise RuntimeError("KRMP2 ovL read buffers must be pinned")

        pool_read = Pool(processes=max(1, int(ovl_record.get("read_processes", READ_PROCESSSES))))
        future = dataset.getitem(read_key, pool=pool_read, buf=out_host)
        future.wait()
        out = np.array(future.view(np.ndarray), copy=True)
    finally:
        if pool_read is not None:
            pool_read.close()
            pool_read.join()
        file.close()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()

    if occ_slice is None:
        return out
    return out[:nocc_read]
