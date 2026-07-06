# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# This file is part of ByteQC.

"""Standalone single-pass GPU k2gamma helpers.

This module is intentionally independent from ``embyte.Tools.k2gamma``.  The
main difference from the older implementation is that ``mo_k2gamma`` always
builds the complex gamma-supercell MO coefficient matrix once, then realifies
that same matrix when requested.  It avoids the recursive "try real -> rebuild
complex -> realify" path, which can double peak memory for large k meshes.
"""

from __future__ import annotations

import os
import time

import cupy as cp
import cupyx
import numpy as np

from byteqc import lib
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh as _gpu_kpts_to_kmesh
from pyscf import lib as pyscf_lib
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import group_by_conj_pairs

__all__ = [
    "kpts_to_kmesh",
    "translation_vectors_for_kmesh",
    "get_phase",
    "double_translation_indices",
    "translation_map",
    "build_k2gamma_cache",
    "gpu_generalized_eigvalsh",
    "mo_k2gamma",
    "k2gamma",
    "to_supercell_ao_integrals",
]


def _normalize_kpts(kpts):
    if kpts is None:
        return None
    if hasattr(kpts, "kpts"):
        kpts = kpts.kpts
    return np.asarray(kpts)


def _to_gpu_stack(arr):
    if isinstance(arr, (list, tuple)):
        return cp.stack([cp.asarray(x) for x in arr], axis=0)
    return cp.asarray(arr)


def _mo_coeff_shape(mo_coeff):
    if isinstance(mo_coeff, (list, tuple)):
        nk = len(mo_coeff)
        nao, nmo = mo_coeff[0].shape
        return nk, int(nao), int(nmo)
    nk, nao, nmo = mo_coeff.shape
    return int(nk), int(nao), int(nmo)


def _mo_coeff_block_gpu(mo_coeff, p0, p1):
    if isinstance(mo_coeff, (list, tuple)):
        return cp.stack(
            [cp.asarray(x[:, p0:p1], dtype=cp.complex128, order="C") for x in mo_coeff],
            axis=0,
        )
    return cp.asarray(mo_coeff[:, :, p0:p1], dtype=cp.complex128, order="C")


def _to_numpy_host(arr, dtype=None, order="C"):
    if isinstance(arr, cp.ndarray):
        host = cupyx.empty_pinned(arr.shape, dtype=arr.dtype, order=order)
        arr.get(out=host, blocking=True)
        arr = host
    if dtype is None:
        arr_np = np.asarray(arr, order=order)
    else:
        arr_np = np.asarray(arr, dtype=dtype, order=order)
    if isinstance(arr_np, np.ndarray) and not lib.is_pinned(arr_np):
        host = cupyx.empty_pinned(arr_np.shape, dtype=arr_np.dtype, order=order)
        host[...] = arr_np
        return host
    return arr_np


def _gpu_to_pinned(arr):
    host = cupyx.empty_pinned(arr.shape, dtype=arr.dtype, order="C")
    arr.get(out=host, blocking=True)
    return host


def _real_pinned(arr):
    out = cupyx.empty_pinned(arr.shape, dtype=np.float64, order="C")
    out[:] = arr.real
    return out


def _free_gpu_blocks():
    cp.cuda.get_current_stream().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _as_float64_gpu(arr):
    if isinstance(arr, cp.ndarray):
        if arr.dtype.kind == "c":
            if cp.max(cp.abs(arr.imag)).get() > 1e-8:
                raise ValueError("Cannot silently discard non-negligible imaginary part")
            arr = arr.real
        return cp.asarray(arr, dtype=cp.float64)
    arr = _to_numpy_host(arr)
    if np.iscomplexobj(arr):
        arr_imag_gpu = cp.asarray(arr.imag)
        imag_max = float(cp.max(cp.abs(arr_imag_gpu)).get())
        del arr_imag_gpu
        _free_gpu_blocks()
        if imag_max > 1e-8:
            raise ValueError("Cannot silently discard non-negligible imaginary part")
        arr = arr.real
    return cp.asarray(arr, dtype=cp.float64, order="C")


def gpu_generalized_eigvalsh(h, s, logger=None):
    """Return generalized eigvals for ``h C = s C e`` using GPU work arrays."""
    t0 = time.time()
    h_gpu = _as_float64_gpu(h)
    s_gpu = _as_float64_gpu(s)

    seig, t = cp.linalg.eigh(s_gpu)
    del s_gpu
    seig_inv = 1.0 / cp.sqrt(seig)
    seig_inv[seig < 1e-15] = 0.0
    t *= seig_inv[None, :]
    del seig, seig_inv

    tmp = cp.dot(h_gpu, t)
    del h_gpu
    heff = cp.dot(t.T, tmp)
    del tmp, t

    e = cp.linalg.eigvalsh(heff)
    e_cpu = _gpu_to_pinned(e)
    del e, heff
    _free_gpu_blocks()
    if logger is not None:
        logger.info(
            "----------- GPU generalized eigvalsh time cost is %s" %
            (time.time() - t0))
    return e_cpu


def _gpu_overlap(cell, kpts=None, bvk_kmesh=None):
    ovlp = int1e.int1e_ovlp(cell, kpts=kpts, bvk_kmesh=bvk_kmesh)
    return cp.asarray(ovlp)


def _gpu_overlap_dot_span_cpu(cell, span, kpts=None, bvk_kmesh=None):
    """Build overlap on GPU only long enough to apply it to a skinny span."""
    span_gpu = cp.asarray(span, dtype=cp.float64, order="C")
    try:
        ovlp_gpu = _gpu_overlap(cell, kpts=kpts, bvk_kmesh=bvk_kmesh)
        out_gpu = cp.dot(ovlp_gpu, span_gpu)
        out = _gpu_to_pinned(out_gpu)
        del ovlp_gpu, out_gpu
    finally:
        del span_gpu
        _free_gpu_blocks()
    return out


def _build_k_phase(cell, kpts):
    nk = len(kpts)
    k_conj_groups = group_by_conj_pairs(cell, kpts, return_kpts_pairs=False)
    k_phase = np.eye(nk, dtype=np.complex128)
    r2x2 = np.array([[1.0, 1j], [1.0, -1j]], dtype=np.complex128) * (0.5 ** 0.5)
    pairs = [
        [k, k_conj]
        for k, k_conj in k_conj_groups
        if k_conj is not None and k != k_conj
    ]
    for idx in np.asarray(pairs, dtype=int):
        k_phase[idx[:, None], idx] = r2x2
    return k_phase


def kpts_to_kmesh(cell, kpts, precision=None, rcut=None):
    """Find the minimal k-point mesh to include all input kpts."""
    return _gpu_kpts_to_kmesh(
        cell, _normalize_kpts(kpts), precision=precision, rcut=rcut
    )


def translation_vectors_for_kmesh(cell, kmesh, wrap_around=False):
    """Translation vectors used to build the corresponding gamma supercell."""
    latt_vec = cell.lattice_vectors()
    r_rel_a = np.arange(kmesh[0])
    r_rel_b = np.arange(kmesh[1])
    r_rel_c = np.arange(kmesh[2])
    if wrap_around:
        r_rel_a[(kmesh[0] + 1) // 2:] -= kmesh[0]
        r_rel_b[(kmesh[1] + 1) // 2:] -= kmesh[1]
        r_rel_c[(kmesh[2] + 1) // 2:] -= kmesh[2]
    r_vec_rel = pyscf_lib.cartesian_prod((r_rel_a, r_rel_b, r_rel_c))
    return np.dot(r_vec_rel, latt_vec)


def get_phase(cell, kpts, kmesh=None, wrap_around=False):
    """Unitary phase matrix that connects k-sampled and gamma-supercell bases."""
    kpts = _normalize_kpts(kpts)
    if kmesh is None:
        kmesh = kpts_to_kmesh(cell, kpts)
    r_vec_abs = translation_vectors_for_kmesh(cell, kmesh, wrap_around)

    nr = len(r_vec_abs)
    phase = np.exp(1j * np.dot(r_vec_abs, np.asarray(kpts).T))
    phase /= np.sqrt(nr)

    scell = tools.super_cell(cell, kmesh, wrap_around)
    return scell, phase


def build_k2gamma_cache(cell, kpts, kmesh=None, wrap_around=False, with_overlap=False):
    """Precompute cell/k-point dependent tensors for repeated k2gamma transforms."""
    kpts = _normalize_kpts(kpts)
    if kmesh is None:
        kmesh = kpts_to_kmesh(cell, kpts)
    kmesh = np.asarray(kmesh, dtype=int)
    scell, phase = get_phase(cell, kpts, kmesh=kmesh, wrap_around=wrap_around)
    k_phase = _build_k_phase(cell, kpts)
    cache = {
        "cell": cell,
        "kpts": kpts,
        "kmesh": kmesh,
        "wrap_around": wrap_around,
        "scell": scell,
        "phase": phase,
        "phase_gpu": cp.asarray(phase),
        "s_k": None,
        "s_sc_cpu": None,
        "k_phase": k_phase,
        "k_phase_gpu": cp.asarray(k_phase),
    }
    if with_overlap:
        cache["s_k"] = _gpu_overlap(cell, kpts=kpts, bvk_kmesh=kmesh)
    return cache


def _ensure_cache_overlap(cache):
    if cache["s_k"] is None:
        cache["s_k"] = _gpu_overlap(
            cache["cell"],
            kpts=cache["kpts"],
            bvk_kmesh=cache["kmesh"],
        )
    return cache["s_k"]


def translation_map(nk):
    idx = np.repeat(np.arange(nk)[None, :], nk - 1, axis=0)
    strides = idx.strides
    return np.ndarray(
        (nk, nk),
        strides=(strides[0] - strides[1], strides[1]),
        dtype=int,
        buffer=np.append(idx.ravel(), 0),
    )


def double_translation_indices(kmesh):
    tx = translation_map(kmesh[0])
    ty = translation_map(kmesh[1])
    tz = translation_map(kmesh[2])
    idx = np.ravel_multi_index(
        [
            tx[:, None, None, :, None, None],
            ty[None, :, None, None, :, None],
            tz[None, None, :, None, None, :],
        ],
        kmesh,
    )
    nk = np.prod(kmesh)
    return idx.reshape(nk, nk)


def _symmetrize(a):
    return (a + a.T.conj()) * 0.5


def _gpu_eigh_cpu_result(a):
    a_gpu = cp.asarray(a, order="C")
    eigval, eigvec = cp.linalg.eigh(a_gpu)
    eigval = _gpu_to_pinned(eigval)
    eigvec = _gpu_to_pinned(eigvec)
    del a_gpu
    _free_gpu_blocks()
    return eigval, eigvec


def _type2_eigh_in_real_span(c_complex, weights, s, metric_tol=1e-10):
    """Solve the non-zero type-2 eigenspace in span([Re(C), Im(C)])."""
    c_complex = _to_numpy_host(c_complex, dtype=np.complex128)
    weights = _to_numpy_host(weights, dtype=np.float64)
    nrow, ncol = c_complex.shape
    span_full = cupyx.empty_pinned((nrow, 2 * ncol), dtype=np.float64, order="C")
    span_full[:, :ncol] = c_complex.real
    span_full[:, ncol:] = c_complex.imag

    span_gpu = cp.asarray(span_full, dtype=cp.float64, order="C")
    col_norm = cp.linalg.norm(span_gpu, axis=0)
    keep_cols = _gpu_to_pinned(col_norm > metric_tol)
    del span_gpu, col_norm
    _free_gpu_blocks()

    nkeep = int(np.count_nonzero(keep_cols))
    span = cupyx.empty_pinned((nrow, nkeep), dtype=np.float64, order="C")
    span[:] = span_full[:, keep_cols]
    del span_full, keep_cols
    if span.size == 0:
        return (
            cupyx.empty_pinned((0,), dtype=np.float64),
            cupyx.empty_pinned((nrow, 0), dtype=np.float64),
        )
    span = np.asarray(span, dtype=np.float64, order="C")

    if callable(s):
        s_span = s(span)
    else:
        s_gpu = cp.asarray(s, dtype=cp.float64, order="C")
        span_gpu = cp.asarray(span, dtype=cp.float64, order="C")
        s_span_gpu = cp.dot(s_gpu, span_gpu)
        s_span = _gpu_to_pinned(s_span_gpu)
        del s_gpu, span_gpu, s_span_gpu
        _free_gpu_blocks()

    span_gpu = cp.asarray(span, dtype=cp.float64, order="C")
    s_span_gpu = cp.asarray(s_span, dtype=cp.float64, order="C")
    metric_gpu = cp.dot(span_gpu.T, s_span_gpu)
    metric_gpu = _symmetrize(metric_gpu).real
    metric = _gpu_to_pinned(metric_gpu)
    del span_gpu, metric_gpu
    _free_gpu_blocks()

    metric_eval, metric_vec = _gpu_eigh_cpu_result(metric)
    metric_eval_gpu = cp.asarray(metric_eval)
    keep = _gpu_to_pinned(
        metric_eval_gpu > metric_tol * cp.maximum(1.0, cp.max(cp.abs(metric_eval_gpu)))
    )
    del metric_eval_gpu, metric
    _free_gpu_blocks()
    if not np.any(keep):
        return (
            cupyx.empty_pinned((0,), dtype=np.float64),
            cupyx.empty_pinned((nrow, 0), dtype=np.float64),
        )

    norm_vec = metric_vec[:, keep] / np.sqrt(metric_eval[keep])
    span_gpu = cp.asarray(span, dtype=cp.float64, order="C")
    s_span_gpu = cp.asarray(s_span, dtype=cp.float64, order="C")
    norm_vec_gpu = cp.asarray(norm_vec, dtype=cp.float64, order="C")
    q_gpu = cp.dot(span_gpu, norm_vec_gpu)
    s_q_gpu = cp.dot(s_span_gpu, norm_vec_gpu)
    q = _gpu_to_pinned(q_gpu)
    s_q = _gpu_to_pinned(s_q_gpu)
    del span, s_span, span_gpu, s_span_gpu, norm_vec_gpu, q_gpu, s_q_gpu
    del metric_eval, metric_vec, norm_vec, keep
    _free_gpu_blocks()

    c_gpu = cp.asarray(c_complex, dtype=cp.complex128, order="C")
    s_q_gpu = cp.asarray(s_q, dtype=cp.float64, order="C")
    weights_gpu = cp.asarray(weights, dtype=cp.float64)
    z = cp.dot(c_gpu.conj().T, s_q_gpu)
    h = cp.dot(z.conj().T * weights_gpu[None, :], z)
    h = _symmetrize(h).real
    h_cpu = _gpu_to_pinned(h)
    del c_gpu, s_q_gpu, weights_gpu, z, h, s_q
    _free_gpu_blocks()

    eigval, eigvec = _gpu_eigh_cpu_result(h_cpu)
    q_gpu = cp.asarray(q, dtype=cp.float64, order="C")
    eigvec_gpu = cp.asarray(eigvec, dtype=cp.float64, order="C")
    orb_gpu = cp.dot(q_gpu, eigvec_gpu)
    orb = _gpu_to_pinned(orb_gpu)
    del q, q_gpu, eigvec_gpu, orb_gpu, eigvec, h_cpu
    _free_gpu_blocks()
    return eigval, orb


def _default_mo_col_blksize(nao_sc, nk=1):
    avail = lib.gpu_avail_bytes()
    block_col_bytes = max(nao_sc * max(int(nk), 1) * np.dtype(np.complex128).itemsize, 1)
    target = max(1, int(avail * 0.05 // block_col_bytes))
    return max(1, min(128, target))


def _validate_cache(cache, cell, kpts):
    if cache.get("cell") is not cell:
        raise ValueError("k2gamma cache cell does not match the requested cell")
    if not np.array_equal(cache["kpts"], _normalize_kpts(kpts)):
        raise ValueError("k2gamma cache kpts does not match the requested kpts")


def _realify_column_components(cell, kpts, mo_energy_by_k, dest_idx, nmo, degen_tol):
    """Small realification groups from local band degeneracy and k/-k pairs."""
    nk = len(mo_energy_by_k)
    ncol = nk * nmo
    parent = np.arange(ncol, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for k, e_k in enumerate(mo_energy_by_k):
        start = 0
        while start < nmo:
            stop = start + 1
            while stop < nmo and abs(e_k[stop] - e_k[stop - 1]) < degen_tol:
                stop += 1
            if stop - start > 1:
                root = k * nmo + start
                for p in range(start + 1, stop):
                    union(root, k * nmo + p)
            start = stop

    k_conj_groups = group_by_conj_pairs(cell, kpts, return_kpts_pairs=False)
    for k, k_conj in k_conj_groups:
        if k_conj is None or k == k_conj:
            continue
        k = int(k)
        k_conj = int(k_conj)
        for p in range(nmo):
            union(k * nmo + p, k_conj * nmo + p)

    groups = {}
    for src in range(ncol):
        groups.setdefault(find(src), []).append(int(dest_idx[src]))
    return [np.asarray(sorted(cols), dtype=np.int64) for cols in groups.values()]


def _phase_align_columns_inplace(c_gamma, cols):
    """Remove arbitrary scalar phases from singleton real orbitals in-place."""
    new_imag = np.empty(len(cols), dtype=np.float64)
    for i, col in enumerate(cols):
        vec = c_gamma[:, int(col)]
        pivot = int(np.argmax(np.abs(vec)))
        amp = vec[pivot]
        amp_abs = abs(amp)
        if amp_abs > 0.0:
            vec *= np.conj(amp / amp_abs)
        new_imag[i] = np.max(np.abs(vec.imag))
    return new_imag


def _debug_check_realify_s_orthogonality(s, orb, cols, tol):
    if callable(s):
        s_orb = s(orb)
        orb_gpu = cp.asarray(orb, dtype=cp.float64, order="C")
        s_orb_gpu = cp.asarray(s_orb, dtype=cp.float64, order="C")
        del s_orb
    else:
        orb_gpu = cp.asarray(orb, dtype=cp.float64, order="C")
        s_gpu = cp.asarray(s, dtype=cp.float64, order="C")
        s_orb_gpu = cp.dot(s_gpu, orb_gpu)
        del s_gpu

    metric_gpu = cp.dot(orb_gpu.T, s_orb_gpu)
    metric_gpu = _symmetrize(metric_gpu).real
    metric_gpu -= cp.eye(metric_gpu.shape[0], dtype=cp.float64)
    err = float(cp.max(cp.abs(metric_gpu)).get()) if metric_gpu.size else 0.0
    del orb_gpu, s_orb_gpu, metric_gpu
    _free_gpu_blocks()
    if err > tol:
        raise ValueError(
            "k2gamma realify block failed S-orthogonality debug check: "
            f"cols={cols.tolist()}, max_err={err:.3e}, tol={tol:.3e}"
        )
    return err


def _build_mo_k2gamma_singlepass(
        cell, mo_energy, mo_coeff, kpts, kmesh, cache, col_blksize,
        force_real, realify_if_needed):
    scell = cache["scell"]
    phase = cache["phase"]

    mo_energy_by_k = [_to_numpy_host(x).ravel() for x in mo_energy]
    e_g = np.hstack(mo_energy_by_k)
    e_sort_idx = np.argsort(e_g, kind="stable")
    e_g = e_g[e_sort_idx]
    dest_idx = np.empty_like(e_sort_idx)
    dest_idx[e_sort_idx] = np.arange(e_sort_idx.size)

    nk, nao, nmo = _mo_coeff_shape(mo_coeff)
    nr = phase.shape[0]
    nao_sc = nao * nr
    nmo_sc = nk * nmo
    if col_blksize is None:
        col_blksize = _default_mo_col_blksize(nao_sc, nk)

    phase_rot = lib.contraction(
        "Rk",
        cache["phase_gpu"],
        "kh",
        cache["k_phase_gpu"],
        "Rkh",
    )
    c_gamma = cupyx.empty_pinned((nao_sc, nmo_sc), dtype=np.complex128, order="C")
    collect_max = force_real and realify_if_needed
    if collect_max:
        c_r_max = cupyx.empty_pinned((nmo_sc,), dtype=np.float64, order="C")
        c_i_max = cupyx.empty_pinned((nmo_sc,), dtype=np.float64, order="C")

    for p0 in range(0, nmo, col_blksize):
        p1 = min(p0 + col_blksize, nmo)
        c_k_blk = _mo_coeff_block_gpu(mo_coeff, p0, p1)
        block = lib.contraction(
            "Rkh",
            phase_rot,
            "kup",
            c_k_blk,
            "Ruhp",
        )
        block = block.reshape(nao_sc, nk * (p1 - p0))

        src_cols = (
            np.arange(nk)[:, None] * nmo + np.arange(p0, p1)[None, :]
        ).ravel()
        dst_cols = dest_idx[src_cols]
        if collect_max:
            block_real = block.real
            block_r_max = cp.max(block_real, axis=0)
            block_r_min = cp.min(block_real, axis=0)
            block_r_min *= -1.0
            cp.maximum(block_r_max, block_r_min, out=block_r_max)

            block_imag = block.imag
            block_i_max = cp.max(block_imag, axis=0)
            block_i_min = cp.min(block_imag, axis=0)
            block_i_min *= -1.0
            cp.maximum(block_i_max, block_i_min, out=block_i_max)

            c_r_max[dst_cols] = _gpu_to_pinned(block_r_max)
            c_i_max[dst_cols] = _gpu_to_pinned(block_i_max)
            del block_real, block_imag
            del block_r_max, block_r_min, block_i_max, block_i_min
        c_gamma[:, dst_cols] = _gpu_to_pinned(block)

        del c_k_blk, block, src_cols, dst_cols
        _free_gpu_blocks()

    del phase_rot
    _free_gpu_blocks()

    if not collect_max:
        return scell, e_g, c_gamma, col_blksize

    imag_only = c_r_max < 1e-5
    if np.any(imag_only):
        c_gamma[:, imag_only] *= -1j
        old_r_max = c_r_max[imag_only].copy()
        c_r_max[imag_only] = c_i_max[imag_only]
        c_i_max[imag_only] = old_r_max
        del old_r_max
    del imag_only

    c_i_max_max = float(np.max(c_i_max))
    if c_i_max_max < 1e-5:
        del c_r_max, c_i_max
        c_gamma_real = _real_pinned(c_gamma)
        del c_gamma
        return scell, e_g, c_gamma_real, col_blksize

    if cache.get("s_sc_cpu") is not None:
        s = np.asarray(cache["s_sc_cpu"]).real
    else:
        s = lambda span: _gpu_overlap_dot_span_cpu(scell, span)

    imag_tol = 1e-5
    degen_tol = 1e-3
    realify_groups = _realify_column_components(
        cell, cache["kpts"], mo_energy_by_k, dest_idx, nmo, degen_tol
    )
    realify_member = np.zeros(e_g.size, dtype=bool)
    for cols in realify_groups:
        if cols.size > 1:
            realify_member[cols] = True

    significant_imag = c_i_max >= imag_tol
    bad_single = significant_imag & ~realify_member
    if np.any(bad_single):
        bad_cols = np.where(bad_single)[0]
        c_i_max[bad_cols] = _phase_align_columns_inplace(c_gamma, bad_cols)
        significant_imag = c_i_max >= imag_tol
        bad_single = significant_imag & ~realify_member

    if np.any(bad_single):
        bad_cols = np.where(bad_single)[0]
        worst = int(bad_cols[np.argmax(c_i_max[bad_cols])])
        left_gap = abs(e_g[worst] - e_g[worst - 1]) if worst > 0 else np.inf
        right_gap = abs(e_g[worst + 1] - e_g[worst]) if worst + 1 < e_g.size else np.inf
        raise ValueError(
            "k2gamma realification found a singleton orbital with a "
            "significant imaginary component after scalar phase alignment; "
            "no local k/-k or band-degenerate partner is available, refusing "
            "full-space realify. "
            f"col={worst}, energy={e_g[worst]:.16e}, "
            f"max_imag={c_i_max[worst]:.3e}, "
            f"left_gap={left_gap:.3e}, right_gap={right_gap:.3e}, "
            f"degen_tol={degen_tol:.3e}"
        )

    c_gamma_real = _real_pinned(c_gamma)
    debug_realify = os.environ.get("BYTEQC_K2GAMMA_DEBUG_REALIFY", "0") == "1"
    debug_tol = float(os.environ.get("BYTEQC_K2GAMMA_DEBUG_REALIFY_TOL", "1e-6"))
    debug_max_err = 0.0
    debug_ncheck = 0
    for cols in realify_groups:
        if cols.size <= 1 or not np.any(significant_imag[cols]):
            continue
        block_len = int(cols.size)
        c_deg = cupyx.empty_pinned(
            (c_gamma.shape[0], block_len), dtype=np.complex128, order="C"
        )
        c_deg[:] = c_gamma[:, cols]
        shift = float(np.min(e_g[cols]) - 0.1)
        weights = np.asarray(
            e_g[cols] - shift, dtype=np.float64, order="C"
        )
        ev, na_orb = _type2_eigh_in_real_span(c_deg, weights, s)
        keep = np.where(ev > 1e-7)[0]
        if len(keep) != block_len:
            keep = np.arange(max(0, len(ev) - block_len), len(ev))
        if len(keep) != block_len:
            raise ValueError(
                "k2gamma realification failed to recover the degenerate "
                f"subspace dimension: cols={cols.tolist()}, "
                f"expected={block_len}, got={len(keep)}"
            )
        c_gamma_real[:, cols] = na_orb[:, keep]
        if debug_realify:
            err = _debug_check_realify_s_orthogonality(
                s, na_orb[:, keep], cols, debug_tol
            )
            debug_max_err = max(debug_max_err, err)
            debug_ncheck += 1
        del c_deg, weights, ev, na_orb, keep

    if debug_realify and debug_ncheck:
        print(
            "BYTEQC_K2GAMMA_DEBUG_REALIFY: checked "
            f"{debug_ncheck} realify blocks, max S-orthogonality err="
            f"{debug_max_err:.3e}, tol={debug_tol:.3e}",
            flush=True,
        )

    del c_r_max, c_i_max, c_gamma, significant_imag
    _free_gpu_blocks()
    return scell, e_g, c_gamma_real, col_blksize


def _compute_mo_phase(mo_coeff, c_gamma, cache):
    c_k = _to_gpu_stack(mo_coeff)
    c_gamma_gpu = cp.asarray(c_gamma)
    s_k = _ensure_cache_overlap(cache)
    nk, nao = c_k.shape[:2]
    nr = cache["phase_gpu"].shape[0]

    s_k_g = lib.contraction(
        "kuv", s_k, "Rk", cache["phase_gpu"], "kuRv", opb="CONJ"
    )
    s_k_g = s_k_g.reshape(nk, nao, nr * nao)
    tmp = lib.contraction("kum", c_k, "kuv", s_k_g, "kmv", opa="CONJ")
    mo_phase = lib.contraction("kmv", tmp, "vi", c_gamma_gpu, "kmi")
    mo_phase = _gpu_to_pinned(mo_phase)

    del c_k, c_gamma_gpu, s_k_g, tmp
    _free_gpu_blocks()
    return mo_phase


def mo_k2gamma(
        cell, mo_energy, mo_coeff, kpts, kmesh=None, cache=None,
        with_mo_phase=True, force_real=True, col_blksize=None,
        realify_if_needed=True):
    """Convert k-point MOs to gamma-supercell MOs without recursive rebuilds."""
    if cache is None:
        cache = build_k2gamma_cache(cell, kpts, kmesh=kmesh)
    else:
        _validate_cache(cache, cell, kpts)

    scell, e_g, c_gamma, col_blksize = _build_mo_k2gamma_singlepass(
        cell, mo_energy, mo_coeff, kpts, kmesh, cache, col_blksize,
        force_real, realify_if_needed
    )

    mo_phase = _compute_mo_phase(mo_coeff, c_gamma, cache) if with_mo_phase else None
    return scell, e_g, c_gamma, mo_phase


def k2gamma(kmf, kmesh=None):
    r"""Convert a k-sampled mean-field object to a gamma-supercell MF."""
    from pyscf.pbc import dft, scf

    if isinstance(kmf.kpts, KPoints):
        kmf = kmf.to_khf()

    cache = build_k2gamma_cache(kmf.cell, kmf.kpts, kmesh=kmesh)

    def transform(mo_energy, mo_coeff, mo_occ):
        assert not isinstance(kmf.kpts, KPoints)
        kpts = kmf.kpts
        scell, e_g, c_gamma = mo_k2gamma(
            kmf.cell,
            mo_energy,
            mo_coeff,
            kpts,
            kmesh,
            cache=cache,
            with_mo_phase=False,
            force_real=True,
        )[:3]
        sort_idx = np.argsort(
            np.hstack([_to_numpy_host(x).ravel() for x in mo_energy]),
            kind="stable",
        )
        occ = np.hstack([_to_numpy_host(x).ravel() for x in mo_occ])[sort_idx]
        return scell, e_g, c_gamma, occ

    mo_coeff = kmf.mo_coeff
    mo_energy = kmf.mo_energy
    mo_occ = kmf.mo_occ

    if isinstance(kmf, scf.kuhf.KUHF):
        scell, ea, ca, occ_a = transform(mo_energy[0], mo_coeff[0], mo_occ[0])
        scell, eb, cb, occ_b = transform(mo_energy[1], mo_coeff[1], mo_occ[1])
        e_g = [ea, eb]
        c_gamma = [ca, cb]
        mo_occ = [occ_a, occ_b]
    elif isinstance(kmf, scf.khf.KRHF):
        scell, e_g, c_gamma, mo_occ = transform(mo_energy, mo_coeff, mo_occ)
    else:
        raise NotImplementedError(f"SCF object {kmf.__class__} not supported")

    known_cls = (
        (dft.kuks.KUKS, dft.uks.UKS),
        (dft.kroks.KROKS, dft.roks.ROKS),
        (dft.krks.KRKS, dft.rks.RKS),
        (dft.kgks.KGKS, dft.gks.GKS),
        (scf.kuhf.KUHF, scf.uhf.UHF),
        (scf.krohf.KROHF, scf.rohf.ROHF),
        (scf.khf.KRHF, scf.hf.RHF),
        (scf.kghf.KGHF, scf.ghf.GHF),
    )
    for k_cls, gamma_cls in known_cls:
        if isinstance(kmf, k_cls):
            mf = gamma_cls(scell)
            mf.exxdiv = kmf.exxdiv
            if isinstance(mf, dft.KohnShamDFT):
                mf.xc = kmf.xc
            break
    else:
        raise RuntimeError(f"k2gamma for SCF object {kmf} not supported.")

    mf.mo_coeff = c_gamma
    mf.mo_energy = e_g
    mf.mo_occ = mo_occ
    return mf


def to_supercell_ao_integrals(
        cell, kpts, ao_ints, kmesh=None, force_real=True, cache=None):
    """Transform k-sampled AO matrices to the gamma-supercell AO basis."""
    if cache is None:
        cache = build_k2gamma_cache(cell, kpts, kmesh=kmesh)
    else:
        _validate_cache(cache, cell, kpts)

    scell = cache["scell"]
    phase_gpu = cache["phase_gpu"]
    ao_ints_gpu = _to_gpu_stack(ao_ints)
    nr = phase_gpu.shape[0]
    nao = int(cell.nao_nr())
    nao_sc = int(scell.nao_nr())
    dtype = np.float64 if force_real else np.complex128
    ao_sc = cupyx.empty_pinned((nao_sc, nao_sc), dtype=dtype, order="C")

    for r in range(nr):
        phase_row = phase_gpu[r:r + 1]
        phase_pair = lib.contraction(
            "ak",
            phase_row,
            "Sk",
            phase_gpu,
            "Sk",
            opb="CONJ",
        )
        row_blocks = lib.contraction("Sk", phase_pair, "kij", ao_ints_gpu, "Sij")
        row_blocks = row_blocks.transpose(1, 0, 2).reshape(nao, nao_sc)
        if force_real:
            row_host = _gpu_to_pinned(row_blocks.real)
        else:
            row_host = _gpu_to_pinned(row_blocks)
        ao_sc[r * nao:(r + 1) * nao] = row_host
        del phase_row, phase_pair, row_blocks, row_host
        _free_gpu_blocks()

    del ao_ints_gpu
    _free_gpu_blocks()
    return ao_sc
