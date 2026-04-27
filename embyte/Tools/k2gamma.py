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

"""GPU-backed k2gamma helpers for periodic embeddings.

The public entry points intentionally follow ``pyscf.pbc.tools.k2gamma``:

    - ``kpts_to_kmesh``
    - ``translation_vectors_for_kmesh``
    - ``get_phase``
    - ``mo_k2gamma``
    - ``k2gamma``
    - ``to_supercell_ao_integrals``

The heavy tensor transforms are moved to GPU with ``byteqc.lib.contraction`` /
``byteqc.lib.gemm`` and GPU overlap builders from ``gpu4pyscf``. Lightweight
mesh / super-cell helpers remain on CPU.
"""

from __future__ import annotations

import cupy as cp
import numpy as np
import time

from byteqc import lib
from pyscf import lib as pyscf_lib
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import group_by_conj_pairs

from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh as _gpu_kpts_to_kmesh

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


def _asnumpy(arr):
    if isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


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
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        if np.max(np.abs(arr.imag)) > 1e-8:
            raise ValueError("Cannot silently discard non-negligible imaginary part")
        arr = arr.real
    return cp.asarray(np.asarray(arr, dtype=np.float64, order="C"))


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
    e_cpu = cp.asnumpy(e)
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
    span_gpu = cp.asarray(np.asarray(span, dtype=np.float64, order="C"))
    try:
        ovlp_gpu = _gpu_overlap(cell, kpts=kpts, bvk_kmesh=bvk_kmesh)
        out_gpu = cp.dot(ovlp_gpu, span_gpu)
        out = cp.asnumpy(out_gpu)
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
    pairs = [[k, k_conj] for k, k_conj in k_conj_groups if k_conj is not None and k != k_conj]
    for idx in np.asarray(pairs, dtype=int):
        k_phase[idx[:, None], idx] = r2x2
    return k_phase


def kpts_to_kmesh(cell, kpts, precision=None, rcut=None):
    """Find the minimal k-point mesh to include all input kpts."""
    return _gpu_kpts_to_kmesh(cell, _normalize_kpts(kpts), precision=precision, rcut=rcut)


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
    phase_gpu = cp.asarray(phase)
    k_phase = _build_k_phase(cell, kpts)
    cache = {
        "cell": cell,
        "kpts": kpts,
        "kmesh": kmesh,
        "wrap_around": wrap_around,
        "scell": scell,
        "phase": phase,
        "phase_gpu": phase_gpu,
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
    a_gpu = cp.asarray(np.asarray(a, order="C"))
    eigval, eigvec = cp.linalg.eigh(a_gpu)
    eigval = cp.asnumpy(eigval)
    eigvec = cp.asnumpy(eigvec)
    del a_gpu
    _free_gpu_blocks()
    return eigval, eigvec


def _type2_eigh_in_real_span(c_complex, weights, s, metric_tol=1e-10):
    """Solve the non-zero type-2 eigenspace in span([Re(C), Im(C)]).

    The full realification branch forms ``f = Re(C diag(weights) C^H)`` and
    solves ``eigh(f, s, type=2)`` in the AO space.  The non-zero eigenspace of
    this low-rank operator is contained in the real span of ``C``.  This helper
    first builds an S-orthonormal basis of that much smaller real span, then
    diagonalizes the type-2 operator projected to it.
    """
    weights = np.asarray(weights, dtype=np.float64)
    span = np.hstack((c_complex.real, c_complex.imag))
    col_norm = np.linalg.norm(span, axis=0)
    span = span[:, col_norm > metric_tol]
    if span.size == 0:
        return np.empty(0), np.empty((c_complex.shape[0], 0))

    if callable(s):
        s_span = s(span)
    else:
        s_span = s @ span
    metric = _symmetrize(span.T @ s_span).real
    metric_eval, metric_vec = _gpu_eigh_cpu_result(metric)
    keep = metric_eval > metric_tol * max(1.0, float(np.max(np.abs(metric_eval))))
    if not np.any(keep):
        return np.empty(0), np.empty((c_complex.shape[0], 0))

    norm_vec = metric_vec[:, keep] / np.sqrt(metric_eval[keep])
    q = span @ norm_vec
    s_q = s_span @ norm_vec
    del span, s_span, metric, metric_eval, metric_vec, norm_vec

    z = c_complex.conj().T @ s_q
    h = (z.conj().T * weights[None, :]) @ z
    h = _symmetrize(h).real
    eigval, eigvec = _gpu_eigh_cpu_result(h)
    return eigval, q @ eigvec


def _realify_mos_if_needed(scell, e_g, c_gamma, s_sc=None, cache=None):
    c_gamma = np.asarray(c_gamma)

    c_r_max = np.max(np.abs(c_gamma.real), axis=0)
    imag_only = c_r_max < 1e-5
    if np.any(imag_only):
        c_gamma[:, imag_only] *= -1j

    sort_idx = np.argsort(e_g, kind="stable")
    e_g = e_g[sort_idx]
    c_gamma = c_gamma[:, sort_idx]

    c_i_max = np.max(np.abs(c_gamma.imag), axis=0)
    if c_i_max.max() < 1e-5:
        return e_g, c_gamma.real

    if s_sc is None:
        if cache is not None and cache.get("s_sc_cpu") is not None:
            s_sc = cache["s_sc_cpu"]
            s = np.asarray(s_sc).real
        else:
            s = lambda span: _gpu_overlap_dot_span_cpu(scell, span)
    else:
        s = np.asarray(s_sc).real

    e_k_degen = np.abs(e_g[1:] - e_g[:-1]) < 1e-3
    degen_mask = np.append(False, e_k_degen) | np.append(e_k_degen, False)
    degen_mask[c_i_max < 1e-5] = False

    if np.any(e_k_degen):
        c_rest = c_gamma[:, ~degen_mask]
        if c_rest.size > 0 and float(np.max(np.abs(c_rest.imag))) < 1e-4:
            shift = float(np.min(e_g[degen_mask]) - 0.1)
            c_deg = c_gamma[:, degen_mask]
            weights = np.asarray(e_g[degen_mask] - shift, dtype=c_deg.dtype)
            ev, na_orb = _type2_eigh_in_real_span(c_deg, weights.real, s)
            c_gamma = c_gamma.real
            n_degen = int(np.count_nonzero(degen_mask))
            keep = np.where(ev > 1e-7)[0]
            if len(keep) != n_degen:
                keep = np.arange(max(0, len(ev) - n_degen), len(ev))
            c_gamma[:, degen_mask] = na_orb[:, keep]
            return e_g, c_gamma

        _, c_gamma_real = _type2_eigh_in_real_span(c_gamma, e_g, s)
        if c_gamma_real.shape[1] == c_gamma.shape[1]:
            return e_g, c_gamma_real

    return e_g, c_gamma


def _default_mo_col_blksize(nao_sc):
    avail = lib.gpu_avail_bytes()
    target = max(1, int(avail * 0.20 // max(nao_sc * 16, 1)))
    return max(16, min(512, target))


def mo_k2gamma(
        cell, mo_energy, mo_coeff, kpts, kmesh=None, cache=None,
        with_mo_phase=True, force_real=True, col_blksize=None,
        realify_if_needed=True):
    if cache is None:
        cache = build_k2gamma_cache(cell, kpts, kmesh=kmesh)
    else:
        if cache.get("cell") is not cell:
            raise ValueError("k2gamma cache cell does not match the requested cell")
        if not np.array_equal(cache["kpts"], _normalize_kpts(kpts)):
            raise ValueError("k2gamma cache kpts does not match the requested kpts")

    scell = cache["scell"]
    phase = cache["phase"]

    e_g = np.hstack([_asnumpy(x).ravel() for x in mo_energy])
    e_sort_idx = np.argsort(e_g, kind="stable")
    e_g = e_g[e_sort_idx]
    dest_idx = np.empty_like(e_sort_idx)
    dest_idx[e_sort_idx] = np.arange(e_sort_idx.size)

    c_k = _to_gpu_stack(mo_coeff)
    nk, nao, nmo = c_k.shape
    nr = phase.shape[0]
    nao_sc = nao * nr
    nmo_sc = nk * nmo
    if col_blksize is None:
        col_blksize = _default_mo_col_blksize(nao_sc)

    phase_gpu = cache["phase_gpu"]
    k_phase_gpu = cache["k_phase_gpu"]
    phase_rot = lib.contraction("Rk", phase_gpu, "kh", k_phase_gpu, "Rkh")
    dtype = np.float64 if force_real else np.complex128
    c_gamma = np.empty((nao_sc, nmo_sc), dtype=dtype)
    complex_realify_needed = False

    for h in range(nk):
        phase_h = cp.ascontiguousarray(phase_rot[:, :, h])
        for p0 in range(0, nmo, col_blksize):
            p1 = min(p0 + col_blksize, nmo)
            block = lib.contraction(
                "Rk", phase_h,
                "kup", c_k[:, :, p0:p1],
                "Rup",
            )
            block = block.reshape(nao_sc, p1 - p0)
            block_cpu = _asnumpy(block)
            del block

            src_cols = h * nmo + np.arange(p0, p1)
            dst_cols = dest_idx[src_cols]
            if force_real:
                real_max = np.max(np.abs(block_cpu.real), axis=0)
                imag_max = np.max(np.abs(block_cpu.imag), axis=0)
                out = np.asarray(block_cpu.real, dtype=np.float64)
                imag_only = real_max < 1e-5
                complex_realify_needed |= bool(np.any((~imag_only) & (imag_max >= 1e-5)))
                if np.any(imag_only):
                    out[:, imag_only] = block_cpu.imag[:, imag_only]
                c_gamma[:, dst_cols] = out
                del out
            else:
                c_gamma[:, dst_cols] = block_cpu
            del block_cpu
        del phase_h

    del phase_rot
    _free_gpu_blocks()

    if force_real and complex_realify_needed and realify_if_needed:
        del c_gamma, c_k
        _free_gpu_blocks()
        scell, e_g, c_complex, _ = mo_k2gamma(
            cell, mo_energy, mo_coeff, kpts,
            kmesh=kmesh, cache=cache,
            with_mo_phase=False, force_real=False,
            col_blksize=col_blksize,
            realify_if_needed=False,
        )
        e_g, c_gamma = _realify_mos_if_needed(
            scell, e_g, c_complex, cache=cache)
        del c_complex
        if np.iscomplexobj(c_gamma):
            imag_max = float(np.max(np.abs(c_gamma.imag)))
            if imag_max >= 1e-5:
                raise ValueError(
                    "k2gamma realification did not produce real orbitals; "
                    f"max imaginary component is {imag_max:.3e}"
                )
            c_gamma = c_gamma.real
        c_gamma = np.asarray(c_gamma, dtype=np.float64, order="C")
        c_k = _to_gpu_stack(mo_coeff) if with_mo_phase else None

    mo_phase = None
    if with_mo_phase:
        c_gamma_gpu = cp.asarray(c_gamma)
        s_k = _ensure_cache_overlap(cache)
        s_k_g = lib.contraction("kuv", s_k, "Rk", phase_gpu, "kuRv", opb="CONJ")
        s_k_g = s_k_g.reshape(nk, nao, nr * nao)
        tmp = lib.contraction("kum", c_k, "kuv", s_k_g, "kmv", opa="CONJ")
        mo_phase = lib.contraction("kmv", tmp, "vi", c_gamma_gpu, "kmi")
        mo_phase = _asnumpy(mo_phase)
        del c_gamma_gpu, s_k_g, tmp

    if c_k is not None:
        del c_k
    _free_gpu_blocks()
    return scell, e_g, c_gamma, mo_phase


def k2gamma(kmf, kmesh=None):
    r"""Convert a k-sampled mean-field object to the corresponding gamma supercell MF."""
    from pyscf.pbc import dft, scf

    if isinstance(kmf.kpts, KPoints):
        kmf = kmf.to_khf()

    cache = build_k2gamma_cache(kmf.cell, kmf.kpts, kmesh=kmesh)

    def transform(mo_energy, mo_coeff, mo_occ):
        assert not isinstance(kmf.kpts, KPoints)
        kpts = kmf.kpts
        scell, e_g, c_gamma = mo_k2gamma(
            kmf.cell, mo_energy, mo_coeff, kpts, kmesh, cache=cache,
            with_mo_phase=False, force_real=True
        )[:3]
        sort_idx = np.argsort(np.hstack([_asnumpy(x).ravel() for x in mo_energy]), kind="stable")
        occ = np.hstack([_asnumpy(x).ravel() for x in mo_occ])[sort_idx]
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


def to_supercell_ao_integrals(cell, kpts, ao_ints, kmesh=None, force_real=True, cache=None):
    """Transform k-sampled AO matrices to the gamma-supercell AO basis."""
    if cache is None:
        cache = build_k2gamma_cache(cell, kpts, kmesh=kmesh)
    else:
        if cache.get("cell") is not cell:
            raise ValueError("k2gamma cache cell does not match the requested cell")
        if not np.array_equal(cache["kpts"], _normalize_kpts(kpts)):
            raise ValueError("k2gamma cache kpts does not match the requested kpts")

    scell = cache["scell"]
    phase_gpu = cache["phase_gpu"]
    ao_ints_gpu = _to_gpu_stack(ao_ints)
    nr = phase_gpu.shape[0]
    nao = int(cell.nao_nr())
    nao_sc = int(scell.nao_nr())
    dtype = np.float64 if force_real else np.complex128
    ao_sc = np.empty((nao_sc, nao_sc), dtype=dtype)

    for r in range(nr):
        phase_row = phase_gpu[r:r + 1]
        phase_pair = lib.contraction(
            "ak", phase_row,
            "Sk", phase_gpu,
            "Sk",
            opb="CONJ",
        )
        row_blocks = lib.contraction("Sk", phase_pair, "kij", ao_ints_gpu, "Sij")
        row_blocks = row_blocks.transpose(1, 0, 2).reshape(nao, nao_sc)
        if force_real:
            ao_sc[r * nao:(r + 1) * nao] = _asnumpy(row_blocks.real)
        else:
            ao_sc[r * nao:(r + 1) * nao] = _asnumpy(row_blocks)
        del phase_row, phase_pair, row_blocks
        _free_gpu_blocks()

    del ao_ints_gpu
    _free_gpu_blocks()
    return ao_sc
