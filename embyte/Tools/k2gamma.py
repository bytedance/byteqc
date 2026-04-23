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
import scipy.linalg

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


def _gpu_overlap(cell, kpts=None, bvk_kmesh=None):
    ovlp = int1e.int1e_ovlp(cell, kpts=kpts, bvk_kmesh=bvk_kmesh)
    return cp.asarray(ovlp)


def _cpu_overlap(cell, kpts=None):
    if kpts is None:
        return np.asarray(cell.pbc_intor("int1e_ovlp"))
    return np.asarray(cell.pbc_intor("int1e_ovlp", kpts=kpts))


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


def build_k2gamma_cache(cell, kpts, kmesh=None, wrap_around=False):
    """Precompute cell/k-point dependent tensors for repeated k2gamma transforms."""
    kpts = _normalize_kpts(kpts)
    if kmesh is None:
        kmesh = kpts_to_kmesh(cell, kpts)
    kmesh = np.asarray(kmesh, dtype=int)
    scell, phase = get_phase(cell, kpts, kmesh=kmesh, wrap_around=wrap_around)
    phase_gpu = cp.asarray(phase)
    s_k = _gpu_overlap(cell, kpts=kpts, bvk_kmesh=kmesh)
    k_phase = _build_k_phase(cell, kpts)
    return {
        "cell": cell,
        "kpts": kpts,
        "kmesh": kmesh,
        "wrap_around": wrap_around,
        "scell": scell,
        "phase": phase,
        "phase_gpu": phase_gpu,
        "s_k": s_k,
        "s_sc_cpu": None,
        "k_phase": k_phase,
        "k_phase_gpu": cp.asarray(k_phase),
    }


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
        else:
            s_sc = _cpu_overlap(scell)
            if cache is not None:
                cache["s_sc_cpu"] = s_sc
        s = np.asarray(s_sc).real
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
            f = np.dot(c_deg * weights[None, :], c_deg.conj().T)
            if float(np.max(np.abs(f.imag))) < 1e-4:
                ev, na_orb = scipy.linalg.eigh(f.real, s, type=2)
                c_gamma = c_gamma.real
                c_gamma[:, degen_mask] = na_orb[:, ev > 1e-7]
                return e_g, c_gamma

        f = np.dot(c_gamma * np.asarray(e_g, dtype=c_gamma.dtype)[None, :], c_gamma.conj().T)
        if float(np.max(np.abs(f.imag))) < 1e-4:
            _, c_gamma = scipy.linalg.eigh(f.real, s, type=2)
            return e_g, c_gamma

    return e_g, c_gamma


def mo_k2gamma(cell, mo_energy, mo_coeff, kpts, kmesh=None, cache=None):
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
    c_k = _to_gpu_stack(mo_coeff)
    nk, nao, nmo = c_k.shape
    nr = phase.shape[0]

    phase_gpu = cache["phase_gpu"]
    k_phase_gpu = cache["k_phase_gpu"]
    phase_rot = lib.contraction("Rk", phase_gpu, "kh", k_phase_gpu, "Rkh")
    c_gamma = lib.contraction("Rkh", phase_rot, "kum", c_k, "Ruhm")
    c_gamma = c_gamma.reshape(nao * nr, nk * nmo)

    del phase_rot

    c_gamma_cpu = _asnumpy(c_gamma)
    del c_gamma

    e_g, c_gamma_cpu = _realify_mos_if_needed(
        scell, e_g, c_gamma_cpu, s_sc=cache.get("s_sc_cpu"), cache=cache
    )
    c_gamma = cp.asarray(c_gamma_cpu)
    del c_gamma_cpu

    s_k = cache["s_k"]
    s_k_g = lib.contraction("kuv", s_k, "Rk", phase_gpu, "kuRv", opb="CONJ")
    s_k_g = s_k_g.reshape(nk, nao, nr * nao)
    tmp = lib.contraction("kum", c_k, "kuv", s_k_g, "kmv", opa="CONJ")
    mo_phase = lib.contraction("kmv", tmp, "vi", c_gamma, "kmi")

    del s_k_g, tmp, c_k
    return scell, e_g, _asnumpy(c_gamma), _asnumpy(mo_phase)


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
            kmf.cell, mo_energy, mo_coeff, kpts, kmesh, cache=cache
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

    phase_pair = lib.contraction("Rk", phase_gpu, "Sk", phase_gpu, "RSk", opb="CONJ")
    ao_sc = lib.contraction("RSk", phase_pair, "kij", ao_ints_gpu, "RiSj")
    ao_sc = ao_sc.reshape(scell.nao_nr(), scell.nao_nr())

    if force_real and float(cp.max(cp.abs(ao_sc.imag)).get()) < 1e-5:
        return _asnumpy(ao_sc.real)
    return _asnumpy(ao_sc)
