import gc
import functools
import os

import cupy as cp
import cupyx
import numpy as np
from pyscf.lib import prange
from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh

from byteqc import lib
from gpu4pyscf.lib import logger as g_logger
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
cp.cuda.set_pinned_memory_allocator(None)

AO_PAIR_BATCH_SIZE_1 = 256 * 256
GBLKSIZE_1 = 1024 * 4
KBLKSIZE_1 = 16
AO_PAIR_BATCH_SIZE_2 = 256 * 256
GBLKSIZE_2 = 1024 * 4
KBLKSIZE_2 = 16
IMAG_TOL = 1e-10
DEFAULT_SVD_COMPRESS_AUX = 30000


_ERI_MEMORY_POOL_STACK = []


def _free_current_eri_pool_blocks():
    cp.cuda.get_current_stream().synchronize()
    if _ERI_MEMORY_POOL_STACK:
        _ERI_MEMORY_POOL_STACK[-1].free_all_blocks()
    lib.free_all_blocks()
    gc.collect()


def _with_eri_memory_pool(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        pool = cp.cuda.MemoryPool()
        _ERI_MEMORY_POOL_STACK.append(pool)
        try:
            with cp.cuda.using_allocator(pool.malloc):
                return func(*args, **kwargs)
        finally:
            cp.cuda.get_current_stream().synchronize()
            pool.free_all_blocks()
            _ERI_MEMORY_POOL_STACK.pop()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            lib.free_all_blocks()
            del pool
            gc.collect()

    return wrapped


def _sqrt_psd_eigs_or_raise(eigs, min_eig_allowed=-1e-10):
    smin = float(cp.min(eigs))
    if smin < min_eig_allowed:
        smax = float(cp.max(eigs))
        raise RuntimeError(
            f"Encountered significant negative eigenvalue: min={smin}, "
            f"allowed_min={min_eig_allowed}, max={smax}"
        )
    if smin < 0.0:
        eigs = cp.maximum(eigs, 0.0)
    return cp.sqrt(eigs)


def _as_host_canonical_matrix(arr, *, name):
    if isinstance(arr, cp.ndarray):
        host = cupyx.empty_pinned(arr.shape, dtype=np.float64, order="C")
        arr.get(out=host, blocking=True)
        arr = host
    elif not isinstance(arr, np.ndarray):
        arr_np = np.asarray(arr, dtype=np.float64, order="C")
        host = cupyx.empty_pinned(arr_np.shape, dtype=np.float64, order="C")
        host[:] = arr_np
        arr = host
    elif arr.dtype != np.float64 or arr.ndim != 2 or not arr.flags.c_contiguous:
        arr_np = np.asarray(arr, dtype=np.float64, order="C")
        host = cupyx.empty_pinned(arr_np.shape, dtype=np.float64, order="C")
        host[:] = arr_np
        arr = host

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D host matrix; got shape {arr.shape}")
    return arr


def allocate_incremental_aux_svd_buffers(
    nov_tot,
    *,
    max_aux_eigh=None,
    max_aux_out=None,
    slice_len_ov=None,
):
    if nov_tot <= 0:
        raise ValueError(f"nov_tot must be positive; got {nov_tot}")
    if max_aux_eigh is None:
        max_aux_eigh = DEFAULT_SVD_COMPRESS_AUX
    max_aux_eigh = int(max_aux_eigh)
    if max_aux_eigh <= 0:
        raise ValueError(f"max_aux_eigh must be positive; got {max_aux_eigh}")
    if max_aux_out is None:
        max_aux_out = max_aux_eigh
    max_aux_out = int(max_aux_out)
    if max_aux_out <= 0:
        raise ValueError(f"max_aux_out must be positive; got {max_aux_out}")

    if slice_len_ov is None:
        avail_elems = max(1, int(lib.gpu_avail_bytes() // np.dtype(np.float64).itemsize))
        gram_elems = max_aux_eigh * max_aux_eigh
        usable_elems = max(1, avail_elems - gram_elems)
        denom = max(2 * max_aux_eigh + max_aux_out, 1)
        slice_len_ov = max(1, min(int(nov_tot), int(usable_elems // denom)))
    else:
        slice_len_ov = max(1, min(int(slice_len_ov), int(nov_tot)))

    return {
        "slice_len_ov": slice_len_ov,
        "max_aux_eigh": max_aux_eigh,
        "max_aux_out": max_aux_out,
        "proj_out_host_buf": cupyx.empty_pinned((slice_len_ov * max_aux_out,), dtype=np.float64),
    }


def compress_canonical_aux_matrix(
    cderi_cpu,
    *,
    svd_tol,
    buffers=None,
    out=None,
    logger=None,
    label="incremental_aux_svd",
):
    cderi_cpu = _as_host_canonical_matrix(cderi_cpu, name="cderi_cpu")
    nov_tot, naux = cderi_cpu.shape
    if nov_tot <= 0:
        raise ValueError("cderi_cpu must contain at least one row")
    if naux <= 0:
        raise ValueError("cderi_cpu must contain at least one auxiliary column")

    if buffers is None or buffers["max_aux_eigh"] < naux or buffers["max_aux_out"] < naux:
        work_buffers = allocate_incremental_aux_svd_buffers(
            nov_tot,
            max_aux_eigh=naux,
            max_aux_out=naux,
        )
        owns_buffers = True
    else:
        work_buffers = buffers
        owns_buffers = False

    slice_len_ov = max(1, min(int(work_buffers["slice_len_ov"]), nov_tot))
    ovslice_list = [slice(p0, p1) for p0, p1 in prange(0, nov_tot, slice_len_ov)]

    LL_svd = lib.empty_from_buf(None, (naux, naux), np.float64, order="F")
    LL_svd[:] = 0
    for sov in ovslice_list:
        sov_len = sov.stop - sov.start
        sov_L = lib.empty_from_buf(None, (sov_len, naux), np.float64)
        sov_L.set(cderi_cpu[sov])
        lib.gemm(sov_L, sov_L, transa="T", c=LL_svd, beta=1.0)
        sov_L = None

    _free_current_eri_pool_blocks()

    S, U_svd = cp.linalg._eigenvalue._syevd(LL_svd, "L", with_eigen_vector=True, overwrite_a=True)
    S = _sqrt_psd_eigs_or_raise(S, min_eig_allowed=-1e-10)
    sort_ind = S.argsort()[::-1]
    S = S[sort_ind]
    U_svd = U_svd[:, sort_ind]
    newind = cp.where(S > svd_tol)[0]
    if newind.size == 0:
        raise RuntimeError(
            f"{label}: no singular values above svd_tol={svd_tol}; max singular value={float(S.max())}"
        )
    naux_cut = int(newind.size)
    U_svd = cp.ascontiguousarray(U_svd[:, newind])

    if logger is not None:
        logger.info("%s, SVD cut aux basis size from : %d to : %d", label, naux, naux_cut)

    if out is None:
        cderi_cut = cupyx.empty_pinned((nov_tot, naux_cut), dtype=np.float64, order="C")
    else:
        if out.shape[0] != nov_tot or out.shape[1] < naux_cut:
            raise ValueError(
                f"Provided output buffer has incompatible shape {out.shape}; "
                f"expected ({nov_tot}, >= {naux_cut})"
            )
        cderi_cut = out[:, :naux_cut]

    for sov in ovslice_list:
        sov_len = sov.stop - sov.start
        ovs_L = lib.empty_from_buf(None, (sov_len, naux), np.float64)
        ovs_L.set(cderi_cpu[sov])
        cderi_cut_s = lib.gemm(ovs_L, U_svd, transa="N", transb="N")
        cderi_cut_s_h = lib.empty_from_buf(
            work_buffers["proj_out_host_buf"], (sov_len, naux_cut), np.float64
        )
        cderi_cut_s.get(out=cderi_cut_s_h, blocking=True)
        cderi_cut[sov] = cderi_cut_s_h
        ovs_L = None
        cderi_cut_s = None

    U_svd = None
    LL_svd = None
    _free_current_eri_pool_blocks()

    if owns_buffers:
        del work_buffers
        _free_current_eri_pool_blocks()

    return cderi_cut


def compress_canonical_aux_matrix_in_aux_chunks(
    cderi_cpu,
    *,
    svd_tol,
    buffers,
    logger=None,
    label="aux_chunk_svd",
    compress_aux=DEFAULT_SVD_COMPRESS_AUX,
    return_chunks=False,
):
    cderi_cpu = _as_host_canonical_matrix(cderi_cpu, name=f"{label}_input")
    nov_tot, naux_tot = cderi_cpu.shape
    compress_aux = int(compress_aux)
    if compress_aux <= 0:
        raise ValueError(f"compress_aux must be positive; got {compress_aux}")

    chunks = []
    for p0 in range(0, naux_tot, compress_aux):
        p1 = min(naux_tot, p0 + compress_aux)
        chunk = compress_canonical_aux_matrix(
            cderi_cpu[:, p0:p1],
            svd_tol=svd_tol,
            buffers=buffers,
            logger=logger,
            label=f"{label} aux[{p0}:{p1}]",
        )
        chunk = _as_host_canonical_matrix(chunk, name=f"{label}_chunk")
        chunks.append(chunk)

    if return_chunks:
        return chunks

    return concatenate_canonical_aux_chunks(chunks, nov_tot=nov_tot)


def concatenate_canonical_aux_chunks(chunks, *, nov_tot=None):
    if len(chunks) == 1:
        return chunks[0]

    if nov_tot is None:
        nov_tot = int(chunks[0].shape[0])
    naux_cut_tot = sum(int(chunk.shape[1]) for chunk in chunks)
    out = cupyx.empty_pinned((nov_tot, naux_cut_tot), dtype=np.float64, order="C")
    pos = 0
    for chunk in chunks:
        width = int(chunk.shape[1])
        out[:, pos:pos + width] = chunk
        pos += width
    chunks = None
    gc.collect()
    return out


def format_canonical_aux_for_solver(cderi_cpu, *, solver_type, out=None):
    cderi_cpu = _as_host_canonical_matrix(cderi_cpu, name="cderi_cpu")
    if solver_type == "MP2":
        if out is None:
            return cderi_cpu
        if out.shape != cderi_cpu.shape:
            raise ValueError(
                f"Provided MP2 output buffer shape {out.shape} does not match {cderi_cpu.shape}"
            )
        out[:] = cderi_cpu
        return out

    if "CC" not in solver_type:
        raise ValueError(f"Unsupported solver_type: {solver_type}")

    nrow, ncol = cderi_cpu.shape[1], cderi_cpu.shape[0]
    if out is None:
        out = cupyx.empty_pinned((nrow, ncol), dtype=np.float64, order="C")
    elif out.shape != (nrow, ncol):
        raise ValueError(
            f"Provided CC output buffer shape {out.shape} does not match {(nrow, ncol)}"
        )
    out[:] = cderi_cpu.T
    return out


def _to_numpy_array(arr, dtype=None):
    if isinstance(arr, cp.ndarray):
        arr = cp.asnumpy(arr)
    return np.asarray(arr, dtype=dtype)


def _max_abs(arr):
    return float(np.max(np.abs(arr))) if np.size(arr) else 0.0


def _env_flag(name, default="0"):
    return bool(int(os.environ.get(name, default)))


def _debug_imag_check_enabled():
    return _env_flag(
        "BYTEQC_ERI_TRANS_PBC_DEBUG_IMAG_CHECK",
        os.environ.get("BYTEQC_ERI_TRANS_PBC_DEBUG_SYNC", "0"),
    )


def _get_t_reversal_weights(cell, kpts, tol=1e-9):
    nkpts = len(kpts)
    scaled = np.asarray(cell.get_scaled_kpts(kpts), dtype=float)
    weights = np.ones(nkpts, dtype=np.int32)
    for i in range(nkpts):
        if weights[i] == 0:
            continue
        for j in range(i + 1, nkpts):
            if weights[j] == 0:
                continue
            delta = scaled[i] + scaled[j]
            delta -= np.round(delta)
            if _max_abs(delta) < tol:
                weights[i] = 2
                weights[j] = 0
                break
    return weights


def _gamma_ao_to_k_coeff(cell, kpts, mo_coeff, kmesh, wrap_around=False):
    mo_coeff = _to_numpy_array(mo_coeff)
    if mo_coeff.ndim != 2:
        raise ValueError(f"mo_coeff must be a 2D array; got {mo_coeff.shape}")

    kpts = _to_numpy_array(kpts, dtype=float)
    nkpts = len(kpts)
    nao = int(cell.nao_nr())
    nao_sc = int(mo_coeff.shape[0])
    if nao_sc != nkpts * nao:
        raise ValueError(
            "The input mo_coeff must be expressed in gamma-supercell AO basis: "
            f"expected AO dimension {nkpts * nao}, got {nao_sc}"
        )

    nmo = int(mo_coeff.shape[1])
    mo_coeff = np.asarray(mo_coeff, dtype=np.complex128, order="C")
    mo_coeff = mo_coeff.reshape(nkpts, nao, nmo)

    phase = np.exp(
        1j * np.asarray(translation_vectors_for_kmesh(cell, kmesh, wrap_around), dtype=float).dot(kpts.T)
    )
    phase /= np.sqrt(nkpts)
    coeff_k = np.empty((nkpts, nao, nmo), dtype=np.complex128, order="C")
    lib.contraction("Rk", phase.conj(), "Rui", mo_coeff, "kui", coeff_k)
    return coeff_k


def _apply_C_dot_kwise_host(cell, coeff_k, nao_out):
    coeff_k = np.asarray(coeff_k, dtype=np.complex128, order="C")
    if coeff_k.ndim != 3:
        raise ValueError(f"Expected k-resolved coefficients to be 3D; got {coeff_k.shape}")
    nkpts, nao_in, nmo = coeff_k.shape
    coeff_out = cupyx.empty_pinned((nkpts, int(nao_out), nmo), dtype=np.complex128, order="C")
    for k in range(nkpts):
        coeff_d = cp.asarray(coeff_k[k], dtype=np.complex128, order="C")
        coeff_t = cell.apply_C_dot(coeff_d.reshape(1, nao_in, nmo), axis=1)
        coeff_t = coeff_t.reshape(int(nao_out), nmo)
        coeff_t.get(out=coeff_out[k], blocking=True)
        coeff_d = None
        coeff_t = None
        _free_current_eri_pool_blocks()
    return coeff_out


def _accumulate_real_aux_channels(dst, src, *, weight, sqrt2, imag_sign=1.0):
    naux = int(src.shape[-1])
    if weight == 1:
        dst[..., :naux] += src.real
        return 0.0
    if weight == 2:
        # src is a disposable host staging block; scale in-place to avoid large temporaries.
        src *= sqrt2
        dst[..., :naux] += src.real
        if imag_sign == 1.0:
            dst[..., naux:2 * naux] += src.imag
        elif imag_sign == -1.0:
            dst[..., naux:2 * naux] -= src.imag
        else:
            dst[..., naux:2 * naux] += imag_sign * src.imag
        return 0.0
    raise ValueError(f"Unsupported time-reversal weight {weight}")


def _update_self_conj_imag_stats(src, imag_max, real_at_imag_max, info_at_imag_max, info):
    this_imag = _max_abs(src.imag)
    if this_imag <= imag_max:
        return imag_max, real_at_imag_max, info_at_imag_max
    return this_imag, _max_abs(src.real), info


def _warn_self_conj_imag(log, label, k_aux, imag_max, real_at_imag_max, imag_tol, info):
    if imag_max <= imag_tol:
        return
    rel = imag_max / max(real_at_imag_max, 1e-300)
    if info is None:
        log.warn(
            "%s q=%d is self-conjugate but max|imag|=%.3e exceeds %.3e; "
            "max|real|=%.3e rel=%.3e; discarding imag part",
            label,
            k_aux,
            imag_max,
            imag_tol,
            real_at_imag_max,
            rel,
        )
    else:
        log.warn(
            "%s q=%d is self-conjugate but max|imag|=%.3e exceeds %.3e; "
            "max|real|=%.3e rel=%.3e at %s; discarding imag part",
            label,
            k_aux,
            imag_max,
            imag_tol,
            real_at_imag_max,
            rel,
            info,
        )


def _log_self_conj_imag_pass(log, label, k_aux, imag_max, real_at_imag_max, imag_tol, info):
    rel = imag_max / max(real_at_imag_max, 1e-300)
    if info is None:
        log.info(
            "%s q=%d self-conjugate aggregate imag check passed: "
            "max|imag|=%.3e <= %.3e, max|real|=%.3e rel=%.3e",
            label,
            k_aux,
            imag_max,
            imag_tol,
            real_at_imag_max,
            rel,
        )
    else:
        log.info(
            "%s q=%d self-conjugate aggregate imag check passed: "
            "max|imag|=%.3e <= %.3e, max|real|=%.3e rel=%.3e at %s",
            label,
            k_aux,
            imag_max,
            imag_tol,
            real_at_imag_max,
            rel,
            info,
        )


def _log_self_conj_imag_result(log, label, k_aux, imag_max, real_at_imag_max, imag_tol, info):
    if imag_max > imag_tol:
        _warn_self_conj_imag(log, label, k_aux, imag_max, real_at_imag_max, imag_tol, info)
    else:
        _log_self_conj_imag_pass(log, label, k_aux, imag_max, real_at_imag_max, imag_tol, info)


def _build_full_q_cderi_host(
    *,
    log,
    label,
    j2c_idx,
    k_aux,
    naux_q,
    shl_pair_batches,
    ao_pair_offsets,
    naux_cart,
    eval_j3c,
    expLk_conjz_full,
    aux_coeff_q,
    with_long_range,
    g_slices,
    auxG_conj,
    eval_ft,
    Gv,
    kpts,
    cderi_q_full_host_buf,
    cderi_q_batch_host_buf,
    debug_sync,
):
    npair_total = int(ao_pair_offsets[-1]) if len(ao_pair_offsets) else 0
    q_cderi_host = lib.empty_from_buf(
        cderi_q_full_host_buf, (naux_q, npair_total), np.complex128
    )

    for batch_id in range(shl_pair_batches):
        p0 = int(ao_pair_offsets[batch_id])
        p1 = int(ao_pair_offsets[batch_id + 1])
        if p1 <= p0:
            continue

        raw_j3c = eval_j3c(shl_pair_batch_id=batch_id)
        pair_size = int(p1 - p0)
        if raw_j3c.shape[1] != naux_cart:
            raise RuntimeError(
                f"Unexpected raw_j3c auxiliary dimension {raw_j3c.shape[1]}; expected {naux_cart}"
            )

        q_j3c_z = lib.empty_from_buf(None, (1, naux_cart, pair_size, 2), np.float64)
        lib.contraction(
            "prL",
            raw_j3c,
            "LKz",
            expLk_conjz_full[:, k_aux:k_aux + 1],
            "Krpz",
            q_j3c_z,
        )
        q_j3c = q_j3c_z.view(np.complex128).reshape(naux_cart, pair_size)

        if with_long_range:
            for g0, g1 in g_slices:
                auxG_c = auxG_conj[:, j2c_idx, g0:g1]
                pqG = eval_ft(Gv[g0:g1] + kpts[k_aux], batch_id)
                lib.contraction("rG", auxG_c, "pG", pqG, "rp", q_j3c, beta=1.0)
                pqG = None
                auxG_c = None

        cderi_q_pair = lib.empty_from_buf(None, (naux_q, pair_size), np.complex128)
        lib.contraction("Lr", aux_coeff_q.T, "rp", q_j3c, "Lp", cderi_q_pair)
        if debug_sync:
            cp.cuda.get_current_stream().synchronize()

        cderi_q_batch_host = lib.empty_from_buf(
            cderi_q_batch_host_buf, (naux_q, pair_size), np.complex128
        )
        cderi_q_pair.get(out=cderi_q_batch_host, blocking=True)
        q_cderi_host[:, p0:p1] = cderi_q_batch_host

        raw_j3c = None
        q_j3c_z = None
        q_j3c = None
        cderi_q_pair = None
        cderi_q_batch_host = None

    log.info(
        "%s k_aux=%d accumulated full cderi_q on host with shape (%d, %d)",
        label,
        k_aux,
        naux_q,
        npair_total,
    )
    return q_cderi_host


@_with_eri_memory_pool
def eri_OVL_SIE_MP2(
    cell,
    auxcell,
    mo_coeff_i1,
    mo_coeff_j1,
    mo_coeff_i2,
    mo_coeff_j2,
    *args,
    omega=None,
    linear_dep_threshold=LINEAR_DEP_THR,
    with_long_range=True,
    ao_pair_batch_size=AO_PAIR_BATCH_SIZE_1,
    Lblksize=KBLKSIZE_1,
    Gblksize=GBLKSIZE_1,
    kpts=None,
    kmesh=None,
    wrap_around=False,
    imag_tol=IMAG_TOL,
):
    """Build gamma-supercell ovL/voL factors from primitive-cell k-point RSDF on the fly.

    This is the PBC analogue of ``byteqc.embyte.ERI.eri_trans_gpu4pyscf.eri_OVL_SIE_MP2``.
    The first positional arguments and the returned objects are kept aligned with
    the gamma-point implementation. The extra mandatory keyword ``kpts`` supplies
    the primitive-cell Monkhorst-Pack mesh used to build the ERI on the fly.

    Important convention:
    ``mo_coeff_i1``, ``mo_coeff_j1``, ``mo_coeff_i2`` and ``mo_coeff_j2`` are
    interpreted as gamma-supercell AO to target-space coefficient matrices.
    Internally, this routine Fourier-transforms them back to primitive-cell
    k-space coefficients and contracts them directly with the k-point CDERI.
    """
    if len(args) == 1:
        j2c = None
        logger = args[0]
    elif len(args) == 2:
        j2c, logger = args
    else:
        raise TypeError(
            'eri_OVL_SIE_MP2 expects either (..., logger) or (..., j2c, logger)'
        )
    _ = j2c

    _free_current_eri_pool_blocks()

    if cell.dimension != 3:
        raise NotImplementedError("The current multi-k eri_OVL_SIE_MP2 path only supports 3D cells")
    if auxcell is None:
        raise ValueError("auxcell is required")
    if kpts is None:
        raise ValueError("kpts must be provided for the multi-k eri_OVL_SIE_MP2 path")

    log = logger if logger is not None else g_logger.new_logger(cell, cell.verbose)
    debug_sync = _env_flag("BYTEQC_ERI_TRANS_PBC_DEBUG_SYNC")
    debug_imag_check = _debug_imag_check_enabled()
    if debug_imag_check:
        log.info(
            "eri_OVL_SIE_MP2 self-conjugate aggregate imag check is enabled "
            "(BYTEQC_ERI_TRANS_PBC_DEBUG_IMAG_CHECK=1)"
        )

    if hasattr(kpts, "kpts"):
        kpts = kpts.kpts
    kpts = _to_numpy_array(kpts, dtype=float)
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise ValueError(f"Invalid k-point shape: {kpts.shape}")
    nkpts = len(kpts)
    if nkpts <= 1:
        raise NotImplementedError("Gamma-only calculations should use eri_trans_gpu4pyscf.eri_OVL_SIE_MP2")

    if kmesh is None:
        kmesh = _to_numpy_array(kpts_to_kmesh(cell, kpts), dtype=int)
    else:
        kmesh = _to_numpy_array(kmesh, dtype=int)
    if int(np.prod(kmesh)) != nkpts:
        raise ValueError(f"kmesh {tuple(kmesh)} is incompatible with {nkpts} k-points")
    coeff_i1_k = _gamma_ao_to_k_coeff(cell, kpts, mo_coeff_i1, kmesh, wrap_around=wrap_around)
    coeff_j1_k = _gamma_ao_to_k_coeff(cell, kpts, mo_coeff_j1, kmesh, wrap_around=wrap_around)
    coeff_i2_k = _gamma_ao_to_k_coeff(cell, kpts, mo_coeff_i2, kmesh, wrap_around=wrap_around)
    coeff_j2_k = _gamma_ao_to_k_coeff(cell, kpts, mo_coeff_j2, kmesh, wrap_around=wrap_around)

    nmo_i1 = int(coeff_i1_k.shape[2])
    nmo_j1 = int(coeff_j1_k.shape[2])
    nmo_i2 = int(coeff_i2_k.shape[2])
    nmo_j2 = int(coeff_j2_k.shape[2])

    if omega is None:
        omega = OMEGA_MIN
    omega = abs(float(omega))

    int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega=-omega, bvk_kmesh=kmesh).build()
    ao_cell = getattr(int3c2e_opt.cell, "mol", cell)
    ao_nao = int(ao_cell.nao)
    int3c_nao = int(int3c2e_opt.cell.nao_nr(cart=True))
    if coeff_i1_k.shape[1] != ao_nao:
        raise ValueError(
            "Primitive-cell AO dimension mismatch: coefficient AO dimension "
            f"{coeff_i1_k.shape[1]} vs AO-space coefficient dimension {ao_nao}"
        )
    nao_unpack = int3c_nao
    if int3c_nao != ao_nao:
        log.info(
            "eri_OVL_SIE_MP2 internal AO dimension differs from external AO dimension: "
            "int3c_nao=%d ao_nao=%d; AO coefficients will be transformed to the "
            "sorted internal AO basis before AO2MO",
            int3c_nao,
            ao_nao,
        )

    coeff_i1_k = _apply_C_dot_kwise_host(int3c2e_opt.cell, coeff_i1_k, nao_unpack)
    coeff_j1_k = _apply_C_dot_kwise_host(int3c2e_opt.cell, coeff_j1_k, nao_unpack)
    coeff_i2_k = _apply_C_dot_kwise_host(int3c2e_opt.cell, coeff_i2_k, nao_unpack)
    coeff_j2_k = _apply_C_dot_kwise_host(int3c2e_opt.cell, coeff_j2_k, nao_unpack)

    cd_j2c_cache, negative_metric_size = _precontract_j2c_aux_coeff(
        int3c2e_opt.auxcell,
        kpts,
        omega,
        with_long_range,
        linear_dep_threshold,
        kmesh,
    )
    if any(int(x) > 0 for x in negative_metric_size.values()):
        raise NotImplementedError("Negative-metric auxiliary sectors are not supported in this first multi-k SIE path")

    kpt_iters = list(kk_adapted_iter(kmesh))
    stored_kpts = kpts[[x[0] for x in kpt_iters]]
    nstored_kpts = len(kpt_iters)
    if nstored_kpts <= 0:
        raise RuntimeError("Failed to identify independent q sectors")

    weights = _get_t_reversal_weights(cell, kpts)
    aux_coeffs = []
    for coeff in cd_j2c_cache:
        aux_coeffs.append(cp.asarray(coeff, dtype=np.complex128, order="C"))
    naux_cart = int(cd_j2c_cache[0].shape[0])

    total_naux_real = 0
    active_kblk_sizes = []
    active_naux_q = []
    for idx, (k_aux, *_rest) in enumerate(kpt_iters):
        weight = int(weights[k_aux])
        if weight > 0:
            naux_q_i = int(aux_coeffs[idx].shape[1])
            total_naux_real += weight * naux_q_i
            active_naux_q.append(naux_q_i)
            if Lblksize is None:
                active_kblk_sizes.append(naux_q_i)
            else:
                active_kblk_sizes.append(max(1, min(int(Lblksize), naux_q_i)))

    ovL_cpu = cupyx.zeros_pinned((nmo_i1, nmo_j1, total_naux_real), dtype=np.float64, order="C")
    voL_cpu = cupyx.zeros_pinned((nmo_j2, nmo_i2, total_naux_real), dtype=np.float64, order="C")

    nsp_per_block = ft_ao.ft_ao_scheme()[0]
    bas_ij_aggregated = int3c2e_opt.cell.aggregate_shl_pairs(int3c2e_opt.bas_ij_cache, nsp_per_block)

    cderi_idx = int3c2e_opt.pair_and_diag_indices(cart=True, original_ao_order=False)
    pair_address = _to_numpy_array(cderi_idx[0], dtype=np.int32)
    eval_j3c, aux_sorting, ao_pair_offsets = int3c2e_opt.int3c2e_evaluator(
        ao_pair_batch_size=ao_pair_batch_size,
        cart=True,
        bas_ij_aggregated=bas_ij_aggregated,
    )[:3]
    ao_pair_offsets = _to_numpy_array(ao_pair_offsets, dtype=np.int64)
    shl_pair_batches = int(len(ao_pair_offsets) - 1)
    pair_sizes = np.diff(ao_pair_offsets).astype(np.int64, copy=False)
    max_pair_size = int(pair_sizes.max()) if pair_sizes.size else 0
    npair_total = int(ao_pair_offsets[-1]) if len(ao_pair_offsets) else 0

    for idx, coeff in enumerate(aux_coeffs):
        coeff_sorted = cp.empty_like(coeff)
        coeff_sorted[aux_sorting] = coeff
        aux_coeffs[idx] = coeff_sorted

    expLk_full = cp.exp(1j * cp.asarray(int3c2e_opt.bvkmesh_Ls.dot(kpts.T)))
    expLk_conjz_full = expLk_full.conj().view(np.float64).reshape(int(np.prod(kmesh)), nkpts, 2)
    conj_mapping = _to_numpy_array(conj_images_in_bvk_cell(kmesh), dtype=np.int32)
    nL_full = int(expLk_full.shape[0])
    pair_upper_bound = nao_unpack * nL_full * nao_unpack
    if pair_address.size:
        pair_max = int(pair_address.max())
        pair_min = int(pair_address.min())
        if pair_min < 0 or pair_max >= pair_upper_bound:
            raise RuntimeError(
                "pair_address is incompatible with unpack dimensions: "
                f"min={pair_min}, max={pair_max}, upper_bound={pair_upper_bound}, "
                f"nao_unpack={nao_unpack}, nL={nL_full}, nkpts={nkpts}"
            )
    if expLk_full.shape != (int(np.prod(kmesh)), nkpts):
        raise RuntimeError(
            f"Unexpected expLk shape {expLk_full.shape}; expected {(int(np.prod(kmesh)), nkpts)}"
        )
    if conj_mapping.shape != (nkpts,):
        raise RuntimeError(
            f"Unexpected conjugation mapping shape {conj_mapping.shape}; expected {(nkpts,)}"
        )
    pair_address_gpu = cp.asarray(pair_address, dtype=np.int32)
    conj_mapping_gpu = cp.asarray(conj_mapping, dtype=np.int32)

    if with_long_range:
        mesh = int3c2e_opt.mesh
        Gv, _Gvbase, kws = cell.get_Gv_weights(mesh)
        ngrids = len(Gv)
        Gk = (Gv + stored_kpts[:, None]).reshape(-1, 3)
        Gk = _Gv_wrap_around(cell, Gk, cp.zeros(3), mesh)
        coulG = get_coulG(cell, Gv=Gk, omega=omega).reshape(nstored_kpts, ngrids)
        coulG *= kws
        coulG[0, 0] -= np.pi / omega**2 / cell.vol

        ft_opt = ft_ao.FTOpt.from_intopt(int3c2e_opt)
        eval_ft, ao_pair_offsets_ft = ft_opt.ft_evaluator(
            ao_pair_batch_size,
            cart=True,
            original_ao_order=False,
            bas_ij_aggregated=bas_ij_aggregated,
        )
        ao_pair_offsets_ft = _to_numpy_array(ao_pair_offsets_ft, dtype=np.int64)
        if not np.array_equal(ao_pair_offsets, ao_pair_offsets_ft):
            raise RuntimeError("AO-pair offsets mismatch between SR and LR evaluators")

        auxG = ft_ao.ft_ao(int3c2e_opt.auxcell, Gk).T
        auxG = auxG.reshape(naux_cart, nstored_kpts, ngrids)
        for k in range(nstored_kpts):
            auxG[aux_sorting, k] = auxG[:, k].conj()
        auxG_conj = auxG
        auxG_conj *= cp.asarray(coulG)

        if Gblksize is None:
            mem_free = cp.cuda.runtime.memGetInfo()[0]
            mem_free -= sum(x.nbytes for x in aux_coeffs)
            mem_free -= nkpts * naux_cart * ao_pair_batch_size * 16 * 2
            Gblksize = int(mem_free // max(16 * ao_pair_batch_size, 1)) // 32 * 32
            Gblksize = max(32, min(Gblksize, ngrids))
        g_slices = list(prange(0, ngrids, Gblksize))
    else:
        eval_ft = None
        Gv = None
        ngrids = 0
        auxG_conj = None
        g_slices = []

    # q-space auxiliary factors B_q are orthonormal to each other. Returning
    # gamma-supercell real-space factors requires the inverse k2gamma transform
    # on the auxiliary index, which contributes a global 1/sqrt(Nk) factor to
    # each ovL/voL block. Without this normalization the reconstructed 4-index
    # ERI is uniformly larger by Nk.
    pair_norm = 1.0 / np.sqrt(nkpts)

    max_naux_q = max(active_naux_q) if active_naux_q else 0
    max_kblk = max(active_kblk_sizes) if active_kblk_sizes else 0
    npair_total = int(ao_pair_offsets[-1]) if len(ao_pair_offsets) else 0
    if max_pair_size > 0 and max_kblk > 0 and max_naux_q > 0:
        cderi_q_full_host_buf = cupyx.empty_pinned((max_naux_q, npair_total), dtype=np.complex128, order="C")
        cderi_q_batch_host_buf = cupyx.empty_pinned((max_naux_q, max_pair_size), dtype=np.complex128, order="C")
        ov_host_buf = cupyx.empty_pinned((nmo_i1 * nmo_j1 * max_kblk,), dtype=np.complex128)
        vo_host_buf = cupyx.empty_pinned((nmo_j2 * nmo_i2 * max_kblk,), dtype=np.complex128)
        ov_debug_sum_buf = (
            cupyx.empty_pinned((nmo_i1, nmo_j1, max_kblk), dtype=np.complex128, order="C")
            if debug_imag_check else None
        )
        vo_debug_sum_buf = (
            cupyx.empty_pinned((nmo_j2, nmo_i2, max_kblk), dtype=np.complex128, order="C")
            if debug_imag_check else None
        )
    else:
        cderi_q_full_host_buf = None
        cderi_q_batch_host_buf = None
        ov_host_buf = None
        vo_host_buf = None
        ov_debug_sum_buf = None
        vo_debug_sum_buf = None

    sqrt2 = np.sqrt(2.0)
    l0 = 0
    for j2c_idx, (k_aux, _k_aux_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
        weight = int(weights[k_aux])
        if weight <= 0:
            continue

        k_aux = int(k_aux)
        ki_idx = _to_numpy_array(ki_idx, dtype=np.int32)
        kj_idx = _to_numpy_array(kj_idx, dtype=np.int32)
        if ki_idx.shape != (nkpts,) or kj_idx.shape != (nkpts,):
            raise RuntimeError(
                f"k-point index permutation for q={k_aux} has wrong shape: "
                f"ki_idx.shape={ki_idx.shape}, kj_idx.shape={kj_idx.shape}, expected {(nkpts,)}"
            )
        if not np.array_equal(np.sort(ki_idx), np.arange(nkpts, dtype=np.int32)):
            raise RuntimeError(f"ki_idx for q={k_aux} is not a permutation of [0, {nkpts})")
        if not np.array_equal(np.sort(kj_idx), np.arange(nkpts, dtype=np.int32)):
            raise RuntimeError(f"kj_idx for q={k_aux} is not a permutation of [0, {nkpts})")
        kj_idx_gpu = cp.asarray(kj_idx, dtype=np.int32)
        naux_q = int(aux_coeffs[j2c_idx].shape[1])
        Kblksize = naux_q if Lblksize is None else max(1, min(int(Lblksize), naux_q))

        log.info(
            "eri_OVL_SIE_MP2 multi-k q=%d/%d k_aux=%d weight=%d naux=%d",
            j2c_idx + 1,
            len(kpt_iters),
            k_aux,
            weight,
            naux_q,
        )
        ov_imag_max = 0.0
        vo_imag_max = 0.0
        ov_imag_real_max = 0.0
        vo_imag_real_max = 0.0
        ov_imag_info = None
        vo_imag_info = None

        q_cderi_host = _build_full_q_cderi_host(
            log=log,
            label="eri_OVL_SIE_MP2",
            j2c_idx=j2c_idx,
            k_aux=k_aux,
            naux_q=naux_q,
            shl_pair_batches=shl_pair_batches,
            ao_pair_offsets=ao_pair_offsets,
            naux_cart=naux_cart,
            eval_j3c=eval_j3c,
            expLk_conjz_full=expLk_conjz_full,
            aux_coeff_q=aux_coeffs[j2c_idx],
            with_long_range=with_long_range,
            g_slices=g_slices,
            auxG_conj=auxG_conj,
            eval_ft=eval_ft,
            Gv=Gv,
            kpts=kpts,
            cderi_q_full_host_buf=cderi_q_full_host_buf,
            cderi_q_batch_host_buf=cderi_q_batch_host_buf,
            debug_sync=debug_sync,
        )
        _free_current_eri_pool_blocks()
        for K0 in range(0, naux_q, Kblksize):
            K1 = min(naux_q, K0 + Kblksize)
            block_len = K1 - K0
            cderi_q = lib.empty_from_buf(None, (block_len, npair_total), np.complex128)
            cderi_q.set(q_cderi_host[K0:K1])
            if weight == 1 and debug_imag_check:
                ov_debug_sum = lib.empty_from_buf(
                    ov_debug_sum_buf, (nmo_i1, nmo_j1, block_len), np.complex128
                )
                vo_debug_sum = lib.empty_from_buf(
                    vo_debug_sum_buf, (nmo_j2, nmo_i2, block_len), np.complex128
                )
                ov_debug_sum[:] = 0.0
                vo_debug_sum[:] = 0.0
            else:
                ov_debug_sum = None
                vo_debug_sum = None
            if debug_sync:
                cp.cuda.get_current_stream().synchronize()

            try:
                unpacked = rsdf_builder._unpack_cderi_v2(
                    cderi_q,
                    pair_address_gpu,
                    kj_idx_gpu,
                    conj_mapping_gpu,
                    expLk_full,
                    nao_unpack,
                    axis=0,
                )
                if debug_sync:
                    cp.cuda.get_current_stream().synchronize()
            except Exception as err:
                raise RuntimeError(
                    "Failed in _unpack_cderi_v2 with "
                    f"q={k_aux}, K-range=[{K0}:{K1}), "
                    f"nao_unpack={nao_unpack}, ao_nao={ao_nao}, nkpts={nkpts}, nL={nL_full}, "
                    f"npair_total={npair_total}, len(kj_idx)={len(kj_idx)}, "
                    f"expLk_shape={tuple(expLk_full.shape)}"
                ) from err

            for ki, kj in zip(ki_idx, kj_idx):
                ao_pair = unpacked[ki]
                ov_host = lib.empty_from_buf(ov_host_buf, (nmo_i1, nmo_j1, block_len), np.complex128)
                vo_host = lib.empty_from_buf(vo_host_buf, (nmo_j2, nmo_i2, block_len), np.complex128)

                coeff_i1_d = cp.asarray(coeff_i1_k[int(ki)], dtype=np.complex128, order="C")
                coeff_j1_d = cp.asarray(coeff_j1_k[int(kj)], dtype=np.complex128, order="C")
                tmp = lib.empty_from_buf(None, (nmo_i1, block_len, nao_unpack), np.complex128)
                lib.contraction(
                    "Lpq",
                    ao_pair,
                    "pi",
                    coeff_i1_d,
                    "iLq",
                    tmp,
                    opb="CONJ",
                )
                ov_blk = lib.empty_from_buf(None, (nmo_i1, nmo_j1, block_len), np.complex128)
                lib.contraction(
                    "iLq",
                    tmp,
                    "qj",
                    coeff_j1_d,
                    "ijL",
                    ov_blk,
                    alpha=pair_norm,
                )
                ov_blk.get(out=ov_host, blocking=True)
                coeff_i1_d = None
                coeff_j1_d = None
                tmp = None
                ov_blk = None

                coeff_j2_d = cp.asarray(coeff_j2_k[int(ki)], dtype=np.complex128, order="C")
                coeff_i2_d = cp.asarray(coeff_i2_k[int(kj)], dtype=np.complex128, order="C")
                tmp = lib.empty_from_buf(None, (nmo_j2, block_len, nao_unpack), np.complex128)
                lib.contraction(
                    "Lpq",
                    ao_pair,
                    "pj",
                    coeff_j2_d,
                    "jLq",
                    tmp,
                    opb="CONJ",
                )
                vo_blk = lib.empty_from_buf(None, (nmo_j2, nmo_i2, block_len), np.complex128)
                lib.contraction(
                    "jLq",
                    tmp,
                    "qi",
                    coeff_i2_d,
                    "jiL",
                    vo_blk,
                    alpha=pair_norm,
                )
                vo_blk.get(out=vo_host, blocking=True)
                coeff_j2_d = None
                coeff_i2_d = None
                tmp = None
                vo_blk = None

                real_slice = slice(l0 + K0, l0 + K1)
                if weight == 1:
                    ovL_cpu[:, :, real_slice] += ov_host.real
                    voL_cpu[:, :, real_slice] += vo_host.real
                    if debug_imag_check:
                        ov_debug_sum += ov_host
                        vo_debug_sum += vo_host
                elif weight == 2:
                    imag_slice = slice(l0 + naux_q + K0, l0 + naux_q + K1)
                    ov_host *= sqrt2
                    vo_host *= sqrt2
                    ovL_cpu[:, :, real_slice] += ov_host.real
                    ovL_cpu[:, :, imag_slice] += ov_host.imag
                    voL_cpu[:, :, real_slice] += vo_host.real
                    voL_cpu[:, :, imag_slice] += vo_host.imag
                else:
                    raise ValueError(f"Unsupported time-reversal weight {weight} for q={k_aux}")

                ao_pair = None
                ov_host = None
                vo_host = None

            unpacked = None
            cderi_q = None
            if weight == 1 and debug_imag_check:
                info = "K=[%d:%d)" % (K0, K1)
                ov_imag_max, ov_imag_real_max, ov_imag_info = _update_self_conj_imag_stats(
                    ov_debug_sum, ov_imag_max, ov_imag_real_max, ov_imag_info, info
                )
                vo_imag_max, vo_imag_real_max, vo_imag_info = _update_self_conj_imag_stats(
                    vo_debug_sum, vo_imag_max, vo_imag_real_max, vo_imag_info, info
                )
                ov_debug_sum = None
                vo_debug_sum = None

        q_cderi_host = None

        if weight == 1:
            if debug_imag_check:
                _log_self_conj_imag_result(
                    log, "ovL", k_aux, ov_imag_max, ov_imag_real_max, imag_tol, ov_imag_info
                )
                _log_self_conj_imag_result(
                    log, "voL", k_aux, vo_imag_max, vo_imag_real_max, imag_tol, vo_imag_info
                )
            l0 += naux_q
        else:
            l0 += 2 * naux_q

        _free_current_eri_pool_blocks()

    del coeff_i1_k, coeff_j1_k, coeff_i2_k, coeff_j2_k
    del aux_coeffs, cd_j2c_cache, eval_j3c, ao_pair_offsets, expLk_full, expLk_conjz_full
    del bas_ij_aggregated, kpt_iters, stored_kpts, pair_address
    del cderi_q_full_host_buf, cderi_q_batch_host_buf
    del ov_host_buf, vo_host_buf
    del ov_debug_sum_buf, vo_debug_sum_buf
    if with_long_range:
        del Gv, auxG_conj, eval_ft
    _free_current_eri_pool_blocks()

    return ovL_cpu, voL_cpu


@_with_eri_memory_pool
def eri_high_level_solver_incore(
    cell,
    auxcell,
    mo_coeff_i,
    mo_coeff_j,
    *args,
    solver_type="MP2",
    svd_tol=1e-4,
    omega=None,
    linear_dep_threshold=LINEAR_DEP_THR,
    with_long_range=True,
    ao_pair_batch_size=AO_PAIR_BATCH_SIZE_1,
    Lblksize=KBLKSIZE_1,
    Gblksize=GBLKSIZE_1,
    kpts=None,
    kmesh=None,
    wrap_around=False,
    imag_tol=IMAG_TOL,
):
    """Build multi-k gamma-supercell ijL and apply the in-core SVD truncation.

    This is the PBC analogue of
    ``byteqc.embyte.ERI.eri_trans_gpu4pyscf.eri_high_level_solver_incore``.
    The positional interface and return semantics are kept aligned with the
    gamma-point implementation. The extra keyword ``kpts`` supplies the
    primitive-cell Monkhorst-Pack mesh used to build the ERI on the fly.

    The current implementation builds the real-space gamma-supercell ``(i, j, L)``
    factors directly from the primitive-cell multi-k CDERI path. Unlike
    ``eri_OVL_SIE_MP2(...)``, it does not build a second AO2MO channel that is
    irrelevant to the returned solver tensor.
    """
    if len(args) == 1:
        j2c = None
        logger = args[0]
    elif len(args) == 2:
        j2c, logger = args
    else:
        raise TypeError(
            'eri_high_level_solver_incore expects either (..., logger) or (..., j2c, logger)'
        )
    _ = j2c

    _free_current_eri_pool_blocks()

    if cell.dimension != 3:
        raise NotImplementedError("The current multi-k eri_high_level_solver_incore path only supports 3D cells")
    if auxcell is None:
        raise ValueError("auxcell is required")
    if kpts is None:
        raise ValueError("kpts must be provided for the multi-k eri_high_level_solver_incore path")
    if mo_coeff_i is None or mo_coeff_j is None:
        raise ValueError("mo_coeff_i and mo_coeff_j are required")

    log = logger if logger is not None else g_logger.new_logger(cell, cell.verbose)
    debug_sync = _env_flag("BYTEQC_ERI_TRANS_PBC_DEBUG_SYNC")
    debug_imag_check = _debug_imag_check_enabled()
    if debug_imag_check:
        log.info(
            "eri_high_level_solver_incore self-conjugate aggregate imag check is enabled "
            "(BYTEQC_ERI_TRANS_PBC_DEBUG_IMAG_CHECK=1)"
        )

    if hasattr(kpts, "kpts"):
        kpts = kpts.kpts
    kpts = _to_numpy_array(kpts, dtype=float)
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise ValueError(f"Invalid k-point shape: {kpts.shape}")
    nkpts = len(kpts)
    if nkpts <= 1:
        raise NotImplementedError("Gamma-only calculations should use eri_trans_gpu4pyscf.eri_high_level_solver_incore")

    if kmesh is None:
        kmesh = _to_numpy_array(kpts_to_kmesh(cell, kpts), dtype=int)
    else:
        kmesh = _to_numpy_array(kmesh, dtype=int)
    if int(np.prod(kmesh)) != nkpts:
        raise ValueError(f"kmesh {tuple(kmesh)} is incompatible with {nkpts} k-points")

    coeff_i_k = _gamma_ao_to_k_coeff(cell, kpts, mo_coeff_i, kmesh, wrap_around=wrap_around)
    coeff_j_k = _gamma_ao_to_k_coeff(cell, kpts, mo_coeff_j, kmesh, wrap_around=wrap_around)
    nmo_i = int(coeff_i_k.shape[2])
    nmo_j = int(coeff_j_k.shape[2])

    if omega is None:
        omega = OMEGA_MIN
    omega = abs(float(omega))

    int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega=-omega, bvk_kmesh=kmesh).build()
    ao_cell = getattr(int3c2e_opt.cell, "mol", cell)
    ao_nao = int(ao_cell.nao)
    int3c_nao = int(int3c2e_opt.cell.nao_nr(cart=True))
    if coeff_i_k.shape[1] != ao_nao:
        raise ValueError(
            "Primitive-cell AO dimension mismatch: coefficient AO dimension "
            f"{coeff_i_k.shape[1]} vs AO-space coefficient dimension {ao_nao}"
        )
    if coeff_j_k.shape[1] != ao_nao:
        raise ValueError(
            "Primitive-cell AO dimension mismatch: coefficient AO dimension "
            f"{coeff_j_k.shape[1]} vs AO-space coefficient dimension {ao_nao}"
        )
    nao_unpack = int3c_nao
    if int3c_nao != ao_nao:
        log.info(
            "eri_high_level_solver_incore internal AO dimension differs from external AO dimension: "
            "int3c_nao=%d ao_nao=%d; AO coefficients will be transformed to the "
            "sorted internal AO basis before AO2MO",
            int3c_nao,
            ao_nao,
        )

    coeff_i_k = np.asarray(
        cp.asnumpy(int3c2e_opt.cell.apply_C_dot(cp.asarray(coeff_i_k), axis=1)),
        dtype=np.complex128,
        order="C",
    )
    coeff_j_k = np.asarray(
        cp.asnumpy(int3c2e_opt.cell.apply_C_dot(cp.asarray(coeff_j_k), axis=1)),
        dtype=np.complex128,
        order="C",
    )

    cd_j2c_cache, negative_metric_size = _precontract_j2c_aux_coeff(
        int3c2e_opt.auxcell,
        kpts,
        omega,
        with_long_range,
        linear_dep_threshold,
        kmesh,
    )
    if any(int(x) > 0 for x in negative_metric_size.values()):
        raise NotImplementedError("Negative-metric auxiliary sectors are not supported in this first multi-k high-level-solver path")

    kpt_iters = list(kk_adapted_iter(kmesh))
    stored_kpts = kpts[[x[0] for x in kpt_iters]]
    nstored_kpts = len(kpt_iters)
    if nstored_kpts <= 0:
        raise RuntimeError("Failed to identify independent q sectors")

    weights = _get_t_reversal_weights(cell, kpts)
    aux_coeffs = [cp.asarray(coeff, dtype=np.complex128, order="C") for coeff in cd_j2c_cache]
    naux_cart = int(cd_j2c_cache[0].shape[0])

    total_naux_real = 0
    active_naux_q = []
    active_kblk_sizes = []
    for idx, (k_aux, *_rest) in enumerate(kpt_iters):
        weight = int(weights[k_aux])
        if weight > 0:
            naux_q_i = int(aux_coeffs[idx].shape[1])
            total_naux_real += weight * naux_q_i
            active_naux_q.append(naux_q_i)
            if Lblksize is None:
                kblk_i = naux_q_i
            else:
                kblk_i = max(1, min(int(Lblksize), naux_q_i))
            active_kblk_sizes.append(kblk_i)
    if total_naux_real <= 0:
        raise RuntimeError("No real auxiliary channels survived the time-reversal reduction")

    nsp_per_block = ft_ao.ft_ao_scheme()[0]
    bas_ij_aggregated = int3c2e_opt.cell.aggregate_shl_pairs(int3c2e_opt.bas_ij_cache, nsp_per_block)

    cderi_idx = int3c2e_opt.pair_and_diag_indices(cart=True, original_ao_order=False)
    pair_address = _to_numpy_array(cderi_idx[0], dtype=np.int32)
    eval_j3c, aux_sorting, ao_pair_offsets = int3c2e_opt.int3c2e_evaluator(
        ao_pair_batch_size=ao_pair_batch_size,
        cart=True,
        bas_ij_aggregated=bas_ij_aggregated,
    )[:3]
    ao_pair_offsets = _to_numpy_array(ao_pair_offsets, dtype=np.int64)
    shl_pair_batches = int(len(ao_pair_offsets) - 1)
    pair_sizes = np.diff(ao_pair_offsets).astype(np.int64, copy=False)
    max_pair_size = int(pair_sizes.max()) if pair_sizes.size else 0
    npair_total = int(ao_pair_offsets[-1]) if len(ao_pair_offsets) else 0

    for idx, coeff in enumerate(aux_coeffs):
        coeff_sorted = cp.empty_like(coeff)
        coeff_sorted[aux_sorting] = coeff
        aux_coeffs[idx] = coeff_sorted

    expLk_full = cp.exp(1j * cp.asarray(int3c2e_opt.bvkmesh_Ls.dot(kpts.T)))
    expLk_conjz_full = expLk_full.conj().view(np.float64).reshape(int(np.prod(kmesh)), nkpts, 2)
    conj_mapping = _to_numpy_array(conj_images_in_bvk_cell(kmesh), dtype=np.int32)
    nL_full = int(expLk_full.shape[0])
    pair_upper_bound = nao_unpack * nL_full * nao_unpack
    if pair_address.size:
        pair_max = int(pair_address.max())
        pair_min = int(pair_address.min())
        if pair_min < 0 or pair_max >= pair_upper_bound:
            raise RuntimeError(
                "pair_address is incompatible with unpack dimensions: "
                f"min={pair_min}, max={pair_max}, upper_bound={pair_upper_bound}, "
                f"nao_unpack={nao_unpack}, nL={nL_full}, nkpts={nkpts}"
            )
    if expLk_full.shape != (int(np.prod(kmesh)), nkpts):
        raise RuntimeError(
            f"Unexpected expLk shape {expLk_full.shape}; expected {(int(np.prod(kmesh)), nkpts)}"
        )
    if conj_mapping.shape != (nkpts,):
        raise RuntimeError(
            f"Unexpected conjugation mapping shape {conj_mapping.shape}; expected {(nkpts,)}"
        )
    pair_address_gpu = cp.asarray(pair_address, dtype=np.int32)
    conj_mapping_gpu = cp.asarray(conj_mapping, dtype=np.int32)

    if with_long_range:
        mesh = int3c2e_opt.mesh
        Gv, _Gvbase, kws = cell.get_Gv_weights(mesh)
        ngrids = len(Gv)
        Gk = (Gv + stored_kpts[:, None]).reshape(-1, 3)
        Gk = _Gv_wrap_around(cell, Gk, cp.zeros(3), mesh)
        coulG = get_coulG(cell, Gv=Gk, omega=omega).reshape(nstored_kpts, ngrids)
        coulG *= kws
        coulG[0, 0] -= np.pi / omega**2 / cell.vol

        ft_opt = ft_ao.FTOpt.from_intopt(int3c2e_opt)
        eval_ft, ao_pair_offsets_ft = ft_opt.ft_evaluator(
            ao_pair_batch_size,
            cart=True,
            original_ao_order=False,
            bas_ij_aggregated=bas_ij_aggregated,
        )
        ao_pair_offsets_ft = _to_numpy_array(ao_pair_offsets_ft, dtype=np.int64)
        if not np.array_equal(ao_pair_offsets, ao_pair_offsets_ft):
            raise RuntimeError("AO-pair offsets mismatch between SR and LR evaluators")

        auxG = ft_ao.ft_ao(int3c2e_opt.auxcell, Gk).T
        auxG = auxG.reshape(naux_cart, nstored_kpts, ngrids)
        for k in range(nstored_kpts):
            auxG[aux_sorting, k] = auxG[:, k].conj()
        auxG_conj = auxG
        auxG_conj *= cp.asarray(coulG)

        if Gblksize is None:
            mem_free = cp.cuda.runtime.memGetInfo()[0]
            mem_free -= sum(x.nbytes for x in aux_coeffs)
            mem_free -= nkpts * naux_cart * ao_pair_batch_size * 16 * 2
            Gblksize = int(mem_free // max(16 * ao_pair_batch_size, 1)) // 32 * 32
            Gblksize = max(32, min(Gblksize, ngrids))
        g_slices = list(prange(0, ngrids, Gblksize))
    else:
        eval_ft = None
        Gv = None
        ngrids = 0
        auxG_conj = None
        g_slices = []

    coeff_i_gpu = [cp.asarray(coeff_i_k[k], dtype=np.complex128, order="C") for k in range(nkpts)]
    coeff_j_gpu = [cp.asarray(coeff_j_k[k], dtype=np.complex128, order="C") for k in range(nkpts)]
    pair_norm = 1.0 / np.sqrt(nkpts)
    sqrt2 = np.sqrt(2.0)
    max_naux_q = max(active_naux_q) if active_naux_q else 0
    max_kblk = max(active_kblk_sizes) if active_kblk_sizes else 0
    max_real_kblk = 2 * max_kblk
    npair_total = int(ao_pair_offsets[-1]) if len(ao_pair_offsets) else 0

    if max_pair_size > 0 and max_kblk > 0 and max_naux_q > 0:
        cderi_q_full_host_buf = cupyx.empty_pinned((max_naux_q, npair_total), dtype=np.complex128, order="C")
        cderi_q_batch_host_buf = cupyx.empty_pinned((max_naux_q, max_pair_size), dtype=np.complex128, order="C")
        pq_host_buf = cupyx.empty_pinned((nmo_i, nmo_j, max_kblk), dtype=np.complex128, order="C")
        pq_real_buf = cupyx.empty_pinned((nmo_i, nmo_j, max_real_kblk), dtype=np.float64, order="C")
        pq_debug_sum_buf = (
            cupyx.empty_pinned((nmo_i, nmo_j, max_kblk), dtype=np.complex128, order="C")
            if debug_imag_check else None
        )
    else:
        cderi_q_full_host_buf = None
        cderi_q_batch_host_buf = None
        pq_host_buf = None
        pq_real_buf = None
        pq_debug_sum_buf = None

    nov_tot = nmo_i * nmo_j
    svd_compress_aux = int(DEFAULT_SVD_COMPRESS_AUX)
    svd_buffer_aux = max(1, min(int(total_naux_real), svd_compress_aux))
    svd_buffers = allocate_incremental_aux_svd_buffers(
        nov_tot,
        max_aux_eigh=svd_buffer_aux,
        max_aux_out=svd_buffer_aux,
    )

    canonical_cderi_cpu = cupyx.empty_pinned((nov_tot, total_naux_real), dtype=np.float64, order="C")
    aux_write_pos = 0
    for j2c_idx, (k_aux, _k_aux_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
        weight = int(weights[k_aux])
        if weight <= 0:
            continue

        k_aux = int(k_aux)
        ki_idx = _to_numpy_array(ki_idx, dtype=np.int32)
        kj_idx = _to_numpy_array(kj_idx, dtype=np.int32)
        if ki_idx.shape != (nkpts,) or kj_idx.shape != (nkpts,):
            raise RuntimeError(
                f"k-point index permutation for q={k_aux} has wrong shape: "
                f"ki_idx.shape={ki_idx.shape}, kj_idx.shape={kj_idx.shape}, expected {(nkpts,)}"
            )
        if not np.array_equal(np.sort(ki_idx), np.arange(nkpts, dtype=np.int32)):
            raise RuntimeError(f"ki_idx for q={k_aux} is not a permutation of [0, {nkpts})")
        if not np.array_equal(np.sort(kj_idx), np.arange(nkpts, dtype=np.int32)):
            raise RuntimeError(f"kj_idx for q={k_aux} is not a permutation of [0, {nkpts})")
        kj_idx_gpu = cp.asarray(kj_idx, dtype=np.int32)
        naux_q = int(aux_coeffs[j2c_idx].shape[1])
        Kblksize = naux_q if Lblksize is None else max(1, min(int(Lblksize), naux_q))

        log.info(
            "eri_high_level_solver_incore multi-k q=%d/%d k_aux=%d weight=%d naux=%d",
            j2c_idx + 1,
            len(kpt_iters),
            k_aux,
            weight,
            naux_q,
        )
        pq_imag_max = 0.0
        pq_imag_real_max = 0.0
        pq_imag_info = None

        q_cderi_host = _build_full_q_cderi_host(
            log=log,
            label="eri_high_level_solver_incore",
            j2c_idx=j2c_idx,
            k_aux=k_aux,
            naux_q=naux_q,
            shl_pair_batches=shl_pair_batches,
            ao_pair_offsets=ao_pair_offsets,
            naux_cart=naux_cart,
            eval_j3c=eval_j3c,
            expLk_conjz_full=expLk_conjz_full,
            aux_coeff_q=aux_coeffs[j2c_idx],
            with_long_range=with_long_range,
            g_slices=g_slices,
            auxG_conj=auxG_conj,
            eval_ft=eval_ft,
            Gv=Gv,
            kpts=kpts,
            cderi_q_full_host_buf=cderi_q_full_host_buf,
            cderi_q_batch_host_buf=cderi_q_batch_host_buf,
            debug_sync=debug_sync,
        )
        _free_current_eri_pool_blocks()
        # Keep the persisted aux-column order independent of KBLKSIZE.
        q_aux_base = aux_write_pos
        for K0 in range(0, naux_q, Kblksize):
            K1 = min(naux_q, K0 + Kblksize)
            block_len = K1 - K0
            real_width = block_len if weight == 1 else 2 * block_len
            pq_real = lib.empty_from_buf(pq_real_buf, (nmo_i, nmo_j, real_width), np.float64)
            pq_real[:] = 0.0
            if weight == 1 and debug_imag_check:
                pq_debug_sum = lib.empty_from_buf(
                    pq_debug_sum_buf, (nmo_i, nmo_j, block_len), np.complex128
                )
                pq_debug_sum[:] = 0.0
            else:
                pq_debug_sum = None

            log.info(
                "eri_high_level_solver_incore q=%d aux-block [%d:%d) width=%d",
                k_aux,
                K0,
                K1,
                real_width,
            )

            cderi_q = lib.empty_from_buf(None, (block_len, npair_total), np.complex128)
            cderi_q.set(q_cderi_host[K0:K1])
            if debug_sync:
                cp.cuda.get_current_stream().synchronize()

            try:
                unpacked = rsdf_builder._unpack_cderi_v2(
                    cderi_q,
                    pair_address_gpu,
                    kj_idx_gpu,
                    conj_mapping_gpu,
                    expLk_full,
                    nao_unpack,
                    axis=0,
                )
                if debug_sync:
                    cp.cuda.get_current_stream().synchronize()
            except Exception as err:
                raise RuntimeError(
                    "Failed in _unpack_cderi_v2 with "
                    f"q={k_aux}, K-range=[{K0}:{K1}), "
                    f"nao_unpack={nao_unpack}, ao_nao={ao_nao}, nkpts={nkpts}, nL={nL_full}, "
                    f"npair_total={npair_total}, len(kj_idx)={len(kj_idx)}, "
                    f"expLk_shape={tuple(expLk_full.shape)}"
                ) from err

            for ki, kj in zip(ki_idx, kj_idx):
                ao_pair = unpacked[ki]
                tmp = lib.empty_from_buf(None, (nmo_i, block_len, nao_unpack), np.complex128)
                lib.contraction(
                    "Lpq",
                    ao_pair,
                    "pi",
                    coeff_i_gpu[int(ki)],
                    "iLq",
                    tmp,
                    opb="CONJ",
                )
                pq_blk = lib.empty_from_buf(None, (nmo_i, nmo_j, block_len), np.complex128)
                lib.contraction(
                    "iLq",
                    tmp,
                    "qj",
                    coeff_j_gpu[int(kj)],
                    "ijL",
                    pq_blk,
                    alpha=pair_norm,
                )
                pq_host = lib.empty_from_buf(pq_host_buf, (nmo_i, nmo_j, block_len), np.complex128)
                pq_blk.get(out=pq_host, blocking=True)
                tmp = None
                pq_blk = None
                if weight == 1 and debug_imag_check:
                    pq_debug_sum += pq_host
                _accumulate_real_aux_channels(pq_real, pq_host, weight=weight, sqrt2=sqrt2)
                pq_host = None
                ao_pair = None

            unpacked = None
            cderi_q = None
            if weight == 1 and debug_imag_check:
                info = "K=[%d:%d)" % (K0, K1)
                pq_imag_max, pq_imag_real_max, pq_imag_info = _update_self_conj_imag_stats(
                    pq_debug_sum, pq_imag_max, pq_imag_real_max, pq_imag_info, info
                )
                pq_debug_sum = None

            pq_real_view = pq_real.reshape(nov_tot, real_width)
            if weight == 1:
                canonical_cderi_cpu[:, q_aux_base + K0:q_aux_base + K1] = pq_real_view
            elif weight == 2:
                real_dst = slice(q_aux_base + K0, q_aux_base + K1)
                imag_dst = slice(q_aux_base + naux_q + K0, q_aux_base + naux_q + K1)
                canonical_cderi_cpu[:, real_dst] = pq_real_view[:, :block_len]
                canonical_cderi_cpu[:, imag_dst] = pq_real_view[:, block_len:real_width]
            else:
                raise ValueError(f"Unsupported time-reversal weight {weight}")
            pq_real_view = None

        q_cderi_host = None
        aux_write_pos += weight * naux_q

        if weight == 1 and debug_imag_check:
            _log_self_conj_imag_result(
                log, "cderi", k_aux, pq_imag_max, pq_imag_real_max, imag_tol, pq_imag_info
            )

        _free_current_eri_pool_blocks()

    if aux_write_pos != total_naux_real:
        raise RuntimeError(
            f"Filled auxiliary width {aux_write_pos}, expected {total_naux_real}"
        )
    compressed_cderi_chunks = compress_canonical_aux_matrix_in_aux_chunks(
        canonical_cderi_cpu,
        svd_tol=svd_tol,
        buffers=svd_buffers,
        logger=log,
        label="eri_high_level_solver_incore",
        compress_aux=svd_compress_aux,
        return_chunks=True,
    )
    canonical_cderi_cpu = None
    gc.collect()
    compressed_cderi_cpu = concatenate_canonical_aux_chunks(
        compressed_cderi_chunks,
        nov_tot=nov_tot,
    )
    compressed_cderi_chunks = None
    gc.collect()
    cderi_cut = format_canonical_aux_for_solver(compressed_cderi_cpu, solver_type=solver_type)

    del coeff_i_gpu, coeff_j_gpu, aux_coeffs, cd_j2c_cache, eval_j3c, ao_pair_offsets
    del expLk_full, expLk_conjz_full, bas_ij_aggregated, kpt_iters, stored_kpts, pair_address
    del cderi_q_full_host_buf, cderi_q_batch_host_buf
    del pq_host_buf, pq_real_buf, pq_debug_sum_buf
    if with_long_range:
        del Gv, auxG_conj, eval_ft
    del compressed_cderi_cpu, svd_buffers
    _free_current_eri_pool_blocks()

    return cderi_cut


@_with_eri_memory_pool
def eri_high_level_solver_incore_with_jk(
    cell,
    auxcell,
    mo_coeff,
    *args,
    svd_tol=1e-4,
    omega=None,
    linear_dep_threshold=LINEAR_DEP_THR,
    with_long_range=True,
    ao_pair_batch_size=AO_PAIR_BATCH_SIZE_2,
    Lblksize=KBLKSIZE_2,
    Gblksize=GBLKSIZE_2,
    kpts=None,
    kmesh=None,
    wrap_around=False,
    imag_tol=IMAG_TOL,
):
    """Build multi-k gamma-supercell cderi_cut together with JK contributions.

    This is the PBC analogue of
    ``byteqc.embyte.ERI.eri_trans_gpu4pyscf.eri_high_level_solver_incore_with_jk``.
    The positional interface and return semantics are kept aligned with the
    gamma-point implementation. The extra keyword ``kpts`` supplies the
    primitive-cell Monkhorst-Pack mesh used to build the ERI on the fly.

    The returned ``cderi_cut`` follows the gamma implementation and is stored
    in the auxiliary-first layout ``(naux_cut, nmo * nmo)``. The returned
    ``vj`` and ``vk`` are accumulated from the untruncated auxiliary space.
    """
    if len(args) == 2:
        j2c = None
        logger, rdm1_core_coeff = args
    elif len(args) == 3:
        j2c, logger, rdm1_core_coeff = args
    else:
        raise TypeError(
            'eri_high_level_solver_incore_with_jk expects either '
            '(..., logger, rdm1_core_coeff) or (..., j2c, logger, rdm1_core_coeff)'
        )
    _ = j2c

    _free_current_eri_pool_blocks()

    if cell.dimension != 3:
        raise NotImplementedError(
            "The current multi-k eri_high_level_solver_incore_with_jk path only supports 3D cells"
        )
    if auxcell is None:
        raise ValueError("auxcell is required")
    if kpts is None:
        raise ValueError("kpts must be provided for the multi-k eri_high_level_solver_incore_with_jk path")
    if mo_coeff is None:
        raise ValueError("mo_coeff is required")
    if rdm1_core_coeff is None:
        raise ValueError("rdm1_core_coeff is required")

    log = logger if logger is not None else g_logger.new_logger(cell, cell.verbose)
    debug_sync = _env_flag("BYTEQC_ERI_TRANS_PBC_DEBUG_SYNC")
    debug_imag_check = _debug_imag_check_enabled()
    if debug_imag_check:
        log.info(
            "eri_high_level_solver_incore_with_jk self-conjugate aggregate imag check is enabled "
            "(BYTEQC_ERI_TRANS_PBC_DEBUG_IMAG_CHECK=1)"
        )

    if hasattr(kpts, "kpts"):
        kpts = kpts.kpts
    kpts = _to_numpy_array(kpts, dtype=float)
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise ValueError(f"Invalid k-point shape: {kpts.shape}")
    nkpts = len(kpts)
    if nkpts <= 1:
        raise NotImplementedError(
            "Gamma-only calculations should use eri_trans_gpu4pyscf.eri_high_level_solver_incore_with_jk"
        )

    if kmesh is None:
        kmesh = _to_numpy_array(kpts_to_kmesh(cell, kpts), dtype=int)
    else:
        kmesh = _to_numpy_array(kmesh, dtype=int)
    if int(np.prod(kmesh)) != nkpts:
        raise ValueError(f"kmesh {tuple(kmesh)} is incompatible with {nkpts} k-points")

    mo_coeff = _to_numpy_array(mo_coeff)
    rdm1_core_coeff = _to_numpy_array(rdm1_core_coeff)
    if mo_coeff.ndim != 2:
        raise ValueError(f"mo_coeff must be a 2D array; got {mo_coeff.shape}")
    if rdm1_core_coeff.ndim != 2:
        raise ValueError(
            f"rdm1_core_coeff must be a 2D array with shape (nao_sc, ncore); got {rdm1_core_coeff.shape}"
        )
    if mo_coeff.shape[0] != rdm1_core_coeff.shape[0]:
        raise ValueError("rdm1_core_coeff must have the same AO dimension as mo_coeff")

    coeff_mo_k = _gamma_ao_to_k_coeff(cell, kpts, mo_coeff, kmesh, wrap_around=wrap_around)
    coeff_core_k = _gamma_ao_to_k_coeff(cell, kpts, rdm1_core_coeff, kmesh, wrap_around=wrap_around)
    nmo = int(coeff_mo_k.shape[2])
    ncore = int(coeff_core_k.shape[2])

    if omega is None:
        omega = OMEGA_MIN
    omega = abs(float(omega))

    int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega=-omega, bvk_kmesh=kmesh).build()
    ao_cell = getattr(int3c2e_opt.cell, "mol", cell)
    ao_nao = int(ao_cell.nao)
    int3c_nao = int(int3c2e_opt.cell.nao_nr(cart=True))
    if coeff_mo_k.shape[1] != ao_nao:
        raise ValueError(
            "Primitive-cell AO dimension mismatch: coefficient AO dimension "
            f"{coeff_mo_k.shape[1]} vs AO-space coefficient dimension {ao_nao}"
        )
    if coeff_core_k.shape[1] != ao_nao:
        raise ValueError(
            "Primitive-cell AO dimension mismatch for rdm1_core_coeff: coefficient AO dimension "
            f"{coeff_core_k.shape[1]} vs AO-space coefficient dimension {ao_nao}"
        )
    nao_unpack = int3c_nao
    if int3c_nao != ao_nao:
        log.info(
            "eri_high_level_solver_incore_with_jk internal AO dimension differs from external AO dimension: "
            "int3c_nao=%d ao_nao=%d; AO coefficients will be transformed to the "
            "sorted internal AO basis before AO2MO",
            int3c_nao,
            ao_nao,
        )

    coeff_mo_k = np.asarray(
        cp.asnumpy(int3c2e_opt.cell.apply_C_dot(cp.asarray(coeff_mo_k), axis=1)),
        dtype=np.complex128,
        order="C",
    )
    coeff_core_k = np.asarray(
        cp.asnumpy(int3c2e_opt.cell.apply_C_dot(cp.asarray(coeff_core_k), axis=1)),
        dtype=np.complex128,
        order="C",
    )

    cd_j2c_cache, negative_metric_size = _precontract_j2c_aux_coeff(
        int3c2e_opt.auxcell,
        kpts,
        omega,
        with_long_range,
        linear_dep_threshold,
        kmesh,
    )
    if any(int(x) > 0 for x in negative_metric_size.values()):
        raise NotImplementedError(
            "Negative-metric auxiliary sectors are not supported in this first multi-k with_jk path"
        )

    kpt_iters = list(kk_adapted_iter(kmesh))
    stored_kpts = kpts[[x[0] for x in kpt_iters]]
    nstored_kpts = len(kpt_iters)
    if nstored_kpts <= 0:
        raise RuntimeError("Failed to identify independent q sectors")

    weights = _get_t_reversal_weights(cell, kpts)
    aux_coeffs = [cp.asarray(coeff, dtype=np.complex128, order="C") for coeff in cd_j2c_cache]
    naux_cart = int(cd_j2c_cache[0].shape[0])

    total_naux_real = 0
    active_naux_q = []
    active_kblk_sizes = []
    for idx, (k_aux, *_rest) in enumerate(kpt_iters):
        weight = int(weights[k_aux])
        if weight > 0:
            naux_q_i = int(aux_coeffs[idx].shape[1])
            total_naux_real += weight * naux_q_i
            active_naux_q.append(naux_q_i)
            if Lblksize is None:
                kblk_i = naux_q_i
            else:
                kblk_i = max(1, min(int(Lblksize), naux_q_i))
            active_kblk_sizes.append(kblk_i)
    if total_naux_real <= 0:
        raise RuntimeError("No real auxiliary channels survived the time-reversal reduction")

    nsp_per_block = ft_ao.ft_ao_scheme()[0]
    bas_ij_aggregated = int3c2e_opt.cell.aggregate_shl_pairs(int3c2e_opt.bas_ij_cache, nsp_per_block)

    cderi_idx = int3c2e_opt.pair_and_diag_indices(cart=True, original_ao_order=False)
    pair_address = _to_numpy_array(cderi_idx[0], dtype=np.int32)
    eval_j3c, aux_sorting, ao_pair_offsets = int3c2e_opt.int3c2e_evaluator(
        ao_pair_batch_size=ao_pair_batch_size,
        cart=True,
        bas_ij_aggregated=bas_ij_aggregated,
    )[:3]
    ao_pair_offsets = _to_numpy_array(ao_pair_offsets, dtype=np.int64)
    shl_pair_batches = int(len(ao_pair_offsets) - 1)
    pair_sizes = np.diff(ao_pair_offsets).astype(np.int64, copy=False)
    max_pair_size = int(pair_sizes.max()) if pair_sizes.size else 0
    npair_total = int(ao_pair_offsets[-1]) if len(ao_pair_offsets) else 0

    for idx, coeff in enumerate(aux_coeffs):
        coeff_sorted = cp.empty_like(coeff)
        coeff_sorted[aux_sorting] = coeff
        aux_coeffs[idx] = coeff_sorted

    expLk_full = cp.exp(1j * cp.asarray(int3c2e_opt.bvkmesh_Ls.dot(kpts.T)))
    expLk_conjz_full = expLk_full.conj().view(np.float64).reshape(int(np.prod(kmesh)), nkpts, 2)
    conj_mapping = _to_numpy_array(conj_images_in_bvk_cell(kmesh), dtype=np.int32)
    nL_full = int(expLk_full.shape[0])
    pair_upper_bound = nao_unpack * nL_full * nao_unpack
    if pair_address.size:
        pair_max = int(pair_address.max())
        pair_min = int(pair_address.min())
        if pair_min < 0 or pair_max >= pair_upper_bound:
            raise RuntimeError(
                "pair_address is incompatible with unpack dimensions: "
                f"min={pair_min}, max={pair_max}, upper_bound={pair_upper_bound}, "
                f"nao_unpack={nao_unpack}, nL={nL_full}, nkpts={nkpts}"
            )
    if expLk_full.shape != (int(np.prod(kmesh)), nkpts):
        raise RuntimeError(
            f"Unexpected expLk shape {expLk_full.shape}; expected {(int(np.prod(kmesh)), nkpts)}"
        )
    if conj_mapping.shape != (nkpts,):
        raise RuntimeError(
            f"Unexpected conjugation mapping shape {conj_mapping.shape}; expected {(nkpts,)}"
        )
    pair_address_gpu = cp.asarray(pair_address, dtype=np.int32)
    conj_mapping_gpu = cp.asarray(conj_mapping, dtype=np.int32)

    if with_long_range:
        mesh = int3c2e_opt.mesh
        Gv, _Gvbase, kws = cell.get_Gv_weights(mesh)
        ngrids = len(Gv)
        Gk = (Gv + stored_kpts[:, None]).reshape(-1, 3)
        Gk = _Gv_wrap_around(cell, Gk, cp.zeros(3), mesh)
        coulG = get_coulG(cell, Gv=Gk, omega=omega).reshape(nstored_kpts, ngrids)
        coulG *= kws
        coulG[0, 0] -= np.pi / omega**2 / cell.vol

        ft_opt = ft_ao.FTOpt.from_intopt(int3c2e_opt)
        eval_ft, ao_pair_offsets_ft = ft_opt.ft_evaluator(
            ao_pair_batch_size,
            cart=True,
            original_ao_order=False,
            bas_ij_aggregated=bas_ij_aggregated,
        )
        ao_pair_offsets_ft = _to_numpy_array(ao_pair_offsets_ft, dtype=np.int64)
        if not np.array_equal(ao_pair_offsets, ao_pair_offsets_ft):
            raise RuntimeError("AO-pair offsets mismatch between SR and LR evaluators")

        auxG = ft_ao.ft_ao(int3c2e_opt.auxcell, Gk).T
        auxG = auxG.reshape(naux_cart, nstored_kpts, ngrids)
        for k in range(nstored_kpts):
            auxG[aux_sorting, k] = auxG[:, k].conj()
        auxG_conj = auxG
        auxG_conj *= cp.asarray(coulG)

        if Gblksize is None:
            mem_free = cp.cuda.runtime.memGetInfo()[0]
            mem_free -= sum(x.nbytes for x in aux_coeffs)
            mem_free -= nkpts * naux_cart * ao_pair_batch_size * 16 * 2
            Gblksize = int(mem_free // max(16 * ao_pair_batch_size, 1)) // 32 * 32
            Gblksize = max(32, min(Gblksize, ngrids))
        g_slices = list(prange(0, ngrids, Gblksize))
    else:
        eval_ft = None
        Gv = None
        ngrids = 0
        auxG_conj = None
        g_slices = []

    coeff_mo_gpu = [cp.asarray(coeff_mo_k[k], dtype=np.complex128, order="C") for k in range(nkpts)]
    coeff_core_gpu = [cp.asarray(coeff_core_k[k], dtype=np.complex128, order="C") for k in range(nkpts)]
    if ncore > 0:
        coeff_left_gpu = [
            cp.concatenate((coeff_mo_gpu[k], coeff_core_gpu[k]), axis=1)
            for k in range(nkpts)
        ]
    else:
        coeff_left_gpu = coeff_mo_gpu

    pair_norm = 1.0 / np.sqrt(nkpts)
    sqrt2 = np.sqrt(2.0)
    nov_tot = nmo * nmo
    max_naux_q = max(active_naux_q) if active_naux_q else 0
    max_kblk = max(active_kblk_sizes) if active_kblk_sizes else 0
    max_real_kblk = 2 * max_kblk

    if max_pair_size > 0 and max_kblk > 0 and max_naux_q > 0:
        cderi_q_full_host_buf = cupyx.empty_pinned((max_naux_q, npair_total), dtype=np.complex128, order="C")
        cderi_q_batch_host_buf = cupyx.empty_pinned((max_naux_q, max_pair_size), dtype=np.complex128, order="C")
        pq_host_buf = cupyx.empty_pinned((nmo, nmo, max_kblk), dtype=np.complex128, order="C")
        pq_real_buf = cupyx.empty_pinned((nmo, nmo, max_real_kblk), dtype=np.float64, order="C")
        pq_debug_sum_buf = (
            cupyx.empty_pinned((nmo, nmo, max_kblk), dtype=np.complex128, order="C")
            if debug_imag_check else None
        )
    else:
        cderi_q_full_host_buf = None
        cderi_q_batch_host_buf = None
        pq_host_buf = None
        pq_real_buf = None
        pq_debug_sum_buf = None

    svd_compress_aux = int(DEFAULT_SVD_COMPRESS_AUX)
    svd_buffer_aux = max(1, min(int(total_naux_real), svd_compress_aux))
    svd_buffers = allocate_incremental_aux_svd_buffers(
        nov_tot,
        max_aux_eigh=svd_buffer_aux,
        max_aux_out=svd_buffer_aux,
    )
    canonical_cderi_cpu = cupyx.empty_pinned((nov_tot, total_naux_real), dtype=np.float64, order="C")
    aux_write_pos = 0
    vj_gpu = cp.zeros((nov_tot, 1), dtype=np.float64)
    vk_gpu = cp.zeros((nmo, nmo), dtype=np.float64)

    for j2c_idx, (k_aux, _k_aux_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
        weight = int(weights[k_aux])
        if weight <= 0:
            continue

        k_aux = int(k_aux)
        ki_idx = _to_numpy_array(ki_idx, dtype=np.int32)
        kj_idx = _to_numpy_array(kj_idx, dtype=np.int32)
        if ki_idx.shape != (nkpts,) or kj_idx.shape != (nkpts,):
            raise RuntimeError(
                f"k-point index permutation for q={k_aux} has wrong shape: "
                f"ki_idx.shape={ki_idx.shape}, kj_idx.shape={kj_idx.shape}, expected {(nkpts,)}"
            )
        if not np.array_equal(np.sort(ki_idx), np.arange(nkpts, dtype=np.int32)):
            raise RuntimeError(f"ki_idx for q={k_aux} is not a permutation of [0, {nkpts})")
        if not np.array_equal(np.sort(kj_idx), np.arange(nkpts, dtype=np.int32)):
            raise RuntimeError(f"kj_idx for q={k_aux} is not a permutation of [0, {nkpts})")
        kj_idx_gpu = cp.asarray(kj_idx, dtype=np.int32)
        naux_q = int(aux_coeffs[j2c_idx].shape[1])
        Kblksize = naux_q if Lblksize is None else max(1, min(int(Lblksize), naux_q))

        log.info(
            "eri_high_level_solver_incore_with_jk multi-k q=%d/%d k_aux=%d weight=%d naux=%d",
            j2c_idx + 1,
            len(kpt_iters),
            k_aux,
            weight,
            naux_q,
        )
        pq_imag_max = 0.0
        pq_imag_real_max = 0.0
        pq_imag_info = None

        q_cderi_host = _build_full_q_cderi_host(
            log=log,
            label="eri_high_level_solver_incore_with_jk",
            j2c_idx=j2c_idx,
            k_aux=k_aux,
            naux_q=naux_q,
            shl_pair_batches=shl_pair_batches,
            ao_pair_offsets=ao_pair_offsets,
            naux_cart=naux_cart,
            eval_j3c=eval_j3c,
            expLk_conjz_full=expLk_conjz_full,
            aux_coeff_q=aux_coeffs[j2c_idx],
            with_long_range=with_long_range,
            g_slices=g_slices,
            auxG_conj=auxG_conj,
            eval_ft=eval_ft,
            Gv=Gv,
            kpts=kpts,
            cderi_q_full_host_buf=cderi_q_full_host_buf,
            cderi_q_batch_host_buf=cderi_q_batch_host_buf,
            debug_sync=debug_sync,
        )
        _free_current_eri_pool_blocks()
        # Keep the persisted aux-column order independent of KBLKSIZE.
        q_aux_base = aux_write_pos
        for K0 in range(0, naux_q, Kblksize):
            K1 = min(naux_q, K0 + Kblksize)
            block_len = K1 - K0
            real_width = block_len if weight == 1 else 2 * block_len

            pq_real = lib.empty_from_buf(pq_real_buf, (nmo, nmo, real_width), np.float64)
            pq_real[:] = 0.0
            if weight == 1 and debug_imag_check:
                pq_debug_sum = lib.empty_from_buf(
                    pq_debug_sum_buf, (nmo, nmo, block_len), np.complex128
                )
                pq_debug_sum[:] = 0.0
            else:
                pq_debug_sum = None
            if ncore > 0:
                cm_sum = lib.empty_from_buf(None, (ncore, nmo, block_len), np.complex128)
                cm_sum[:] = 0.0
                cc_sum = lib.empty_from_buf(None, (block_len,), np.complex128)
                cc_sum[:] = 0.0
            else:
                cm_sum = None
                cc_sum = None

            log.info(
                "eri_high_level_solver_incore_with_jk q=%d aux-block [%d:%d) width=%d",
                k_aux,
                K0,
                K1,
                real_width,
            )

            cderi_q = lib.empty_from_buf(None, (block_len, npair_total), np.complex128)
            cderi_q.set(q_cderi_host[K0:K1])
            if debug_sync:
                cp.cuda.get_current_stream().synchronize()

            try:
                unpacked = rsdf_builder._unpack_cderi_v2(
                    cderi_q,
                    pair_address_gpu,
                    kj_idx_gpu,
                    conj_mapping_gpu,
                    expLk_full,
                    nao_unpack,
                    axis=0,
                )
                if debug_sync:
                    cp.cuda.get_current_stream().synchronize()
            except Exception as err:
                raise RuntimeError(
                    "Failed in _unpack_cderi_v2 with "
                    f"q={k_aux}, K-range=[{K0}:{K1}), "
                    f"nao_unpack={nao_unpack}, ao_nao={ao_nao}, nkpts={nkpts}, nL={nL_full}, "
                    f"npair_total={npair_total}, len(kj_idx)={len(kj_idx)}, "
                    f"expLk_shape={tuple(expLk_full.shape)}"
                ) from err

            for ki, kj in zip(ki_idx, kj_idx):
                ao_pair = unpacked[ki]

                tmp_left = lib.empty_from_buf(
                    None, (nmo + ncore, block_len, nao_unpack), np.complex128
                )
                lib.contraction(
                    "Lpq",
                    ao_pair,
                    "pi",
                    coeff_left_gpu[int(ki)],
                    "iLq",
                    tmp_left,
                    opb="CONJ",
                )
                tmp_mo = tmp_left[:nmo]
                pq_blk = lib.empty_from_buf(None, (nmo, nmo, block_len), np.complex128)
                lib.contraction(
                    "iLq",
                    tmp_mo,
                    "qj",
                    coeff_mo_gpu[int(kj)],
                    "ijL",
                    pq_blk,
                    alpha=pair_norm,
                )
                pq_host = lib.empty_from_buf(pq_host_buf, (nmo, nmo, block_len), np.complex128)
                pq_blk.get(out=pq_host, blocking=True)
                pq_blk = None
                if weight == 1 and debug_imag_check:
                    pq_debug_sum += pq_host
                _accumulate_real_aux_channels(pq_real, pq_host, weight=weight, sqrt2=sqrt2)
                pq_host = None

                if ncore > 0:
                    tmp_core = tmp_left[nmo:]
                    lib.contraction(
                        "iLq",
                        tmp_core,
                        "qj",
                        coeff_mo_gpu[int(kj)],
                        "ijL",
                        cm_sum,
                        alpha=pair_norm,
                        beta=1.0,
                    )

                    lib.contraction(
                        "iLq",
                        tmp_core,
                        "qi",
                        coeff_core_gpu[int(kj)],
                        "L",
                        cc_sum,
                        alpha=pair_norm,
                        beta=1.0,
                    )

                    tmp_core = None

                tmp_left = None
                tmp_mo = None
                ao_pair = None

            unpacked = None
            cderi_q = None
            if weight == 1 and debug_imag_check:
                info = "K=[%d:%d)" % (K0, K1)
                pq_imag_max, pq_imag_real_max, pq_imag_info = _update_self_conj_imag_stats(
                    pq_debug_sum, pq_imag_max, pq_imag_real_max, pq_imag_info, info
                )
                pq_debug_sum = None

            pq_real_cpu = pq_real.reshape(nov_tot, real_width)
            if weight == 1:
                canonical_cderi_cpu[:, q_aux_base + K0:q_aux_base + K1] = pq_real_cpu
            elif weight == 2:
                real_dst = slice(q_aux_base + K0, q_aux_base + K1)
                imag_dst = slice(q_aux_base + naux_q + K0, q_aux_base + naux_q + K1)
                canonical_cderi_cpu[:, real_dst] = pq_real_cpu[:, :block_len]
                canonical_cderi_cpu[:, imag_dst] = pq_real_cpu[:, block_len:real_width]
            else:
                raise ValueError(f"Unsupported time-reversal weight {weight}")
            if ncore > 0:
                pq_real_d = lib.empty_from_buf(None, (nov_tot, real_width), np.float64)
                pq_real_d.set(pq_real_cpu)
                # pq_real_cpu is a pinned view reused by the next K block; the H2D copy
                # must finish before that host staging buffer can be overwritten.
                cp.cuda.get_current_stream().synchronize()
                if weight == 2:
                    cc_sum *= sqrt2
                    cm_sum *= sqrt2

                cc_real_d = lib.empty_from_buf(None, (real_width, 1), np.float64)
                if weight == 1:
                    cc_real_d[:block_len, 0] = cc_sum.real
                elif weight == 2:
                    cc_real_d[:block_len, 0] = cc_sum.real
                    cc_real_d[block_len:real_width, 0] = cc_sum.imag
                else:
                    raise ValueError(f"Unsupported time-reversal weight {weight}")
                lib.gemm(pq_real_d, cc_real_d, c=vj_gpu, beta=1.0)

                cm_rows_d = lib.empty_from_buf(None, (real_width, ncore, nmo), np.float64)
                cm_sum_Lij = cm_sum.transpose(2, 0, 1)
                if weight == 1:
                    cm_rows_d[:block_len] = cm_sum_Lij.real
                elif weight == 2:
                    cm_rows_d[:block_len] = cm_sum_Lij.real
                    cm_rows_d[block_len:real_width] = cm_sum_Lij.imag
                else:
                    raise ValueError(f"Unsupported time-reversal weight {weight}")
                cm_rows_2d = cm_rows_d.reshape(real_width * ncore, nmo)
                lib.gemm(cm_rows_2d, cm_rows_2d, c=vk_gpu, beta=1.0, transa="T")

                pq_real_d = None
                cc_real_d = None
                cm_sum_Lij = None
                cm_rows_d = None

            pq_real_cpu = None
            cm_sum = None
            cc_sum = None

        q_cderi_host = None
        aux_write_pos += weight * naux_q

        if weight == 1 and debug_imag_check:
            _log_self_conj_imag_result(
                log, "cderi", k_aux, pq_imag_max, pq_imag_real_max, imag_tol, pq_imag_info
            )

        _free_current_eri_pool_blocks()

    if aux_write_pos != total_naux_real:
        raise RuntimeError(
            f"Filled auxiliary width {aux_write_pos}, expected {total_naux_real}"
        )
    compressed_cderi_chunks = compress_canonical_aux_matrix_in_aux_chunks(
        canonical_cderi_cpu,
        svd_tol=svd_tol,
        buffers=svd_buffers,
        logger=log,
        label="eri_high_level_solver_incore_with_jk",
        compress_aux=svd_compress_aux,
        return_chunks=True,
    )
    canonical_cderi_cpu = None
    gc.collect()
    compressed_cderi_cpu = concatenate_canonical_aux_chunks(
        compressed_cderi_chunks,
        nov_tot=nov_tot,
    )
    compressed_cderi_chunks = None
    gc.collect()
    cderi_cut = format_canonical_aux_for_solver(compressed_cderi_cpu, solver_type="CCSD")

    vj = vj_gpu.get(blocking=True).reshape(nmo, nmo)
    vk = vk_gpu.get(blocking=True)

    del coeff_mo_gpu, coeff_core_gpu, coeff_left_gpu, aux_coeffs, cd_j2c_cache, eval_j3c, ao_pair_offsets
    del expLk_full, expLk_conjz_full, bas_ij_aggregated, kpt_iters, stored_kpts, pair_address
    del cderi_q_full_host_buf, cderi_q_batch_host_buf
    del pq_host_buf, pq_real_buf, pq_debug_sum_buf
    if with_long_range:
        del Gv, auxG_conj, eval_ft
    del compressed_cderi_cpu, svd_buffers, vj_gpu, vk_gpu
    _free_current_eri_pool_blocks()

    return cderi_cut, vj, vk
