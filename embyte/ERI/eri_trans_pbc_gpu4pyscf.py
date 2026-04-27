import gc
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


AO_PAIR_BATCH_SIZE_1 = 128 * 128
GBLKSIZE_1 = 1024 * 4
KBLKSIZE_1 = 16
AO_PAIR_BATCH_SIZE_2 = 128 * 128
GBLKSIZE_2 = 1024 * 4
KBLKSIZE_2 = 16
IMAG_TOL = 1e-10
DEFAULT_SVD_COMPRESS_AUX = 20000


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


def _empty_host_canonical_matrix(nrow):
    return cupyx.empty_pinned((int(nrow), 0), dtype=np.float64, order="C")


def _concat_host_canonical_matrices(left, right, *, name):
    left = _as_host_canonical_matrix(left, name=f"{name}_left")
    right = _as_host_canonical_matrix(right, name=f"{name}_right")
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            f"{name}: row mismatch between left {left.shape} and right {right.shape}"
        )

    if left.shape[1] == 0 and right.shape[1] == 0:
        return _empty_host_canonical_matrix(left.shape[0])

    merged = cupyx.empty_pinned(
        (left.shape[0], left.shape[1] + right.shape[1]),
        dtype=np.float64,
        order="C",
    )
    if left.shape[1] > 0:
        merged[:, :left.shape[1]] = left
    if right.shape[1] > 0:
        merged[:, left.shape[1]:] = right
    return merged


def _normalize_incremental_aux_state(active_cpu, *, nov_tot, name):
    nov_tot = int(nov_tot)
    if active_cpu is None:
        return {
            "compressed_cpu": _empty_host_canonical_matrix(nov_tot),
            "pending_cpu": _empty_host_canonical_matrix(nov_tot),
        }

    if isinstance(active_cpu, dict):
        compressed_cpu = active_cpu.get("compressed_cpu")
        pending_cpu = active_cpu.get("pending_cpu")
    else:
        compressed_cpu = active_cpu
        pending_cpu = None

    if compressed_cpu is None:
        compressed_cpu = _empty_host_canonical_matrix(nov_tot)
    else:
        compressed_cpu = _as_host_canonical_matrix(
            compressed_cpu,
            name=f"{name}_compressed_cpu",
        )
    if pending_cpu is None:
        pending_cpu = _empty_host_canonical_matrix(nov_tot)
    else:
        pending_cpu = _as_host_canonical_matrix(
            pending_cpu,
            name=f"{name}_pending_cpu",
        )

    if compressed_cpu.shape[0] != nov_tot:
        raise ValueError(
            f"{name}: compressed row mismatch {compressed_cpu.shape} vs nov_tot={nov_tot}"
        )
    if pending_cpu.shape[0] != nov_tot:
        raise ValueError(
            f"{name}: pending row mismatch {pending_cpu.shape} vs nov_tot={nov_tot}"
        )

    return {
        "compressed_cpu": compressed_cpu,
        "pending_cpu": pending_cpu,
    }


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
        "gram_slice_buf": cp.empty((slice_len_ov * max_aux_eigh,), dtype=np.float64),
        "gram_mat_buf": cp.empty((max_aux_eigh * max_aux_eigh,), dtype=np.float64),
        "proj_slice_buf": cp.empty((slice_len_ov * max_aux_eigh,), dtype=np.float64),
        "proj_out_buf": cp.empty((slice_len_ov * max_aux_out,), dtype=np.float64),
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

    LL_svd = lib.empty_from_buf(
        work_buffers["gram_mat_buf"], (naux, naux), np.float64, order="F"
    )
    LL_svd[:] = 0
    for sov in ovslice_list:
        sov_len = sov.stop - sov.start
        sov_L = lib.empty_from_buf(work_buffers["gram_slice_buf"], (sov_len, naux), np.float64)
        sov_L.set(cderi_cpu[sov])
        lib.gemm(sov_L, sov_L, transa="T", c=LL_svd, beta=1.0)

    cp.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    gc.collect()

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
        ovs_L = lib.empty_from_buf(work_buffers["proj_slice_buf"], (sov_len, naux), np.float64)
        ovs_L.set(cderi_cpu[sov])
        cderi_cut_s = lib.gemm(ovs_L, U_svd, buf=work_buffers["proj_out_buf"], transa="N", transb="N")
        cderi_cut_s_h = lib.empty_from_buf(
            work_buffers["proj_out_host_buf"], (sov_len, naux_cut), np.float64
        )
        cderi_cut_s.get(out=cderi_cut_s_h, blocking=True)
        cderi_cut[sov] = cderi_cut_s_h

    cp.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    gc.collect()

    if owns_buffers:
        del work_buffers
        cp.cuda.get_current_stream().synchronize()
        lib.free_all_blocks()
        gc.collect()

    return cderi_cut


def append_canonical_aux_block(
    active_cpu,
    new_cpu,
    *,
    svd_tol,
    buffers=None,
    logger=None,
    label="incremental_aux_svd",
    compress_aux=DEFAULT_SVD_COMPRESS_AUX,
    finalize=False,
):
    new_cpu = _as_host_canonical_matrix(new_cpu, name="new_cpu")
    nov_tot, naux_new = new_cpu.shape
    compress_aux = int(compress_aux)
    if compress_aux <= 0:
        raise ValueError(f"compress_aux must be positive; got {compress_aux}")
    state = _normalize_incremental_aux_state(
        active_cpu,
        nov_tot=nov_tot,
        name="active_cpu",
    )
    compressed_cpu = state["compressed_cpu"]
    pending_cpu = state["pending_cpu"]

    pos = 0
    while pos < naux_new:
        pending_cols = int(pending_cpu.shape[1])
        if pending_cols >= compress_aux:
            merged_cpu = _concat_host_canonical_matrices(
                compressed_cpu,
                pending_cpu,
                name=f"{label}_compress_input",
            )
            compressed_cpu = _as_host_canonical_matrix(
                compress_canonical_aux_matrix(
                    merged_cpu,
                    svd_tol=svd_tol,
                    buffers=buffers,
                    logger=logger,
                    label=label,
                ),
                name=f"{label}_compressed",
            )
            pending_cpu = _empty_host_canonical_matrix(nov_tot)
            pending_cols = 0
        take = min(naux_new - pos, compress_aux - pending_cols)
        pending_cpu = _concat_host_canonical_matrices(
            pending_cpu,
            new_cpu[:, pos:pos + take],
            name=f"{label}_pending_append",
        )
        pos += take

        if pending_cpu.shape[1] >= compress_aux:
            merged_cpu = _concat_host_canonical_matrices(
                compressed_cpu,
                pending_cpu,
                name=f"{label}_compress_input",
            )
            compressed_cpu = _as_host_canonical_matrix(
                compress_canonical_aux_matrix(
                    merged_cpu,
                    svd_tol=svd_tol,
                    buffers=buffers,
                    logger=logger,
                    label=label,
                ),
                name=f"{label}_compressed",
            )
            pending_cpu = _empty_host_canonical_matrix(nov_tot)

    if finalize and pending_cpu.shape[1] > 0:
        merged_cpu = _concat_host_canonical_matrices(
            compressed_cpu,
            pending_cpu,
            name=f"{label}_finalize_input",
        )
        compressed_cpu = _as_host_canonical_matrix(
            compress_canonical_aux_matrix(
                merged_cpu,
                svd_tol=svd_tol,
                buffers=buffers,
                logger=logger,
                label=label,
            ),
            name=f"{label}_compressed",
        )
        pending_cpu = _empty_host_canonical_matrix(nov_tot)

    return {
        "compressed_cpu": compressed_cpu,
        "pending_cpu": pending_cpu,
    }


def format_canonical_aux_for_solver(cderi_cpu, *, solver_type, out=None):
    if isinstance(cderi_cpu, dict):
        state = _normalize_incremental_aux_state(
            cderi_cpu,
            nov_tot=cderi_cpu["compressed_cpu"].shape[0]
            if cderi_cpu.get("compressed_cpu") is not None
            else cderi_cpu["pending_cpu"].shape[0],
            name="cderi_cpu_state",
        )
        if state["pending_cpu"].shape[1] > 0:
            raise ValueError(
                "Incremental auxiliary state still contains pending columns; "
                "the final append_canonical_aux_block call should use finalize=True"
            )
        cderi_cpu = state["compressed_cpu"]
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


def _accumulate_real_aux_channels(dst, src, *, weight, sqrt2, imag_sign=1.0):
    naux = int(src.shape[-1])
    if weight == 1:
        dst[..., :naux] += src.real
        return _max_abs(src.imag)
    if weight == 2:
        dst[..., :naux] += sqrt2 * src.real
        dst[..., naux:2 * naux] += imag_sign * sqrt2 * src.imag
        return 0.0
    raise ValueError(f"Unsupported time-reversal weight {weight}")


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
    q_j3c_z_buf,
    cderi_q_pair_buf,
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

        q_j3c_z = lib.empty_from_buf(q_j3c_z_buf, (1, naux_cart, pair_size, 2), np.float64)
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

        cderi_q_pair = lib.empty_from_buf(
            cderi_q_pair_buf, (naux_q, pair_size), np.complex128
        )
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

    cp.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    gc.collect()

    if cell.dimension != 3:
        raise NotImplementedError("The current multi-k eri_OVL_SIE_MP2 path only supports 3D cells")
    if auxcell is None:
        raise ValueError("auxcell is required")
    if kpts is None:
        raise ValueError("kpts must be provided for the multi-k eri_OVL_SIE_MP2 path")

    log = logger if logger is not None else g_logger.new_logger(cell, cell.verbose)
    debug_sync = bool(int(os.environ.get("BYTEQC_ERI_TRANS_PBC_DEBUG_SYNC", "1")))

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
    # import ipdb
    # ipdb.set_trace()
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

    coeff_i1_k = np.asarray(
        cp.asnumpy(int3c2e_opt.cell.apply_C_dot(cp.asarray(coeff_i1_k), axis=1)),
        dtype=np.complex128,
        order="C",
    )
    coeff_j1_k = np.asarray(
        cp.asnumpy(int3c2e_opt.cell.apply_C_dot(cp.asarray(coeff_j1_k), axis=1)),
        dtype=np.complex128,
        order="C",
    )
    coeff_i2_k = np.asarray(
        cp.asnumpy(int3c2e_opt.cell.apply_C_dot(cp.asarray(coeff_i2_k), axis=1)),
        dtype=np.complex128,
        order="C",
    )
    coeff_j2_k = np.asarray(
        cp.asnumpy(int3c2e_opt.cell.apply_C_dot(cp.asarray(coeff_j2_k), axis=1)),
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

    coeff_i1_gpu = [cp.asarray(coeff_i1_k[k], dtype=np.complex128, order="C") for k in range(nkpts)]
    coeff_j1_gpu = [cp.asarray(coeff_j1_k[k], dtype=np.complex128, order="C") for k in range(nkpts)]
    coeff_i2_gpu = [cp.asarray(coeff_i2_k[k], dtype=np.complex128, order="C") for k in range(nkpts)]
    coeff_j2_gpu = [cp.asarray(coeff_j2_k[k], dtype=np.complex128, order="C") for k in range(nkpts)]
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
        q_j3c_z_buf = cp.empty((naux_cart * max_pair_size * 2,), dtype=np.float64)
        cderi_q_pair_buf = cp.empty((max_naux_q * max_pair_size,), dtype=np.complex128)
        cderi_q_full_buf = cp.empty((max_kblk * npair_total,), dtype=np.complex128)
        cderi_q_full_host_buf = cupyx.empty_pinned((max_naux_q, npair_total), dtype=np.complex128, order="C")
        cderi_q_batch_host_buf = cupyx.empty_pinned((max_naux_q, max_pair_size), dtype=np.complex128, order="C")
        tmp_iLq_buf = cp.empty((nmo_i1 * max_kblk * nao_unpack,), dtype=np.complex128)
        tmp_jLq_buf = cp.empty((nmo_j2 * max_kblk * nao_unpack,), dtype=np.complex128)
        ov_blk_buf = cp.empty((nmo_i1 * nmo_j1 * max_kblk,), dtype=np.complex128)
        vo_blk_buf = cp.empty((nmo_j2 * nmo_i2 * max_kblk,), dtype=np.complex128)
        ov_host_buf = cupyx.empty_pinned((nmo_i1 * nmo_j1 * max_kblk,), dtype=np.complex128)
        vo_host_buf = cupyx.empty_pinned((nmo_j2 * nmo_i2 * max_kblk,), dtype=np.complex128)
    else:
        q_j3c_z_buf = None
        cderi_q_pair_buf = None
        cderi_q_full_buf = None
        cderi_q_full_host_buf = None
        cderi_q_batch_host_buf = None
        tmp_iLq_buf = None
        tmp_jLq_buf = None
        ov_blk_buf = None
        vo_blk_buf = None
        ov_host_buf = None
        vo_host_buf = None

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
            q_j3c_z_buf=q_j3c_z_buf,
            cderi_q_pair_buf=cderi_q_pair_buf,
            cderi_q_full_host_buf=cderi_q_full_host_buf,
            cderi_q_batch_host_buf=cderi_q_batch_host_buf,
            debug_sync=debug_sync,
        )

        for K0 in range(0, naux_q, Kblksize):
            K1 = min(naux_q, K0 + Kblksize)
            block_len = K1 - K0
            cderi_q = lib.empty_from_buf(
                cderi_q_full_buf, (block_len, npair_total), np.complex128
            )
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
                ov_host = lib.empty_from_buf(ov_host_buf, (nmo_i1, nmo_j1, block_len), np.complex128)
                vo_host = lib.empty_from_buf(vo_host_buf, (nmo_j2, nmo_i2, block_len), np.complex128)

                tmp = lib.empty_from_buf(tmp_iLq_buf, (nmo_i1, block_len, nao_unpack), np.complex128)
                lib.contraction(
                    "Lpq",
                    ao_pair,
                    "pi",
                    coeff_i1_gpu[int(ki)],
                    "iLq",
                    tmp,
                    opb="CONJ",
                )
                ov_blk = lib.empty_from_buf(ov_blk_buf, (nmo_i1, nmo_j1, block_len), np.complex128)
                lib.contraction(
                    "iLq",
                    tmp,
                    "qj",
                    coeff_j1_gpu[int(kj)],
                    "ijL",
                    ov_blk,
                )
                ov_blk *= pair_norm
                ov_blk.get(out=ov_host, blocking=True)

                tmp = lib.empty_from_buf(tmp_jLq_buf, (nmo_j2, block_len, nao_unpack), np.complex128)
                lib.contraction(
                    "Lpq",
                    ao_pair,
                    "pj",
                    coeff_j2_gpu[int(ki)],
                    "jLq",
                    tmp,
                    opb="CONJ",
                )
                vo_blk = lib.empty_from_buf(vo_blk_buf, (nmo_j2, nmo_i2, block_len), np.complex128)
                lib.contraction(
                    "jLq",
                    tmp,
                    "qi",
                    coeff_i2_gpu[int(kj)],
                    "jiL",
                    vo_blk,
                )
                vo_blk *= pair_norm
                vo_blk.get(out=vo_host, blocking=True)

                real_slice = slice(l0 + K0, l0 + K1)
                if weight == 1:
                    ovL_cpu[:, :, real_slice] += ov_host.real
                    voL_cpu[:, :, real_slice] += vo_host.real
                    ov_imag_max = max(ov_imag_max, _max_abs(ov_host.imag))
                    vo_imag_max = max(vo_imag_max, _max_abs(vo_host.imag))
                elif weight == 2:
                    imag_slice = slice(l0 + naux_q + K0, l0 + naux_q + K1)
                    ovL_cpu[:, :, real_slice] += sqrt2 * ov_host.real
                    ovL_cpu[:, :, imag_slice] += sqrt2 * ov_host.imag
                    voL_cpu[:, :, real_slice] += sqrt2 * vo_host.real
                    voL_cpu[:, :, imag_slice] += sqrt2 * vo_host.imag
                else:
                    raise ValueError(f"Unsupported time-reversal weight {weight} for q={k_aux}")

                tmp = None
                ov_blk = None
                vo_blk = None
                ao_pair = None

            unpacked = None
            cderi_q = None

        q_cderi_host = None

        if weight == 1:
            if ov_imag_max > imag_tol:
                log.warn(
                    "ovL q=%d is self-conjugate but max|imag|=%.3e exceeds %.3e; discarding imag part",
                    k_aux,
                    ov_imag_max,
                    imag_tol,
                )
            if vo_imag_max > imag_tol:
                log.warn(
                    "voL q=%d is self-conjugate but max|imag|=%.3e exceeds %.3e; discarding imag part",
                    k_aux,
                    vo_imag_max,
                    imag_tol,
                )
            l0 += naux_q
        else:
            l0 += 2 * naux_q

        cp.cuda.get_current_stream().synchronize()
        lib.free_all_blocks()
        gc.collect()

    del coeff_i1_gpu, coeff_j1_gpu, coeff_i2_gpu, coeff_j2_gpu
    del aux_coeffs, cd_j2c_cache, eval_j3c, ao_pair_offsets, expLk_full, expLk_conjz_full
    del bas_ij_aggregated, kpt_iters, stored_kpts, pair_address
    del q_j3c_z_buf, cderi_q_pair_buf, cderi_q_full_buf, cderi_q_full_host_buf, cderi_q_batch_host_buf
    del tmp_iLq_buf, tmp_jLq_buf, ov_blk_buf, vo_blk_buf, ov_host_buf, vo_host_buf
    if with_long_range:
        del Gv, auxG_conj, eval_ft
    cp.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    gc.collect()

    return ovL_cpu, voL_cpu


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

    cp.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    gc.collect()

    if cell.dimension != 3:
        raise NotImplementedError("The current multi-k eri_high_level_solver_incore path only supports 3D cells")
    if auxcell is None:
        raise ValueError("auxcell is required")
    if kpts is None:
        raise ValueError("kpts must be provided for the multi-k eri_high_level_solver_incore path")
    if mo_coeff_i is None or mo_coeff_j is None:
        raise ValueError("mo_coeff_i and mo_coeff_j are required")

    log = logger if logger is not None else g_logger.new_logger(cell, cell.verbose)
    debug_sync = bool(int(os.environ.get("BYTEQC_ERI_TRANS_PBC_DEBUG_SYNC", "1")))

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
    total_aux_blocks = 0
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
            total_aux_blocks += (naux_q_i + kblk_i - 1) // kblk_i
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
        q_j3c_z_buf = cp.empty((naux_cart * max_pair_size * 2,), dtype=np.float64)
        cderi_q_pair_buf = cp.empty((max_naux_q * max_pair_size,), dtype=np.complex128)
        cderi_q_full_buf = cp.empty((max_kblk * npair_total,), dtype=np.complex128)
        cderi_q_full_host_buf = cupyx.empty_pinned((max_naux_q, npair_total), dtype=np.complex128, order="C")
        cderi_q_batch_host_buf = cupyx.empty_pinned((max_naux_q, max_pair_size), dtype=np.complex128, order="C")
        tmp_iLq_buf = cp.empty((nmo_i * max_kblk * nao_unpack,), dtype=np.complex128)
        pq_blk_buf = cp.empty((nmo_i * nmo_j * max_kblk,), dtype=np.complex128)
        pq_host_buf = cupyx.empty_pinned((nmo_i, nmo_j, max_kblk), dtype=np.complex128, order="C")
        pq_real_buf = cupyx.empty_pinned((nmo_i, nmo_j, max_real_kblk), dtype=np.float64, order="C")
    else:
        q_j3c_z_buf = None
        cderi_q_pair_buf = None
        cderi_q_full_buf = None
        cderi_q_full_host_buf = None
        cderi_q_batch_host_buf = None
        tmp_iLq_buf = None
        pq_blk_buf = None
        pq_host_buf = None
        pq_real_buf = None

    nov_tot = nmo_i * nmo_j
    svd_compress_aux = int(DEFAULT_SVD_COMPRESS_AUX)
    svd_buffer_aux = max(1, min(int(total_naux_real), svd_compress_aux))
    svd_buffers = allocate_incremental_aux_svd_buffers(
        nov_tot,
        max_aux_eigh=svd_buffer_aux,
        max_aux_out=svd_buffer_aux,
    )

    active_cderi_cpu = None
    processed_aux_blocks = 0
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
            q_j3c_z_buf=q_j3c_z_buf,
            cderi_q_pair_buf=cderi_q_pair_buf,
            cderi_q_full_host_buf=cderi_q_full_host_buf,
            cderi_q_batch_host_buf=cderi_q_batch_host_buf,
            debug_sync=debug_sync,
        )

        for K0 in range(0, naux_q, Kblksize):
            K1 = min(naux_q, K0 + Kblksize)
            block_len = K1 - K0
            real_width = block_len if weight == 1 else 2 * block_len
            pq_real = lib.empty_from_buf(pq_real_buf, (nmo_i, nmo_j, real_width), np.float64)
            pq_real[:] = 0.0

            log.info(
                "eri_high_level_solver_incore q=%d aux-block [%d:%d) width=%d",
                k_aux,
                K0,
                K1,
                real_width,
            )

            cderi_q = lib.empty_from_buf(
                cderi_q_full_buf, (block_len, npair_total), np.complex128
            )
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
                tmp = lib.empty_from_buf(tmp_iLq_buf, (nmo_i, block_len, nao_unpack), np.complex128)
                lib.contraction(
                    "Lpq",
                    ao_pair,
                    "pi",
                    coeff_i_gpu[int(ki)],
                    "iLq",
                    tmp,
                    opb="CONJ",
                )
                pq_blk = lib.empty_from_buf(pq_blk_buf, (nmo_i, nmo_j, block_len), np.complex128)
                lib.contraction(
                    "iLq",
                    tmp,
                    "qj",
                    coeff_j_gpu[int(kj)],
                    "ijL",
                    pq_blk,
                )
                pq_blk *= pair_norm
                pq_host = lib.empty_from_buf(pq_host_buf, (nmo_i, nmo_j, block_len), np.complex128)
                pq_blk.get(out=pq_host, blocking=True)
                pq_imag_max = max(
                    pq_imag_max,
                    _accumulate_real_aux_channels(pq_real, pq_host, weight=weight, sqrt2=sqrt2),
                )
                tmp = None
                pq_blk = None
                pq_host = None
                ao_pair = None

            unpacked = None
            cderi_q = None

            processed_aux_blocks += 1
            pq_real_view = pq_real.reshape(nov_tot, real_width)
            active_cderi_cpu = append_canonical_aux_block(
                active_cderi_cpu,
                pq_real_view,
                svd_tol=svd_tol,
                buffers=svd_buffers,
                logger=log,
                label="eri_high_level_solver_incore incremental",
                compress_aux=svd_compress_aux,
                finalize=(processed_aux_blocks == total_aux_blocks),
            )

        q_cderi_host = None

        if weight == 1 and pq_imag_max > imag_tol:
            log.warn(
                "cderi q=%d is self-conjugate but max|imag|=%.3e exceeds %.3e; discarding imag part",
                k_aux,
                pq_imag_max,
                imag_tol,
            )

        cp.cuda.get_current_stream().synchronize()
        lib.free_all_blocks()
        gc.collect()

    cderi_cut = format_canonical_aux_for_solver(active_cderi_cpu, solver_type=solver_type)

    del coeff_i_gpu, coeff_j_gpu, aux_coeffs, cd_j2c_cache, eval_j3c, ao_pair_offsets
    del expLk_full, expLk_conjz_full, bas_ij_aggregated, kpt_iters, stored_kpts, pair_address
    del q_j3c_z_buf, cderi_q_pair_buf, cderi_q_full_buf, cderi_q_full_host_buf, cderi_q_batch_host_buf
    del tmp_iLq_buf, pq_blk_buf, pq_host_buf, pq_real_buf
    if with_long_range:
        del Gv, auxG_conj, eval_ft
    del active_cderi_cpu, svd_buffers
    cp.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    gc.collect()

    return cderi_cut


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

    cp.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    gc.collect()

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
    debug_sync = bool(int(os.environ.get("BYTEQC_ERI_TRANS_PBC_DEBUG_SYNC", "1")))

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
    total_aux_blocks = 0
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
            total_aux_blocks += (naux_q_i + kblk_i - 1) // kblk_i
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
        q_j3c_z_buf = cp.empty((naux_cart * max_pair_size * 2,), dtype=np.float64)
        cderi_q_pair_buf = cp.empty((max_naux_q * max_pair_size,), dtype=np.complex128)
        cderi_q_full_buf = cp.empty((max_kblk * npair_total,), dtype=np.complex128)
        cderi_q_full_host_buf = cupyx.empty_pinned((max_naux_q, npair_total), dtype=np.complex128, order="C")
        cderi_q_batch_host_buf = cupyx.empty_pinned((max_naux_q, max_pair_size), dtype=np.complex128, order="C")
        tmp_left_buf = cp.empty(((nmo + ncore) * max_kblk * nao_unpack,), dtype=np.complex128)
        pq_blk_buf = cp.empty((nmo * nmo * max_kblk,), dtype=np.complex128)
        pq_host_buf = cupyx.empty_pinned((nmo, nmo, max_kblk), dtype=np.complex128, order="C")
        pq_real_buf = cupyx.empty_pinned((nmo, nmo, max_real_kblk), dtype=np.float64, order="C")
        pq_real_d_buf = cp.empty((nov_tot * max_real_kblk,), dtype=np.float64)
        if ncore > 0:
            cm_blk_buf = cp.empty((ncore * nmo * max_kblk,), dtype=np.complex128)
            cm_sum_buf = cp.empty((ncore * nmo * max_kblk,), dtype=np.complex128)
            cm_rows_d_buf = cp.empty((max_real_kblk * ncore * nmo,), dtype=np.float64)
            cc_trace_buf = cp.empty((max_kblk,), dtype=np.complex128)
            cc_sum_buf = cp.empty((max_kblk,), dtype=np.complex128)
            cc_real_d_buf = cp.empty((max_real_kblk,), dtype=np.float64)
        else:
            cm_blk_buf = None
            cm_sum_buf = None
            cm_rows_d_buf = None
            cc_trace_buf = None
            cc_sum_buf = None
            cc_real_d_buf = None
    else:
        q_j3c_z_buf = None
        cderi_q_pair_buf = None
        cderi_q_full_buf = None
        cderi_q_full_host_buf = None
        cderi_q_batch_host_buf = None
        tmp_left_buf = None
        pq_blk_buf = None
        pq_host_buf = None
        pq_real_buf = None
        pq_real_d_buf = None
        cm_blk_buf = None
        cm_sum_buf = None
        cm_rows_d_buf = None
        cc_trace_buf = None
        cc_sum_buf = None
        cc_real_d_buf = None

    svd_compress_aux = int(DEFAULT_SVD_COMPRESS_AUX)
    svd_buffer_aux = max(1, min(int(total_naux_real), svd_compress_aux))
    svd_buffers = allocate_incremental_aux_svd_buffers(
        nov_tot,
        max_aux_eigh=svd_buffer_aux,
        max_aux_out=svd_buffer_aux,
    )
    active_cderi_cpu = None
    vj_gpu = cp.zeros((nov_tot, 1), dtype=np.float64)
    vk_gpu = cp.zeros((nmo, nmo), dtype=np.float64)
    processed_aux_blocks = 0

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
            q_j3c_z_buf=q_j3c_z_buf,
            cderi_q_pair_buf=cderi_q_pair_buf,
            cderi_q_full_host_buf=cderi_q_full_host_buf,
            cderi_q_batch_host_buf=cderi_q_batch_host_buf,
            debug_sync=debug_sync,
        )

        for K0 in range(0, naux_q, Kblksize):
            K1 = min(naux_q, K0 + Kblksize)
            block_len = K1 - K0
            real_width = block_len if weight == 1 else 2 * block_len

            pq_real = lib.empty_from_buf(pq_real_buf, (nmo, nmo, real_width), np.float64)
            pq_real[:] = 0.0
            if ncore > 0:
                cm_sum = lib.empty_from_buf(
                    cm_sum_buf, (ncore, nmo, block_len), np.complex128
                )
                cm_sum[:] = 0.0
                cc_sum = lib.empty_from_buf(cc_sum_buf, (block_len,), np.complex128)
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

            cderi_q = lib.empty_from_buf(
                cderi_q_full_buf, (block_len, npair_total), np.complex128
            )
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
                    tmp_left_buf, (nmo + ncore, block_len, nao_unpack), np.complex128
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
                pq_blk = lib.empty_from_buf(pq_blk_buf, (nmo, nmo, block_len), np.complex128)
                lib.contraction(
                    "iLq",
                    tmp_mo,
                    "qj",
                    coeff_mo_gpu[int(kj)],
                    "ijL",
                    pq_blk,
                )
                pq_blk *= pair_norm
                pq_host = lib.empty_from_buf(pq_host_buf, (nmo, nmo, block_len), np.complex128)
                pq_blk.get(out=pq_host, blocking=True)
                pq_imag_max = max(
                    pq_imag_max,
                    _accumulate_real_aux_channels(pq_real, pq_host, weight=weight, sqrt2=sqrt2),
                )

                if ncore > 0:
                    tmp_core = tmp_left[nmo:]
                    cm_blk = lib.empty_from_buf(
                        cm_blk_buf, (ncore, nmo, block_len), np.complex128
                    )
                    lib.contraction(
                        "iLq",
                        tmp_core,
                        "qj",
                        coeff_mo_gpu[int(kj)],
                        "ijL",
                        cm_blk,
                    )
                    cm_blk *= pair_norm
                    cm_sum += cm_blk

                    cc_trace = lib.empty_from_buf(
                        cc_trace_buf, (block_len,), np.complex128
                    )
                    lib.contraction(
                        "iLq",
                        tmp_core,
                        "qi",
                        coeff_core_gpu[int(kj)],
                        "L",
                        cc_trace,
                    )
                    cc_trace *= pair_norm
                    cc_sum += cc_trace

                    cm_blk = None
                    cc_trace = None

                tmp_left = None
                tmp_mo = None
                pq_blk = None
                pq_host = None
                ao_pair = None

            unpacked = None
            cderi_q = None

            processed_aux_blocks += 1
            pq_real_cpu = pq_real.reshape(nov_tot, real_width)
            if ncore > 0:
                pq_real_d = lib.empty_from_buf(
                    pq_real_d_buf, (nov_tot, real_width), np.float64
                )
                pq_real_d.set(pq_real_cpu)

                cc_real_d = lib.empty_from_buf(
                    cc_real_d_buf, (real_width, 1), np.float64
                )
                if weight == 1:
                    cc_real_d[:block_len, 0] = cc_sum.real
                elif weight == 2:
                    cc_real_d[:block_len, 0] = sqrt2 * cc_sum.real
                    cc_real_d[block_len:real_width, 0] = sqrt2 * cc_sum.imag
                else:
                    raise ValueError(f"Unsupported time-reversal weight {weight}")
                lib.gemm(pq_real_d, cc_real_d, c=vj_gpu, beta=1.0)

                cm_rows_d = lib.empty_from_buf(
                    cm_rows_d_buf, (real_width, ncore, nmo), np.float64
                )
                cm_sum_Lij = cm_sum.transpose(2, 0, 1)
                if weight == 1:
                    cm_rows_d[:block_len] = cm_sum_Lij.real
                elif weight == 2:
                    cm_rows_d[:block_len] = sqrt2 * cm_sum_Lij.real
                    cm_rows_d[block_len:real_width] = sqrt2 * cm_sum_Lij.imag
                else:
                    raise ValueError(f"Unsupported time-reversal weight {weight}")
                cm_rows_2d = cm_rows_d.reshape(real_width * ncore, nmo)
                lib.gemm(cm_rows_2d, cm_rows_2d, c=vk_gpu, beta=1.0, transa="T")

                pq_real_d = None
                cc_real_d = None
                cm_sum_Lij = None
                cm_rows_d = None

            active_cderi_cpu = append_canonical_aux_block(
                active_cderi_cpu,
                pq_real_cpu,
                svd_tol=svd_tol,
                buffers=svd_buffers,
                logger=log,
                label="eri_high_level_solver_incore_with_jk incremental",
                compress_aux=svd_compress_aux,
                finalize=(processed_aux_blocks == total_aux_blocks),
            )
            cm_sum = None
            cc_sum = None

        q_cderi_host = None

        if weight == 1:
            if pq_imag_max > imag_tol:
                log.warn(
                    "cderi q=%d is self-conjugate but max|imag(pq)|=%.3e exceeds %.3e; discarding imag part",
                    k_aux,
                    pq_imag_max,
                    imag_tol,
                )

        cp.cuda.get_current_stream().synchronize()
        lib.free_all_blocks()
        gc.collect()

    cderi_cut = format_canonical_aux_for_solver(active_cderi_cpu, solver_type="CCSD")

    vj = vj_gpu.get(blocking=True).reshape(nmo, nmo)
    vk = vk_gpu.get(blocking=True)

    del coeff_mo_gpu, coeff_core_gpu, coeff_left_gpu, aux_coeffs, cd_j2c_cache, eval_j3c, ao_pair_offsets
    del expLk_full, expLk_conjz_full, bas_ij_aggregated, kpt_iters, stored_kpts, pair_address
    del q_j3c_z_buf, cderi_q_pair_buf, cderi_q_full_buf, cderi_q_full_host_buf, cderi_q_batch_host_buf
    del tmp_left_buf, pq_blk_buf, pq_host_buf, pq_real_buf, pq_real_d_buf
    del cm_blk_buf, cm_sum_buf, cm_rows_d_buf
    del cc_trace_buf, cc_sum_buf, cc_real_d_buf
    if with_long_range:
        del Gv, auxG_conj, eval_ft
    del active_cderi_cpu, svd_buffers, vj_gpu, vk_gpu
    cp.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    gc.collect()

    return cderi_cut, vj, vk
