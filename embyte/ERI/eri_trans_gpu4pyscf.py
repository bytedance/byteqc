import sys
import os
import numpy
import cupy
import cupyx
import gc
from byteqc import lib
from pyscf.lib import prange
from gpu4pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc.df.int3c2e import SRInt3c2eOpt
from gpu4pyscf.pbc.df.rsdf_builder import (
    _precontract_j2c_aux_coeff,
    estimate_ke_cutoff_for_omega,
    _weighted_coulG_LR,
    OMEGA_MIN,
    LINEAR_DEP_THR,
)

cupy.cuda.set_pinned_memory_allocator(None)

AO_PAIR_BATCH_SIZE_1 = 256 * 256
GBLKSIZE_1 = 1024 * 4
KBLKSIZE_1 = 64
AO_PAIR_BATCH_SIZE_2 = 256 * 256
GBLKSIZE_2 = 1024 * 4
KBLKSIZE_2 = 64


def _validate_gamma_kpts(kpts):
    if kpts is None:
        return None
    if hasattr(kpts, 'kpts'):
        kpts = kpts.kpts
    kpts = numpy.asarray(kpts, dtype=float)
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise ValueError(f'Invalid kpts shape {kpts.shape}, expected (nkpts, 3)')
    if len(kpts) > 1:
        raise ValueError(
            'eri_trans_gpu4pyscf only supports gamma-point inputs. '
            'Use byteqc.embyte.ERI.eri_trans or eri_trans_pbc_gpu4pyscf for multi-k calculations.'
        )
    return kpts


def _to_real_or_raise(arr, name, atol=1e-14, rtol=1e-10):
    """Convert an array to float64; raise if non-negligible imaginary part exists."""
    arr = cupy.asarray(arr)
    if cupy.iscomplexobj(arr):
        if arr.size == 0:
            return cupy.asarray(arr.real, dtype=cupy.float64, order="C")
        max_abs_real = float(cupy.max(cupy.abs(arr.real)))
        max_abs_imag = float(cupy.max(cupy.abs(arr.imag)))
        tol = max(atol, rtol * max_abs_real)
        if max_abs_imag > tol:
            raise ValueError(
                f"{name} contains non-negligible imaginary part: "
                f"max|imag|={max_abs_imag}, max|real|={max_abs_real}, tol={tol}"
            )
        return cupy.asarray(arr.real, dtype=cupy.float64, order="C")
    return cupy.asarray(arr, dtype=cupy.float64, order="C")


def _sqrt_psd_eigs_or_raise(S, min_eig_allowed=-1e-10):
    """
    For Gram-like spectra: allow tiny negative noise, but reject significant negatives.
    """
    smin = float(cupy.min(S))
    if smin < min_eig_allowed:
        smax = float(cupy.max(S))
        raise RuntimeError(
            f"Encountered significant negative eigenvalue: min={smin}, "
            f"allowed_min={min_eig_allowed}, max={smax}"
        )
    if smin < 0.0:
        S = cupy.maximum(S, 0.0)
    return cupy.sqrt(S)


def _fit_aux_ft_block(aux_coeff, auxG_cart_raw, buf, naux_cart):
    if auxG_cart_raw.ndim != 2:
        raise ValueError(
            f"Unexpected ft_ao output ndim={auxG_cart_raw.ndim}, expected 2"
        )
    if auxG_cart_raw.shape[0] == naux_cart:
        return lib.gemm(
            aux_coeff, auxG_cart_raw, buf=buf, transa='T', transb='N'
        )
    if auxG_cart_raw.shape[1] == naux_cart:
        return lib.gemm(
            aux_coeff, auxG_cart_raw, buf=buf, transa='T', transb='H'
        )
    raise ValueError(
        f"Unexpected ft_ao output shape {auxG_cart_raw.shape}; "
        f"neither axis matches naux_cart={naux_cart}"
    )


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
    **kwargs,
):
    _validate_gamma_kpts(kpts)
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
    
    cupy.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    gc.collect()
    

    if omega is None:
        omega = OMEGA_MIN

    kmesh = numpy.array([1, 1, 1])

    int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega=-omega, bvk_kmesh=kmesh).build()
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell

    cd_j2c_cache, _negative_metric_size = _precontract_j2c_aux_coeff(
        auxcell, None, omega, with_long_range, linear_dep_threshold
    )
    aux_coeff0 = _to_real_or_raise(cd_j2c_cache[0], "cd_j2c_cache[0]")
    naux_cart, naux_fit = aux_coeff0.shape

    cderi_idx = int3c2e_opt.pair_and_diag_indices()
    if with_long_range:
        omega_lr = abs(int3c2e_opt.omega)
        ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega_lr)
        mesh = cell.cutoff_to_mesh(ke_cutoff)
        mesh = cell.symmetrize_mesh(mesh)
        Gv, _Gvbase, kws = cell.get_Gv_weights(mesh)
        ngrids = len(Gv)
        coulG = _to_real_or_raise(_weighted_coulG_LR(auxcell, Gv, omega_lr, kws), "coulG")

        ft_opt = ft_ao.FTOpt(cell, kmesh)
        ft_opt.__dict__.update(int3c2e_opt.__dict__)
        ft_opt._aft_envs = int3c2e_opt._int3c2e_envs
    else:
        Gv = None
        ft_opt = None

    nsp_per_block = ft_ao.ft_ao_scheme()[0]
    bas_ij_aggregated = cell.aggregate_shl_pairs(int3c2e_opt.bas_ij_cache, nsp_per_block)

    # Use a single full auxiliary batch for SR part (no aux slicing).
    eval_j3c, aux_sorting, ao_pair_offsets, _aux_offsets = int3c2e_opt.int3c2e_evaluator(
        ao_pair_batch_size=ao_pair_batch_size,
        aux_batch_size=None,
        bas_ij_aggregated=bas_ij_aggregated,
    )
    _aux_offsets = None

    if hasattr(ao_pair_offsets, "get"):
        ao_pair_offsets = ao_pair_offsets.get()
    ao_pair_offsets = numpy.asarray(ao_pair_offsets, dtype=numpy.int64)

    if with_long_range:
        eval_ft, _ao_pair_offsets_ft = ft_opt.ft_evaluator(
            ao_pair_batch_size, bas_ij_aggregated=bas_ij_aggregated
        )
        if not numpy.array_equal(ao_pair_offsets, _ao_pair_offsets_ft):
            raise RuntimeError("AO-pair offsets mismatch between SR and LR evaluators")
        _ao_pair_offsets_ft = None
        # Use complex128 to keep lib.gemm on the cublas GEMM path for LR fit build.
        aux_coeff0_c = cupy.asarray(aux_coeff0, dtype=cupy.complex128, order="C")

        if Gblksize is None:
            mem_free = cupy.cuda.runtime.memGetInfo()[0]
            # Rough memory model for one LR block:
            # auxG_fit_blk(naux_fit,gblk,complex128) + pqG(npair,gblk,complex128)
            elem_bytes = 16
            avail_mem = max(0, mem_free - 512 * 1024 * 1024)
            denom = elem_bytes * max(1, (naux_fit + ao_pair_batch_size))
            Gblksize = int(avail_mem // denom) // 32 * 32
            Gblksize = max(32, min(Gblksize, ngrids))

        g_slices = list(prange(0, ngrids, Gblksize))
        # Precompute LR aux-fit blocks on GPU then store on host memory.
        # This avoids keeping the full auxG_fit tensor on device.
        auxG_fit_blocks_cpu = []
        auxG_fit_buf = cupy.empty((naux_fit, Gblksize), dtype=cupy.complex128, order="C")
        for g0, g1 in g_slices:
            logger.info(
                'eri_OVL_SIE_MP2, get int3c long range part, g_slices:%d/%d' %
                (g0, ngrids))
            auxG_cart_raw = cupy.asarray(
                # ft_ao.ft_ao(auxcell, Gv[g0:g1], sort_cell=False),
                ft_ao.ft_ao(auxcell, Gv[g0:g1]),
                dtype=cupy.complex128,
                order="C",
            )

            auxG_fit_blk = _fit_aux_ft_block(
                aux_coeff0_c, auxG_cart_raw, auxG_fit_buf, naux_cart
            )
            auxG_fit_blk *= coulG[g0:g1]

            auxG_fit_blk_host = cupyx.empty_pinned(
                auxG_fit_blk.shape, dtype=auxG_fit_blk.dtype, order="C"
            )
            auxG_fit_blk.get(out=auxG_fit_blk_host, blocking=True)
            auxG_fit_blocks_cpu.append(auxG_fit_blk_host)
            auxG_cart_raw = None
            auxG_fit_blk = None
        aux_coeff0_c = None
        coulG = None
        auxG_fit_buf = None
    else:
        eval_ft = None
        g_slices = []
        auxG_fit_blocks_cpu = []
        ngrids = 0
        Gblksize = 0
    bas_ij_aggregated = None

    aux_coeff = cupy.empty_like(aux_coeff0)
    aux_coeff[aux_sorting] = aux_coeff0
    aux_coeff0 = None
    aux_sorting = None

    shl_pair_batches = int(len(ao_pair_offsets) - 1)
    if mo_coeff_i1 is None or mo_coeff_j1 is None:
        raise ValueError("mo_coeff_i1 and mo_coeff_j1 are required")
    if mo_coeff_i2 is None or mo_coeff_j2 is None:
        raise ValueError("mo_coeff_i2 and mo_coeff_j2 are required")
    mo_coeff_i1 = cupy.asarray(mo_coeff_i1, dtype=cupy.float64, order="C")
    mo_coeff_j1 = cupy.asarray(mo_coeff_j1, dtype=cupy.float64, order="C")
    if mo_coeff_i1.shape[0] != mo_coeff_j1.shape[0]:
        raise ValueError("mo_coeff_i1 and mo_coeff_j1 must have the same AO dimension")
    mo_coeff_i2 = cupy.asarray(mo_coeff_i2, dtype=cupy.float64, order="C")
    mo_coeff_j2 = cupy.asarray(mo_coeff_j2, dtype=cupy.float64, order="C")
    if mo_coeff_i2.shape[0] != mo_coeff_j2.shape[0]:
        raise ValueError("mo_coeff_i2 and mo_coeff_j2 must have the same AO dimension")
    nmo_i1 = int(mo_coeff_i1.shape[1])
    nmo_j1 = int(mo_coeff_j1.shape[1])
    nmo_i2 = int(mo_coeff_i2.shape[1])
    nmo_j2 = int(mo_coeff_j2.shape[1])

    nao = int(mo_coeff_i1.shape[0])
    pair_address = cupy.asarray(cderi_idx[0], dtype=cupy.int64)
    diag_idx = cupy.asarray(cderi_idx[1], dtype=cupy.int64)
    if pair_address.size > 0:
        max_addr = int(pair_address.max().get())
        if max_addr >= nao * nao:
            raise ValueError(
                f"AO dimension mismatch: max(pair_address)={max_addr}, "
                f"but mo_coeff AO dimension={nao}"
            )

    pair_weight = cupy.ones((pair_address.size,), dtype=cupy.float64)
    if diag_idx.size > 0:
        pair_weight[diag_idx] = 0.5
    diag_idx = None

    if Lblksize is None:
        Kblksize = naux_fit
    else:
        Kblksize = max(1, min(int(Lblksize), naux_fit))

    max_nao_blk = 0
    for p0, p1 in zip(ao_pair_offsets[:-1], ao_pair_offsets[1:]):
        if p1 <= p0:
            continue
        addr = pair_address[p0:p1]
        i_blk = addr // nao
        j_blk = addr % nao
        ij_cat_blk = cupy.concatenate((i_blk, j_blk), axis=0)
        nao_blk_tmp = int(cupy.unique(ij_cat_blk).size)
        if nao_blk_tmp > max_nao_blk:
            max_nao_blk = nao_blk_tmp
        addr = None
        i_blk = None
        j_blk = None
        ij_cat_blk = None

    npair_blk_max = int(numpy.max(ao_pair_offsets[1:] - ao_pair_offsets[:-1]))
    # Accumulate in host memory to reduce device-memory pressure for large MO spaces.
    ovL_cpu = cupyx.zeros_pinned((nmo_i1, nmo_j1, naux_fit), dtype=numpy.float64, order="C")
    voL_cpu = cupyx.zeros_pinned((nmo_j2, nmo_i2, naux_fit), dtype=numpy.float64, order="C")
    out_pqk_host = cupyx.empty_pinned((Kblksize, max(nmo_i1*nmo_j1, nmo_i2*nmo_j2)), dtype=numpy.float64, order="C")
    
    j3c_fit_pair_buf = cupy.empty((naux_fit, npair_blk_max), dtype=cupy.float64, order="C")
    pqG_buf = cupy.empty((npair_blk_max, Gblksize), dtype=cupy.complex128, order="C")
    j3c_raw_buf = cupy.empty(
        (
            max(
                naux_cart*npair_blk_max,
                int(Kblksize*max_nao_blk*max_nao_blk/2)+1,
                int(Kblksize*max_nao_blk*max(nmo_i1, nmo_j2)/2)+1,
                int(Kblksize*max(nmo_j1*nmo_i1, nmo_j2*nmo_i2)/2)+1,
                    ),
            ),
        dtype=cupy.complex128,
        order="C")

    auxG_fit_block_buf = cupy.empty(
        (
            max(
                naux_cart * max(1, Gblksize),
                int(Kblksize * max(nmo_j1*nmo_i1, nmo_j2*nmo_i2) / 2) + 1,
            ),
        ),
        dtype=cupy.complex128,
        order="C",
    )

    j3c_tril_buf = cupy.empty(
        (Kblksize*max_nao_blk*max_nao_blk),
        dtype=cupy.float64,
        order="C")

    for ij_batch_id in range(shl_pair_batches):
        logger.info(
            'eri_OVL_SIE_MP2, get int3c transform, ij_batch_id:%d/%d' %
            (ij_batch_id, shl_pair_batches))
        p0 = int(ao_pair_offsets[ij_batch_id])
        p1 = int(ao_pair_offsets[ij_batch_id + 1])
        if p1 <= p0:
            continue

        addr = pair_address[p0:p1]
        npair_blk = int(p1 - p0)
        i = addr // nao
        j = addr % nao
        ij_cat = cupy.concatenate((i, j), axis=0)
        ao_u, inv = cupy.unique(ij_cat, return_inverse=True)
        inv_i = inv[:npair_blk]
        inv_j = inv[npair_blk:]
        nao_blk = int(ao_u.size)
        local_addr = inv_i.astype(cupy.int64) * nao_blk + inv_j.astype(cupy.int64)
        Ci1_blk = mo_coeff_i1[ao_u]
        Cj1_blk = mo_coeff_j1[ao_u]
        Ci2_blk = mo_coeff_i2[ao_u]
        Cj2_blk = mo_coeff_j2[ao_u]

        w = pair_weight[p0:p1]

        # Build the full fit-space pair block once for each AO-pair batch:
        # J_fit(K,pair) = aux_coeff^T * J_sr_cart + auxG_fit * J_lr_G
        j3c_fit_pair = lib.empty_from_buf(j3c_fit_pair_buf, (naux_fit, npair_blk), dtype=cupy.float64)
        j3c_fit_pair[:] = 0

        # SR contribution: evaluate one full auxiliary batch.
        j3c_raw = eval_j3c(shl_pair_batch_id=ij_batch_id, aux_batch_id=0, out=j3c_raw_buf)
        if j3c_raw.dtype != cupy.float64:
            raise TypeError(f"Unexpected j3c_raw dtype: {j3c_raw.dtype}, expected float64")
        if j3c_raw.size > 0:
            j3c_cart = j3c_raw[:, :, 0]
            lib.gemm(aux_coeff, j3c_cart, c=j3c_fit_pair, transa='T', transb='T')
            j3c_cart = None
        j3c_raw = None

        # LR contribution: accumulate in G-blocks to control memory usage.
        if with_long_range:
            for iblk, (g0, g1) in enumerate(g_slices):
                auxG_fit_blk = lib.empty_from_buf(auxG_fit_block_buf, auxG_fit_blocks_cpu[iblk].shape, dtype=cupy.complex128)
                auxG_fit_blk.set(auxG_fit_blocks_cpu[iblk])
                # auxG_fit_blk = cupy.asarray(auxG_fit_blocks_cpu[iblk])
                pqG = eval_ft(Gv[g0:g1], ij_batch_id, out=pqG_buf)
                j3c_fit_pair_lr = lib.gemm(auxG_fit_blk, pqG, buf=j3c_raw_buf, transb='T')
                j3c_fit_pair += j3c_fit_pair_lr.real
                pqG = None
                auxG_fit_blk = None
                j3c_fit_pair_lr = None

        # AO2MO in K-blocks to limit GPU memory footprint.
        for K0 in range(0, naux_fit, Kblksize):
            K1 = min(naux_fit, K0 + Kblksize)
            Kblk = int(K1 - K0)
            j3c_pair_blk = lib.empty_from_buf(j3c_raw_buf, (Kblk, npair_blk), dtype=cupy.float64)
            j3c_pair_blk[:] = j3c_fit_pair[K0:K1]
            j3c_pair_blk *= w[None, :]
            j3c_tril = lib.empty_from_buf(j3c_tril_buf, (Kblk, nao_blk * nao_blk), dtype=cupy.float64)
            j3c_tril[:] = 0
            j3c_tril[:, local_addr] = j3c_pair_blk
            j3c_tril = j3c_tril.reshape(Kblk, nao_blk, nao_blk)
            j3c_tril_trans = lib.empty_from_buf(j3c_raw_buf, (Kblk, nao_blk, nao_blk), dtype=cupy.float64)
            j3c_tril_trans[:] = j3c_tril.transpose(0, 2, 1)
            j3c_tril += j3c_tril_trans

            # Two-step AO2MO: Kij,ip->Kpj; Kpj,jq->Kpq
            # Kij,jq->qKi
            tmp = lib.gemm(Ci1_blk, j3c_tril.reshape(-1, nao_blk), buf=j3c_raw_buf, transa='T', transb='T')
            # qKi,ip->pqK
            out_pqk = lib.gemm(Cj1_blk, tmp.reshape(-1, nao_blk), buf=auxG_fit_block_buf, transa='T', transb='T')
            out_pqk = out_pqk.reshape(nmo_j1, nmo_i1, Kblk)
            out_qpk = lib.empty_from_buf(j3c_raw_buf, (nmo_i1, nmo_j1, Kblk), 'f8')
            out_qpk[:] = out_pqk.transpose(1, 0, 2)
            out_qpk_tmp = lib.empty_from_buf(out_pqk_host, out_qpk.shape, 'f8')
            out_qpk.get(out=out_qpk_tmp, blocking=True)
            ovL_cpu[:, :, K0:K1] += out_qpk_tmp

            # Two-step AO2MO: Kij,ip->Kpj; Kpj,jq->Kpq
            # Kij,jq->qKi
            tmp = lib.gemm(Cj2_blk, j3c_tril.reshape(-1, nao_blk), buf=j3c_raw_buf, transa='T', transb='T')
            # qKi,ip->pqK
            out_pqk = lib.gemm(Ci2_blk, tmp.reshape(-1, nao_blk), buf=auxG_fit_block_buf, transa='T', transb='T')
            out_pqk = out_pqk.reshape(nmo_i2, nmo_j2, Kblk)
            out_qpk = lib.empty_from_buf(j3c_raw_buf, (nmo_j2, nmo_i2, Kblk), 'f8')
            out_qpk[:] = out_pqk.transpose(1, 0, 2)
            out_qpk_tmp = lib.empty_from_buf(out_pqk_host, out_qpk.shape, 'f8')
            out_qpk.get(out=out_qpk_tmp, blocking=True)
            voL_cpu[:, :, K0:K1] += out_qpk_tmp

            j3c_pair_blk = None
            j3c_tril = None
            j3c_tril_trans = None
            tmp = None
            out_pqk = None
            out_qpk = None
            out_qpk_tmp = None

        j3c_fit_pair = None
        addr = None
        i = None
        j = None
        ij_cat = None
        ao_u = None
        inv = None
        inv_i = None
        inv_j = None
        local_addr = None
        Ci1_blk = None
        Cj1_blk = None
        Ci2_blk = None
        Cj2_blk = None
        w = None

    j3c_fit_pair_buf = None
    pqG_buf = None
    j3c_raw_buf = None
    j3c_tril_buf = None
    auxG_fit_block_buf = None
    auxG_fit_blocks_cpu = None
    pair_weight = None
    pair_address = None
    aux_coeff = None
    eval_j3c = None
    eval_ft = None
    Gv = None
    ft_opt = None
    int3c2e_opt = None

    out_pqk_host = None

    return ovL_cpu, voL_cpu


def eri_high_level_solver_incore(
    cell,
    auxcell,
    mo_coeff_i,
    mo_coeff_j,
    *args,
    solver_type='MP2',
    svd_tol=1e-4,
    omega=None,
    linear_dep_threshold=LINEAR_DEP_THR,
    with_long_range=True,
    ao_pair_batch_size=AO_PAIR_BATCH_SIZE_1,
    Lblksize=KBLKSIZE_1,
    Gblksize=GBLKSIZE_1,
    kpts=None,
    **kwargs,
):
    _validate_gamma_kpts(kpts)
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
    
    cupy.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    gc.collect()
    

    if omega is None:
        omega = OMEGA_MIN

    kmesh = numpy.array([1, 1, 1])

    int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega=-omega, bvk_kmesh=kmesh).build()
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell

    cd_j2c_cache, _negative_metric_size = _precontract_j2c_aux_coeff(
        auxcell, None, omega, with_long_range, linear_dep_threshold
    )
    aux_coeff0 = _to_real_or_raise(cd_j2c_cache[0], "cd_j2c_cache[0]")
    naux_cart, naux_fit = aux_coeff0.shape

    cderi_idx = int3c2e_opt.pair_and_diag_indices()
    if with_long_range:
        omega_lr = abs(int3c2e_opt.omega)
        ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega_lr)
        mesh = cell.cutoff_to_mesh(ke_cutoff)
        mesh = cell.symmetrize_mesh(mesh)
        Gv, _Gvbase, kws = cell.get_Gv_weights(mesh)
        ngrids = len(Gv)
        coulG = _to_real_or_raise(_weighted_coulG_LR(auxcell, Gv, omega_lr, kws), "coulG")

        ft_opt = ft_ao.FTOpt(cell, kmesh)
        ft_opt.__dict__.update(int3c2e_opt.__dict__)
        ft_opt._aft_envs = int3c2e_opt._int3c2e_envs
    else:
        Gv = None
        ft_opt = None

    nsp_per_block = ft_ao.ft_ao_scheme()[0]
    bas_ij_aggregated = cell.aggregate_shl_pairs(int3c2e_opt.bas_ij_cache, nsp_per_block)

    # Use a single full auxiliary batch for SR part (no aux slicing).
    eval_j3c, aux_sorting, ao_pair_offsets, _aux_offsets = int3c2e_opt.int3c2e_evaluator(
        ao_pair_batch_size=ao_pair_batch_size,
        aux_batch_size=None,
        bas_ij_aggregated=bas_ij_aggregated,
    )
    _aux_offsets = None

    if hasattr(ao_pair_offsets, "get"):
        ao_pair_offsets = ao_pair_offsets.get()
    ao_pair_offsets = numpy.asarray(ao_pair_offsets, dtype=numpy.int64)

    if with_long_range:
        eval_ft, _ao_pair_offsets_ft = ft_opt.ft_evaluator(
            ao_pair_batch_size, bas_ij_aggregated=bas_ij_aggregated
        )
        if not numpy.array_equal(ao_pair_offsets, _ao_pair_offsets_ft):
            raise RuntimeError("AO-pair offsets mismatch between SR and LR evaluators")
        _ao_pair_offsets_ft = None
        # Use complex128 to keep lib.gemm on the cublas GEMM path for LR fit build.
        aux_coeff0_c = cupy.asarray(aux_coeff0, dtype=cupy.complex128, order="C")

        if Gblksize is None:
            mem_free = cupy.cuda.runtime.memGetInfo()[0]
            # Rough memory model for one LR block:
            # auxG_fit_blk(naux_fit,gblk,complex128) + pqG(npair,gblk,complex128)
            elem_bytes = 16
            avail_mem = max(0, mem_free - 512 * 1024 * 1024)
            denom = elem_bytes * max(1, (naux_fit + ao_pair_batch_size))
            Gblksize = int(avail_mem // denom) // 32 * 32
            Gblksize = max(32, min(Gblksize, ngrids))

        g_slices = list(prange(0, ngrids, Gblksize))
        # Precompute LR aux-fit blocks on GPU then store on host memory.
        # This avoids keeping the full auxG_fit tensor on device.
        auxG_fit_blocks_cpu = []
        auxG_fit_buf = cupy.empty((naux_fit, Gblksize), dtype=cupy.complex128, order="C")
        for g0, g1 in g_slices:
            logger.info(
                'eri_high_level_solver_incore, get int3c long range part, g_slices:%d/%d' %
                (g0, ngrids))
            auxG_cart_raw = cupy.asarray(
                # ft_ao.ft_ao(auxcell, Gv[g0:g1], sort_cell=False),
                ft_ao.ft_ao(auxcell, Gv[g0:g1]),
                dtype=cupy.complex128,
                order="C",
            )

            auxG_fit_blk = _fit_aux_ft_block(
                aux_coeff0_c, auxG_cart_raw, auxG_fit_buf, naux_cart
            )
            auxG_fit_blk *= coulG[g0:g1]

            auxG_fit_blk_host = cupyx.empty_pinned(
                auxG_fit_blk.shape, dtype=auxG_fit_blk.dtype, order="C"
            )
            auxG_fit_blk.get(out=auxG_fit_blk_host, blocking=True)
            auxG_fit_blocks_cpu.append(auxG_fit_blk_host)
            auxG_cart_raw = None
            auxG_fit_blk = None
        aux_coeff0_c = None
        coulG = None
        auxG_fit_buf = None
    else:
        eval_ft = None
        g_slices = []
        auxG_fit_blocks_cpu = []
        ngrids = 0
        Gblksize = 0
    bas_ij_aggregated = None

    aux_coeff = cupy.empty_like(aux_coeff0)
    aux_coeff[aux_sorting] = aux_coeff0
    aux_coeff0 = None
    aux_sorting = None

    shl_pair_batches = int(len(ao_pair_offsets) - 1)
    if mo_coeff_i is None or mo_coeff_j is None:
        raise ValueError("mo_coeff_i and mo_coeff_j are required")
    mo_coeff_i = cupy.asarray(mo_coeff_i, dtype=cupy.float64, order="C")
    mo_coeff_j = cupy.asarray(mo_coeff_j, dtype=cupy.float64, order="C")
    if mo_coeff_i.shape[0] != mo_coeff_j.shape[0]:
        raise ValueError("mo_coeff_i and mo_coeff_j must have the same AO dimension")
    nmo_i = int(mo_coeff_i.shape[1])
    nmo_j = int(mo_coeff_j.shape[1])

    nao = int(mo_coeff_i.shape[0])
    pair_address = cupy.asarray(cderi_idx[0], dtype=cupy.int64)
    diag_idx = cupy.asarray(cderi_idx[1], dtype=cupy.int64)
    if pair_address.size > 0:
        max_addr = int(pair_address.max().get())
        if max_addr >= nao * nao:
            raise ValueError(
                f"AO dimension mismatch: max(pair_address)={max_addr}, "
                f"but mo_coeff AO dimension={nao}"
            )

    pair_weight = cupy.ones((pair_address.size,), dtype=cupy.float64)
    if diag_idx.size > 0:
        pair_weight[diag_idx] = 0.5
    diag_idx = None

    if Lblksize is None:
        Kblksize = naux_fit
    else:
        Kblksize = max(1, min(int(Lblksize), naux_fit))

    max_nao_blk = 0
    for p0, p1 in zip(ao_pair_offsets[:-1], ao_pair_offsets[1:]):
        if p1 <= p0:
            continue
        addr = pair_address[p0:p1]
        i_blk = addr // nao
        j_blk = addr % nao
        ij_cat_blk = cupy.concatenate((i_blk, j_blk), axis=0)
        nao_blk_tmp = int(cupy.unique(ij_cat_blk).size)
        if nao_blk_tmp > max_nao_blk:
            max_nao_blk = nao_blk_tmp
        addr = None
        i_blk = None
        j_blk = None
        ij_cat_blk = None

    npair_blk_max = int(numpy.max(ao_pair_offsets[1:] - ao_pair_offsets[:-1]))
    # Accumulate in host memory to reduce device-memory pressure for large MO spaces.
    cderi_mo_cpu = cupyx.zeros_pinned((nmo_i, nmo_j, naux_fit), dtype=numpy.float64, order="C")
    out_pqk_host = cupyx.empty_pinned((Kblksize, nmo_i, nmo_j), dtype=numpy.float64, order="C")
    
    j3c_fit_pair_buf = cupy.empty((naux_fit, npair_blk_max), dtype=cupy.float64, order="C")
    pqG_buf = cupy.empty((npair_blk_max, Gblksize), dtype=cupy.complex128, order="C")
    j3c_raw_buf = cupy.empty(
        (
            max(
                naux_cart*npair_blk_max,
                int(Kblksize*max_nao_blk*max_nao_blk/2)+1,
                int(Kblksize*max_nao_blk*nmo_j/2)+1,
                    ),
            ),
        dtype=cupy.complex128,
        order="C")
    auxG_fit_block_buf = cupy.empty(
        (
            max(
                naux_cart * max(1, Gblksize),
                int(Kblksize * max_nao_blk * max_nao_blk / 2) + 1,
                int(Kblksize * nmo_j * nmo_i / 2) + 1,
            ),
        ),
        dtype=cupy.complex128,
        order="C",
    )

    for ij_batch_id in range(shl_pair_batches):
        logger.info(
            'eri_high_level_solver_incore, get int3c transform, ij_batch_id:%d/%d' %
            (ij_batch_id, shl_pair_batches))
        p0 = int(ao_pair_offsets[ij_batch_id])
        p1 = int(ao_pair_offsets[ij_batch_id + 1])
        if p1 <= p0:
            continue

        addr = pair_address[p0:p1]
        npair_blk = int(p1 - p0)
        i = addr // nao
        j = addr % nao
        ij_cat = cupy.concatenate((i, j), axis=0)
        ao_u, inv = cupy.unique(ij_cat, return_inverse=True)
        inv_i = inv[:npair_blk]
        inv_j = inv[npair_blk:]
        nao_blk = int(ao_u.size)
        local_addr = inv_i.astype(cupy.int64) * nao_blk + inv_j.astype(cupy.int64)
        Ci_blk = mo_coeff_i[ao_u]
        Cj_blk = mo_coeff_j[ao_u]
        w = pair_weight[p0:p1]

        # Build the full fit-space pair block once for each AO-pair batch:
        # J_fit(K,pair) = aux_coeff^T * J_sr_cart + auxG_fit * J_lr_G
        j3c_fit_pair = lib.empty_from_buf(j3c_fit_pair_buf, (naux_fit, npair_blk), dtype=cupy.float64)
        j3c_fit_pair[:] = 0

        # SR contribution: evaluate one full auxiliary batch.
        j3c_raw = eval_j3c(shl_pair_batch_id=ij_batch_id, aux_batch_id=0, out=j3c_raw_buf)
        if j3c_raw.dtype != cupy.float64:
            raise TypeError(f"Unexpected j3c_raw dtype: {j3c_raw.dtype}, expected float64")
        if j3c_raw.size > 0:
            j3c_cart = j3c_raw[:, :, 0]
            lib.gemm(aux_coeff, j3c_cart, c=j3c_fit_pair, transa='T', transb='T')
            j3c_cart = None
        j3c_raw = None

        # LR contribution: accumulate in G-blocks to control memory usage.
        if with_long_range:
            for iblk, (g0, g1) in enumerate(g_slices):
                auxG_fit_blk = lib.empty_from_buf(auxG_fit_block_buf, auxG_fit_blocks_cpu[iblk].shape, dtype=cupy.complex128)
                auxG_fit_blk.set(auxG_fit_blocks_cpu[iblk])
                # auxG_fit_blk = cupy.asarray(auxG_fit_blocks_cpu[iblk])
                pqG = eval_ft(Gv[g0:g1], ij_batch_id, out=pqG_buf)
                j3c_fit_pair_lr = lib.gemm(auxG_fit_blk, pqG, buf=j3c_raw_buf, transb='T')
                j3c_fit_pair += j3c_fit_pair_lr.real
                pqG = None
                auxG_fit_blk = None
                j3c_fit_pair_lr = None

        # AO2MO in K-blocks to limit GPU memory footprint.
        for K0 in range(0, naux_fit, Kblksize):
            K1 = min(naux_fit, K0 + Kblksize)
            Kblk = int(K1 - K0)
            j3c_pair_blk = lib.empty_from_buf(j3c_raw_buf, (Kblk, npair_blk), dtype=cupy.float64)
            j3c_pair_blk[:] = j3c_fit_pair[K0:K1]
            j3c_pair_blk *= w[None, :]
            j3c_tril = lib.empty_from_buf(auxG_fit_block_buf, (Kblk, nao_blk * nao_blk), dtype=cupy.float64)
            j3c_tril[:] = 0
            j3c_tril[:, local_addr] = j3c_pair_blk
            j3c_tril = j3c_tril.reshape(Kblk, nao_blk, nao_blk)
            j3c_tril_trans = lib.empty_from_buf(j3c_raw_buf, (Kblk, nao_blk, nao_blk), dtype=cupy.float64)
            j3c_tril_trans[:] = j3c_tril.transpose(0, 2, 1)
            j3c_tril += j3c_tril_trans

            # Two-step AO2MO: Kij,ip->Kpj; Kpj,jq->Kpq
            # Kij,jq->qKi
            tmp = lib.gemm(Ci_blk, j3c_tril.reshape(-1, nao_blk), buf=j3c_raw_buf, transa='T', transb='T')
            # qKi,ip->pqK
            out_pqk = lib.gemm(Cj_blk, tmp.reshape(-1, nao_blk), buf=auxG_fit_block_buf, transa='T', transb='T')
            out_pqk = out_pqk.reshape(nmo_j, nmo_i, Kblk)
            out_qpk = lib.empty_from_buf(j3c_raw_buf, (nmo_i, nmo_j, Kblk), 'f8')
            out_qpk[:] = out_pqk.transpose(1, 0, 2)
            out_qpk_tmp = lib.empty_from_buf(out_pqk_host, out_qpk.shape, 'f8')
            out_qpk.get(out=out_qpk_tmp, blocking=True)
            cderi_mo_cpu[:, :,K0:K1] += out_qpk_tmp

            j3c_pair_blk = None
            j3c_tril = None
            j3c_tril_trans = None
            tmp = None
            out_pqk = None
            out_qpk = None
            out_qpk_tmp = None

        j3c_fit_pair = None
        addr = None
        i = None
        j = None
        ij_cat = None
        ao_u = None
        inv = None
        inv_i = None
        inv_j = None
        local_addr = None
        Ci_blk = None
        Cj_blk = None
        w = None

    j3c_fit_pair_buf = None
    pqG_buf = None
    j3c_raw_buf = None
    auxG_fit_block_buf = None
    auxG_fit_blocks_cpu = None
    pair_weight = None
    pair_address = None
    mo_coeff_i = None
    mo_coeff_j = None
    aux_coeff = None
    eval_j3c = None
    eval_ft = None
    Gv = None
    ft_opt = None
    int3c2e_opt = None


    cupy.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    gc.collect()    

    LL_svd = cupy.zeros((naux_fit, naux_fit), dtype='f8', order='F')
    nov_tot = nmo_i * nmo_j
    if nov_tot <= 0:
        raise ValueError(f"Invalid MO dimensions: nmo_i={nmo_i}, nmo_j={nmo_j}")
    avail_gpu_memory = lib.gpu_avail_bytes() / 8
    slice_len_ov = int(avail_gpu_memory / naux_fit)
    slice_len_ov = max(1, min(slice_len_ov, nov_tot))

    ovslice_list = [
        slice(
            i[0],
            i[1]) for i in prange(
            0,
            nov_tot,
            slice_len_ov)]
    
    cderi_mo_cpu = cderi_mo_cpu.reshape(-1, naux_fit)
    buffer_ovL = cupy.empty((slice_len_ov, naux_fit), dtype='f8')

    for sov in ovslice_list:
        sov_len = sov.stop - sov.start
        sov_L = lib.empty_from_buf(buffer_ovL, (sov_len, naux_fit), 'f8')
        sov_L.set(cderi_mo_cpu[sov])
        lib.gemm(sov_L, sov_L, transa='T', c=LL_svd, beta=1.0)

    buffer_ovL = sov_L = None

    cupy.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    gc.collect()

    S, U_svd = cupy.linalg._eigenvalue._syevd(
        LL_svd, 'L', with_eigen_vector=True, overwrite_a=True)
    S = _sqrt_psd_eigs_or_raise(S, min_eig_allowed=-1e-10)
    sort_ind = S.argsort()[::-1]
    S = S[sort_ind]
    U_svd = U_svd[:, sort_ind]
    newind = cupy.where(S > svd_tol)[0]
    if newind.size == 0:
        raise RuntimeError(
            f"No singular values above svd_tol={svd_tol}; max singular value={float(S.max())}"
        )
    naux_cut = newind.size
    U_svd = cupy.ascontiguousarray(U_svd[:, newind])
    L_cd = lib.gemm(U_svd, U_svd, transa='T', transb='N')
    lib.linalg.cholesky(L_cd, overwrite=True)
    U_svd = lib.gemm(U_svd, L_cd, transa='N', transb='N')
    L_cd = S = LL_svd = None

    logger.info(f'----- SVD cut aux basis size from : {naux_fit} to : {naux_cut}')

    avail_gpu_memory = lib.gpu_avail_bytes() / 8
    slice_len_ov = int(avail_gpu_memory / (naux_cut + naux_fit))
    slice_len_ov = max(1, min(slice_len_ov, nov_tot))
    logger.info('----- The slice_len_ov for SVD cut: %s' % slice_len_ov)
    sov_list = [
        slice(
            i[0],
            i[1]) for i in prange(
            0,
            nov_tot,
            slice_len_ov)]

    if solver_type == 'MP2':
        cderi_cut = cupyx.empty_pinned((nov_tot, naux_cut), dtype='f8')
    elif 'CC' in solver_type:
        cderi_cut = cupyx.empty_pinned((naux_cut, nov_tot), dtype='f8')
    else:
        raise ValueError(f"Unsupported solver_type: {solver_type}")

    buff_ovL = cupy.empty((naux_fit * slice_len_ov), dtype='f8')
    buff_cderi_cut = cupy.empty((naux_cut * slice_len_ov), dtype='f8')
    buff_cderi_cut_h = None
    if 'CC' in solver_type:
        buff_cderi_cut_h = cupyx.empty_pinned(buff_cderi_cut.shape, dtype='f8')

    ovs_L = None
    cderi_cut_s = None
    cderi_cut_s_h = None
    for sov_ind, sov in enumerate(sov_list):
        logger.info(
            'eri_high_level_solver_incore, SVD cut, for cderi_cut, sov:%d/%d' %
            (sov_ind + 1, len(sov_list)))
        nov = sov.stop - sov.start
        ovs_L = lib.empty_from_buf(buff_ovL, (nov, naux_fit), 'f8')
        ovs_L.set(cderi_mo_cpu[sov])

        if solver_type == 'MP2':
            cupy.cuda.get_current_stream().synchronize()
            cderi_cut_s = lib.gemm(
                ovs_L,
                U_svd,
                buf=buff_cderi_cut,
                transa='N',
                transb='N')
            cderi_cut_s.get(out=cderi_cut[sov], blocking=True)
        elif 'CC' in solver_type:
            cderi_cut_s = lib.gemm(
                U_svd,
                ovs_L,
                buf=buff_cderi_cut,
                transa='T',
                transb='T')  # may cause problem
            cderi_cut_s_h = lib.empty_from_buf(
                buff_cderi_cut_h, cderi_cut_s.shape, 'f8')
            cderi_cut_s.get(out=cderi_cut_s_h, blocking=True)
            cderi_cut[:, sov] = cderi_cut_s_h
        cderi_cut_s = None
        cderi_cut_s_h = None

    cupy.cuda.get_current_stream().synchronize()
    U_svd = buff_ovL = buff_cderi_cut = buff_cderi_cut_h = cderi_mo_cpu = ovs_L = cderi_cut_s = cderi_cut_s_h = None
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
    **kwargs,
):
    _validate_gamma_kpts(kpts)
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
    
    cupy.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    gc.collect()
    
    if rdm1_core_coeff is None:
        raise ValueError("rdm1_core_coeff is required")
    rdm1_core_coeff = cupy.asarray(rdm1_core_coeff, dtype=cupy.float64, order="C")
    if rdm1_core_coeff.ndim != 2:
        raise ValueError("rdm1_core_coeff must be a 2D array with shape (nao, ncore)")
    ncore = int(rdm1_core_coeff.shape[1])

    if omega is None:
        omega = OMEGA_MIN

    kmesh = numpy.array([1, 1, 1])

    int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega=-omega, bvk_kmesh=kmesh).build()
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell

    cd_j2c_cache, _negative_metric_size = _precontract_j2c_aux_coeff(
        auxcell, None, omega, with_long_range, linear_dep_threshold
    )
    aux_coeff0 = _to_real_or_raise(cd_j2c_cache[0], "cd_j2c_cache[0]")
    naux_cart, naux_fit = aux_coeff0.shape

    cderi_idx = int3c2e_opt.pair_and_diag_indices()
    if with_long_range:
        omega_lr = abs(int3c2e_opt.omega)
        ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega_lr)
        mesh = cell.cutoff_to_mesh(ke_cutoff)
        mesh = cell.symmetrize_mesh(mesh)
        Gv, _Gvbase, kws = cell.get_Gv_weights(mesh)
        ngrids = len(Gv)
        coulG = _to_real_or_raise(_weighted_coulG_LR(auxcell, Gv, omega_lr, kws), "coulG")

        ft_opt = ft_ao.FTOpt(cell, kmesh)
        ft_opt.__dict__.update(int3c2e_opt.__dict__)
        ft_opt._aft_envs = int3c2e_opt._int3c2e_envs
    else:
        Gv = None
        ft_opt = None

    nsp_per_block = ft_ao.ft_ao_scheme()[0]
    bas_ij_aggregated = cell.aggregate_shl_pairs(int3c2e_opt.bas_ij_cache, nsp_per_block)

    # Use a single full auxiliary batch for SR part (no aux slicing).
    eval_j3c, aux_sorting, ao_pair_offsets, _aux_offsets = int3c2e_opt.int3c2e_evaluator(
        ao_pair_batch_size=ao_pair_batch_size,
        aux_batch_size=None,
        bas_ij_aggregated=bas_ij_aggregated,
    )
    _aux_offsets = None

    if hasattr(ao_pair_offsets, "get"):
        ao_pair_offsets = ao_pair_offsets.get()
    ao_pair_offsets = numpy.asarray(ao_pair_offsets, dtype=numpy.int64)

    if with_long_range:
        eval_ft, _ao_pair_offsets_ft = ft_opt.ft_evaluator(
            ao_pair_batch_size, bas_ij_aggregated=bas_ij_aggregated
        )
        if not numpy.array_equal(ao_pair_offsets, _ao_pair_offsets_ft):
            raise RuntimeError("AO-pair offsets mismatch between SR and LR evaluators")
        _ao_pair_offsets_ft = None
        # Use complex128 to keep lib.gemm on the cublas GEMM path for LR fit build.
        aux_coeff0_c = cupy.asarray(aux_coeff0, dtype=cupy.complex128, order="C")

        if Gblksize is None:
            mem_free = cupy.cuda.runtime.memGetInfo()[0]
            # Rough memory model for one LR block:
            # auxG_fit_blk(naux_fit,gblk,complex128) + pqG(npair,gblk,complex128)
            elem_bytes = 16
            avail_mem = max(0, mem_free - 512 * 1024 * 1024)
            denom = elem_bytes * max(1, (naux_fit + ao_pair_batch_size))
            Gblksize = int(avail_mem // denom) // 32 * 32
            Gblksize = max(32, min(Gblksize, ngrids))

        g_slices = list(prange(0, ngrids, Gblksize))
        # Precompute LR aux-fit blocks on GPU then store on host memory.
        # This avoids keeping the full auxG_fit tensor on device.
        auxG_fit_blocks_cpu = []
        auxG_fit_buf = cupy.empty((naux_fit, Gblksize), dtype=cupy.complex128, order="C")
        for g0, g1 in g_slices:
            logger.info(
                'eri_high_level_solver_incore, get int3c long range part, g_slices:%d/%d' %
                (g0, ngrids))
            auxG_cart_raw = cupy.asarray(
                # ft_ao.ft_ao(auxcell, Gv[g0:g1], sort_cell=False),
                ft_ao.ft_ao(auxcell, Gv[g0:g1]),
                dtype=cupy.complex128,
                order="C",
            )

            auxG_fit_blk = _fit_aux_ft_block(
                aux_coeff0_c, auxG_cart_raw, auxG_fit_buf, naux_cart
            )
            auxG_fit_blk *= coulG[g0:g1]

            auxG_fit_blk_host = cupyx.empty_pinned(
                auxG_fit_blk.shape, dtype=auxG_fit_blk.dtype, order="C"
            )
            auxG_fit_blk.get(out=auxG_fit_blk_host, blocking=True)
            auxG_fit_blocks_cpu.append(auxG_fit_blk_host)
            auxG_cart_raw = None
            auxG_fit_blk = None
        aux_coeff0_c = None
        coulG = None
        auxG_fit_buf = None
    else:
        eval_ft = None
        g_slices = []
        auxG_fit_blocks_cpu = []
        ngrids = 0
        Gblksize = 0
    bas_ij_aggregated = None

    aux_coeff = cupy.empty_like(aux_coeff0)
    aux_coeff[aux_sorting] = aux_coeff0
    aux_coeff0 = None
    aux_sorting = None

    shl_pair_batches = int(len(ao_pair_offsets) - 1)
    if mo_coeff is None:
        raise ValueError("mo_coeff is required")
    mo_coeff = cupy.asarray(mo_coeff, dtype=cupy.float64, order="C")
    if rdm1_core_coeff.shape[0] != mo_coeff.shape[0]:
        raise ValueError(
            "rdm1_core_coeff must have the same AO dimension as mo_coeff"
        )
    nmo = int(mo_coeff.shape[1])

    nao = int(mo_coeff.shape[0])
    pair_address = cupy.asarray(cderi_idx[0], dtype=cupy.int64)
    diag_idx = cupy.asarray(cderi_idx[1], dtype=cupy.int64)
    if pair_address.size > 0:
        max_addr = int(pair_address.max().get())
        if max_addr >= nao * nao:
            raise ValueError(
                f"AO dimension mismatch: max(pair_address)={max_addr}, "
                f"but mo_coeff AO dimension={nao}"
            )

    pair_weight = cupy.ones((pair_address.size,), dtype=cupy.float64)
    if diag_idx.size > 0:
        pair_weight[diag_idx] = 0.5
    diag_idx = None

    if Lblksize is None:
        Kblksize = naux_fit
    else:
        Kblksize = max(1, min(int(Lblksize), naux_fit))

    max_nao_blk = 0
    for p0, p1 in zip(ao_pair_offsets[:-1], ao_pair_offsets[1:]):
        if p1 <= p0:
            continue
        addr = pair_address[p0:p1]
        i_blk = addr // nao
        j_blk = addr % nao
        ij_cat_blk = cupy.concatenate((i_blk, j_blk), axis=0)
        nao_blk_tmp = int(cupy.unique(ij_cat_blk).size)
        if nao_blk_tmp > max_nao_blk:
            max_nao_blk = nao_blk_tmp
        addr = None
        i_blk = None
        j_blk = None
        ij_cat_blk = None

    npair_blk_max = int(numpy.max(ao_pair_offsets[1:] - ao_pair_offsets[:-1]))
    # Accumulate in host memory to reduce device-memory pressure for large MO spaces.
    cderi_mo_cpu = cupyx.zeros_pinned((nmo, nmo, naux_fit), dtype=numpy.float64, order="C")
    out_pqk_host = cupyx.empty_pinned((Kblksize, nmo*max(nmo, ncore)), dtype=numpy.float64, order="C")
    
    j3c_fit_pair_buf = cupy.empty((naux_fit, npair_blk_max), dtype=cupy.float64, order="C")
    pqG_buf = cupy.empty((npair_blk_max, Gblksize), dtype=cupy.complex128, order="C")

    # eri_vj_cpu = cupyx.zeros_pinned((naux_fit, ncore, ncore), dtype=numpy.float64, order="C")
    eri_vj = cupy.zeros((naux_fit, 1), dtype=cupy.float64, order="C")
    eri_vk_cpu = cupyx.zeros_pinned((naux_fit, ncore, nmo), dtype=numpy.float64, order="C")

    j3c_raw_buf = cupy.empty(
        (
            max(
                naux_cart*npair_blk_max,
                int(Kblksize*max_nao_blk*max_nao_blk/2)+1,
                int(Kblksize*max_nao_blk*nmo/2)+1,
                int(Kblksize*nmo*ncore/2)+1,
                    ),
            ),
        dtype=cupy.complex128,
        order="C")

    auxG_fit_block_buf = cupy.empty(
        (
            max(
                naux_cart * max(1, Gblksize),
                int(Kblksize * nmo * nmo / 2) + 1,
                int(Kblksize * nmo * ncore / 2) + 1,
            ),
        ),
        dtype=cupy.complex128,
        order="C",
    )

    j3c_tril_buf = cupy.empty(
        (Kblksize*max_nao_blk*max_nao_blk),
        dtype=cupy.float64,
        order="C")

    for ij_batch_id in range(shl_pair_batches):
        logger.info(
            'eri_high_level_solver_incore, get int3c transform, ij_batch_id:%d/%d' %
            (ij_batch_id, shl_pair_batches))
        p0 = int(ao_pair_offsets[ij_batch_id])
        p1 = int(ao_pair_offsets[ij_batch_id + 1])
        if p1 <= p0:
            continue

        addr = pair_address[p0:p1]
        npair_blk = int(p1 - p0)
        i = addr // nao
        j = addr % nao
        ij_cat = cupy.concatenate((i, j), axis=0)
        ao_u, inv = cupy.unique(ij_cat, return_inverse=True)
        inv_i = inv[:npair_blk]
        inv_j = inv[npair_blk:]
        nao_blk = int(ao_u.size)
        local_addr = inv_i.astype(cupy.int64) * nao_blk + inv_j.astype(cupy.int64)
        C_blk = mo_coeff[ao_u]
        rdm1_core_coeff_blk = rdm1_core_coeff[ao_u]
        rdm1_core_blk = lib.gemm(rdm1_core_coeff_blk, rdm1_core_coeff_blk, transb='T')
        w = pair_weight[p0:p1]

        # Build the full fit-space pair block once for each AO-pair batch:
        # J_fit(K,pair) = aux_coeff^T * J_sr_cart + auxG_fit * J_lr_G
        j3c_fit_pair = lib.empty_from_buf(j3c_fit_pair_buf, (naux_fit, npair_blk), dtype=cupy.float64)
        j3c_fit_pair[:] = 0

        # SR contribution: evaluate one full auxiliary batch.
        j3c_raw = eval_j3c(shl_pair_batch_id=ij_batch_id, aux_batch_id=0, out=j3c_raw_buf)
        if j3c_raw.dtype != cupy.float64:
            raise TypeError(f"Unexpected j3c_raw dtype: {j3c_raw.dtype}, expected float64")
        if j3c_raw.size > 0:
            j3c_cart = j3c_raw[:, :, 0]
            lib.gemm(aux_coeff, j3c_cart, c=j3c_fit_pair, transa='T', transb='T')
            j3c_cart = None
        j3c_raw = None

        # LR contribution: accumulate in G-blocks to control memory usage.
        if with_long_range:
            for iblk, (g0, g1) in enumerate(g_slices):
                auxG_fit_blk = lib.empty_from_buf(auxG_fit_block_buf, auxG_fit_blocks_cpu[iblk].shape, dtype=cupy.complex128)
                auxG_fit_blk.set(auxG_fit_blocks_cpu[iblk])
                # auxG_fit_blk = cupy.asarray(auxG_fit_blocks_cpu[iblk])
                pqG = eval_ft(Gv[g0:g1], ij_batch_id, out=pqG_buf)
                j3c_fit_pair_lr = lib.gemm(auxG_fit_blk, pqG, buf=j3c_raw_buf, transb='T')
                j3c_fit_pair += j3c_fit_pair_lr.real
                pqG = None
                auxG_fit_blk = None
                j3c_fit_pair_lr = None

        # AO2MO in K-blocks to limit GPU memory footprint.
        for K0 in range(0, naux_fit, Kblksize):
            K1 = min(naux_fit, K0 + Kblksize)
            Kblk = int(K1 - K0)
            j3c_pair_blk = lib.empty_from_buf(j3c_raw_buf, (Kblk, npair_blk), dtype=cupy.float64)
            j3c_pair_blk[:] = j3c_fit_pair[K0:K1]
            j3c_pair_blk *= w[None, :]
            j3c_tril = lib.empty_from_buf(j3c_tril_buf, (Kblk, nao_blk * nao_blk), dtype=cupy.float64)
            j3c_tril[:] = 0
            j3c_tril[:, local_addr] = j3c_pair_blk
            j3c_tril = j3c_tril.reshape(Kblk, nao_blk, nao_blk)
            j3c_tril_trans = lib.empty_from_buf(j3c_raw_buf, (Kblk, nao_blk, nao_blk), dtype=cupy.float64)
            j3c_tril_trans[:] = j3c_tril.transpose(0, 2, 1)
            j3c_tril += j3c_tril_trans

            rdm1_core_vec = rdm1_core_blk.reshape(-1, 1)
            lib.gemm(
                j3c_tril.reshape(Kblk, -1),
                rdm1_core_vec,
                c=eri_vj[K0:K1],
                beta=1.0,
            )

            # Two-step AO2MO: Kij,ip->Kpj; Kpj,jq->Kpq
            tmp = lib.gemm(C_blk, j3c_tril.reshape(-1, nao_blk), buf=j3c_raw_buf, transa='T', transb='T')
            out_pqk = lib.gemm(C_blk, tmp.reshape(-1, nao_blk), buf=auxG_fit_block_buf, transa='T', transb='T')
            out_pqk = out_pqk.reshape(nmo, nmo, Kblk)
            out_pqk_tmp = lib.empty_from_buf(out_pqk_host, out_pqk.shape, 'f8')
            out_pqk.get(out=out_pqk_tmp, blocking=True)
            cderi_mo_cpu[:, :,K0:K1] += out_pqk_tmp

            out_pqk = lib.gemm(rdm1_core_coeff_blk, tmp.reshape(-1, nao_blk), buf=auxG_fit_block_buf, transa='T', transb='T')
            out_pqk = out_pqk.reshape(ncore, nmo, Kblk)
            out_kpq = lib.empty_from_buf(j3c_raw_buf, (Kblk, ncore, nmo), 'f8')
            out_kpq[:] = out_pqk.transpose(2, 0, 1)
            out_kpq_tmp = lib.empty_from_buf(out_pqk_host, out_kpq.shape, 'f8')
            out_kpq.get(out=out_kpq_tmp, blocking=True)
            eri_vk_cpu[K0:K1] += out_kpq_tmp

            j3c_pair_blk = None
            j3c_tril = None
            j3c_tril_trans = None
            tmp = None
            out_pqk = None
            out_pqk_tmp = None
            out_kpq = None
            out_kpq_tmp = None
            rdm1_core_vec = None


        j3c_fit_pair = None
        addr = None
        i = None
        j = None
        ij_cat = None
        ao_u = None
        inv = None
        inv_i = None
        inv_j = None
        local_addr = None
        C_blk = None
        rdm1_core_coeff_blk = None
        rdm1_core_blk = None
        w = None

    j3c_fit_pair_buf = None
    pqG_buf = None
    j3c_raw_buf = None
    j3c_tril_buf = None
    auxG_fit_block_buf = None
    auxG_fit_blocks_cpu = None
    pair_weight = None
    pair_address = None
    mo_coeff = None
    aux_coeff = None
    cderi_idx = None
    ao_pair_offsets = None
    out_pqk_host = None
    rdm1_core_coeff = None
    eval_j3c = None
    eval_ft = None
    Gv = None
    ft_opt = None
    int3c2e_opt = None

    cupy.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    gc.collect()

    eri_buf = cupy.empty(
        (
            max(
                Kblksize*nmo*nmo,
                Kblksize*nmo*ncore,
                )
            ),
        dtype='f8',
        order='C')


    cderi_mo_host_buf = cupyx.empty_pinned((Kblksize, nmo, nmo), dtype=numpy.float64, order="C")

    vj = cupy.zeros((nmo * nmo, 1), dtype=cupy.float64)
    vk = cupy.zeros((nmo, nmo), dtype=cupy.float64)

    for K0 in range(0, naux_fit, Kblksize):
        K1 = min(naux_fit, K0 + Kblksize)
        Kblk = int(K1 - K0)
        cderi_mo_host_tmp = lib.empty_from_buf(cderi_mo_host_buf, (nmo, nmo, Kblk), 'f8')
        cderi_mo_host_tmp[:] = cderi_mo_cpu[:, :,K0:K1]
        cderi_mo_blk = lib.empty_from_buf(eri_buf, (nmo, nmo, Kblk), 'f8')
        cderi_mo_blk.set(cderi_mo_host_tmp)

        lib.gemm(cderi_mo_blk.reshape(-1, Kblk), eri_vj[K0:K1], c=vj, beta=1.0)

        eri_vk_blk = lib.empty_from_buf(eri_buf, (Kblk, ncore, nmo), 'f8')
        eri_vk_blk.set(eri_vk_cpu[K0:K1])

        lib.gemm(eri_vk_blk.reshape(-1, nmo), eri_vk_blk.reshape(-1, nmo), c=vk, beta=1.0, transa='T')
        cupy.cuda.get_current_stream().synchronize()

    eri_buf = None
    cderi_mo_host_buf = None
    cderi_mo_host_tmp = None
    cderi_mo_blk = None
    eri_vj = None
    eri_vk_blk = None
    eri_vk_cpu = None

    vj = vj.get(blocking=True).reshape(nmo, nmo)
    vk = vk.get(blocking=True)


    cupy.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    gc.collect()    

    LL_svd = cupy.zeros((naux_fit, naux_fit), dtype='f8', order='F')
    nov_tot = nmo * nmo
    if nov_tot <= 0:
        raise ValueError(f"Invalid MO dimensions: nmo={nmo}")
    avail_gpu_memory = lib.gpu_avail_bytes() / 8
    slice_len_ov = int(avail_gpu_memory / naux_fit)
    slice_len_ov = max(1, min(slice_len_ov, nov_tot))

    ovslice_list = [
        slice(
            i[0],
            i[1]) for i in prange(
            0,
            nov_tot,
            slice_len_ov)]
    
    cderi_mo_cpu = cderi_mo_cpu.reshape(-1, naux_fit)
    buffer_ovL = cupy.empty((slice_len_ov, naux_fit), dtype='f8')

    for sov in ovslice_list:
        sov_len = sov.stop - sov.start
        sov_L = lib.empty_from_buf(buffer_ovL, (sov_len, naux_fit), 'f8')
        sov_L.set(cderi_mo_cpu[sov])
        lib.gemm(sov_L, sov_L, transa='T', c=LL_svd, beta=1.0)

    buffer_ovL = sov_L = None
    cupy.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    gc.collect()

    S, U_svd = cupy.linalg._eigenvalue._syevd(
        LL_svd, 'L', with_eigen_vector=True, overwrite_a=True)
    S = _sqrt_psd_eigs_or_raise(S, min_eig_allowed=-1e-10)
    sort_ind = S.argsort()[::-1]
    S = S[sort_ind]
    U_svd = U_svd[:, sort_ind]
    newind = cupy.where(S > svd_tol)[0]
    if newind.size == 0:
        raise RuntimeError(
            f"No singular values above svd_tol={svd_tol}; max singular value={float(S.max())}"
        )
    naux_cut = newind.size
    U_svd = cupy.ascontiguousarray(U_svd[:, newind])
    L_cd = lib.gemm(U_svd, U_svd, transa='T', transb='N')
    lib.linalg.cholesky(L_cd, overwrite=True)
    U_svd = lib.gemm(U_svd, L_cd, transa='N', transb='N')
    L_cd = S = LL_svd = None

    logger.info(f'----- SVD cut aux basis size from : {naux_fit} to : {naux_cut}')

    avail_gpu_memory = lib.gpu_avail_bytes() / 8
    slice_len_ov = int(avail_gpu_memory / (naux_cut + naux_fit))
    slice_len_ov = max(1, min(slice_len_ov, nov_tot))
    logger.info('----- The slice_len_ov for SVD cut: %s' % slice_len_ov)
    sov_list = [
        slice(
            i[0],
            i[1]) for i in prange(
            0,
            nov_tot,
            slice_len_ov)]

    cderi_cut = cupyx.empty_pinned((naux_cut, nov_tot), dtype='f8')

    buff_ovL = cupy.empty((naux_fit * slice_len_ov), dtype='f8')
    buff_cderi_cut = cupy.empty((naux_cut * slice_len_ov), dtype='f8')
    buff_cderi_cut_h = None
    buff_cderi_cut_h = cupyx.empty_pinned(buff_cderi_cut.shape, dtype='f8')

    ovs_L = None
    cderi_cut_s = None
    cderi_cut_s_h = None
    for sov_ind, sov in enumerate(sov_list):
        logger.info(
            'eri_high_level_solver_incore, SVD cut, for cderi_cut, sov:%d/%d' %
            (sov_ind + 1, len(sov_list)))
        nov = sov.stop - sov.start
        ovs_L = lib.empty_from_buf(buff_ovL, (nov, naux_fit), 'f8')
        ovs_L.set(cderi_mo_cpu[sov])

        cderi_cut_s = lib.gemm(
            U_svd,
            ovs_L,
            buf=buff_cderi_cut,
            transa='T',
            transb='T')  # may cause problem
        cderi_cut_s_h = lib.empty_from_buf(
            buff_cderi_cut_h, cderi_cut_s.shape, 'f8')
        cderi_cut_s.get(out=cderi_cut_s_h, blocking=True)
        cderi_cut[:, sov] = cderi_cut_s_h
        cderi_cut_s = None
        cderi_cut_s_h = None

    cupy.cuda.get_current_stream().synchronize()
    U_svd = buff_ovL = buff_cderi_cut = buff_cderi_cut_h = cderi_mo_cpu = ovs_L = cderi_cut_s = cderi_cut_s_h = None
    lib.free_all_blocks()
    gc.collect()

    return cderi_cut, vj, vk
