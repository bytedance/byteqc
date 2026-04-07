import sys
import os
import numpy
import cupy
import cupyx
import gc
from byteqc import lib
from byteqc.lib import Mg
from pyscf.lib import prange, param, logger
from gpu4pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc.df.int3c2e import SRInt3c2eOpt
from gpu4pyscf.pbc.df.rsdf_builder import (
    _precontract_j2c_aux_coeff,
    estimate_ke_cutoff_for_omega,
    _weighted_coulG_LR,
    OMEGA_MIN,
    LINEAR_DEP_THR,
)
import tempfile
from multiprocessing import Pool


cupy.cuda.set_pinned_memory_allocator(None)

AO_PAIR_BATCH_SIZE_1 = 512 * 256
KBLKSIZE_1 = 16
GBLKSIZE_1 = 1024 * 4

# AO_PAIR_BATCH_SIZE_1 = 512
# KBLKSIZE_1 = 16
# GBLKSIZE_1 = 128

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

def cderi_ovL_gamma_point_outcore_gpu4pyscf(
    cell,
    auxcell,
    mo_coeff_i,
    mo_coeff_j,
    oblk,
    vblk,
    log=None,
    path=None,
    omega=None,
    linear_dep_threshold=LINEAR_DEP_THR,
    with_long_range=True,
    ao_pair_batch_size=AO_PAIR_BATCH_SIZE_1,
    Lblksize=KBLKSIZE_1,
    Gblksize=GBLKSIZE_1,
):
    
    cupy.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    gc.collect()

    if path is None:
        path = tempfile.NamedTemporaryFile(dir=param.TMPDIR).name
    if not os.path.exists(path):
        os.mkdir(path)
    
    if log is None:
        log = log.new_logger(cell, cell.verbose)

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
            log.info(
                'eri_high_level_solver_incore, get int3c long range part, g_slices:%d/%d' %
                (g0, ngrids))
            auxG_cart_raw = cupy.asarray(
                # ft_ao.ft_ao(auxcell, Gv[g0:g1], sort_cell=False),
                ft_ao.ft_ao(auxcell, Gv[g0:g1]),
                dtype=cupy.complex128,
                order="C",
            )

            auxG_fit_blk = lib.gemm(
                aux_coeff0_c, auxG_cart_raw, buf=auxG_fit_buf, transa='T', transb='H'
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

    file = lib.FileMp(path + '/eris.dat', 'w')
    cderi_file = file.create_dataset('cderi', (nmo_i, nmo_j, naux_fit), 'f8',
                                blksizes=(oblk, vblk, Kblksize))

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
    read_host = cupyx.empty_pinned((Kblksize, nmo_i, nmo_j), dtype=numpy.float64, order="C")
    write_host = cupyx.empty_pinned((Kblksize, nmo_i, nmo_j), dtype=numpy.float64, order="C")
    
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

    pool_read = Pool(processes=int(lib.NumFileProcess / 2))
    pool_write = Pool(processes=int(lib.NumFileProcess / 2))

    for ij_batch_id in range(shl_pair_batches):
        log.info(
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

        waits_write = None
        # AO2MO in K-blocks to limit GPU memory footprint.
        for K0 in range(0, naux_fit, Kblksize):
            
            K1 = min(naux_fit, K0 + Kblksize)
            Kblk = int(K1 - K0)
            
            cupy.cuda.get_current_stream().synchronize()
            out_read_host = lib.empty_from_buf(read_host, (nmo_i, nmo_j, Kblk), 'f8')
            waits_read = cderi_file.getitem(numpy.s_[:, :, K0:K1], pool=pool_read, buf=out_read_host)

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
            tmp = lib.contraction('Lij', j3c_tril, 'io', Ci_blk, 'Loj', buf=j3c_raw_buf)
            # qKi,ip->pqK
            for p in waits_read:
                p.wait()
            out_tmp = lib.empty_from_buf(auxG_fit_block_buf, (nmo_i, nmo_j, Kblk), 'f8')
            out_tmp.set(out_read_host)
            lib.contraction('Loj', tmp, 'jv', Cj_blk, 'ovL', out_tmp, beta=1.0)

            if waits_write is not None:
                for p in waits_write:
                    p.wait()
            out_write_host = lib.empty_from_buf(write_host, (nmo_i, nmo_j, Kblk), 'f8')
            out_tmp.get(out=out_write_host, blocking=True)
            waits_write =cderi_file.setitem(numpy.s_[:, :, K0:K1], out_write_host, pool=pool_write)

            j3c_pair_blk = None
            j3c_tril = None
            j3c_tril_trans = None
            tmp = None
            out_tmp = None

        for p in waits_write:
            p.wait()

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

    pool_read.terminate()
    pool_read.join()
    pool_write.terminate()
    pool_write.join()


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
    oslices = [slice(*i) for i in prange(0, nmo_i, oblk)]
    return path, oslices


def cderi_ovL_gamma_point_outcore_gpu4pyscf_Mg(
    cell,
    auxcell,
    mo_coeff_i,
    mo_coeff_j,
    oblk,
    vblk,
    log=None,
    path=None,
    omega=None,
    linear_dep_threshold=LINEAR_DEP_THR,
    with_long_range=True,
    ao_pair_batch_size=AO_PAIR_BATCH_SIZE_1,
    Lblksize=KBLKSIZE_1,
    Gblksize=GBLKSIZE_1,
):
    
    Mg.mapgpu(lambda: cupy.cuda.get_current_stream().synchronize())
    Mg.mapgpu(lambda: lib.free_all_blocks())
    gc.collect()

    if path is None:
        path = tempfile.NamedTemporaryFile(dir=param.TMPDIR).name
    if not os.path.exists(path):
        os.mkdir(path)
    
    if log is None:
        log = log.new_logger(cell, cell.verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    if omega is None:
        omega = OMEGA_MIN

    kmesh = numpy.array([1, 1, 1])
    ngpu = Mg.ngpu
    
    int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega=-omega, bvk_kmesh=kmesh).build()
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell

    bas_ij_cache = int3c2e_opt.bas_ij_cache
    for key in bas_ij_cache.keys():
        bas_ij_cache[key] = bas_ij_cache[key].get()
    int3c2e_opt.bas_ij_cache = bas_ij_cache

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
    bas_ij_aggregated_h = []
    for tmp_i in bas_ij_aggregated:
        bas_ij_aggregated_h.append(tmp_i.get())
    # Use a single full auxiliary batch for SR part (no aux slicing).
    # eval_j3c, aux_sorting, ao_pair_offsets, _aux_offsets = int3c2e_opt.int3c2e_evaluator(
    #     ao_pair_batch_size=ao_pair_batch_size,
    #     aux_batch_size=None,
    #     bas_ij_aggregated=bas_ij_aggregated,
    # )
    tmp_int3c2e_evaluator_list = []
    for gid in range(ngpu):
        with cupy.cuda.Device(gid):
            bas_ij_aggregated_tmp = []
            for i in range(len(bas_ij_aggregated_h)):
                bas_ij_aggregated_tmp.append(cupy.asarray(bas_ij_aggregated_h[i]))
            bas_ij_aggregated_tmp = tuple(bas_ij_aggregated_tmp)
            tmp_int3c2e_evaluator_list.append(int3c2e_opt.int3c2e_evaluator(
                ao_pair_batch_size=ao_pair_batch_size,
                aux_batch_size=None,
                bas_ij_aggregated=bas_ij_aggregated_tmp,
            ))

    # tmp_int3c2e_evaluator_list = Mg.mapgpu(lambda: int3c2e_opt.int3c2e_evaluator(
    #     ao_pair_batch_size=ao_pair_batch_size,
    #     aux_batch_size=None,
    #     bas_ij_aggregated=bas_ij_aggregated,
    # ))
    eval_j3c = [tmp_int3c2e_evaluator_list[i][0] for i in range(ngpu)]
    aux_sorting = tmp_int3c2e_evaluator_list[0][1]
    ao_pair_offsets = tmp_int3c2e_evaluator_list[0][2]
    
    # _aux_offsets = None

    if hasattr(ao_pair_offsets, "get"):
        ao_pair_offsets = ao_pair_offsets.get()
    ao_pair_offsets = numpy.asarray(ao_pair_offsets, dtype=numpy.int64)

    if with_long_range:
        # eval_ft, _ao_pair_offsets_ft = ft_opt.ft_evaluator(
        #     ao_pair_batch_size, bas_ij_aggregated=bas_ij_aggregated
        # )
        # tmp_ft_evaluator_list = Mg.mapgpu(lambda: ft_opt.ft_evaluator(
        #     ao_pair_batch_size, bas_ij_aggregated=bas_ij_aggregated
        # ))

        tmp_ft_evaluator_list = []
        for gid in range(ngpu):
            with cupy.cuda.Device(gid):
                bas_ij_aggregated_tmp = []
                for i in range(len(bas_ij_aggregated_h)):
                    bas_ij_aggregated_tmp.append(cupy.asarray(bas_ij_aggregated_h[i]))
                bas_ij_aggregated_tmp = tuple(bas_ij_aggregated_tmp)
                tmp_ft_evaluator_list.append(ft_opt.ft_evaluator(
                    ao_pair_batch_size,
                    bas_ij_aggregated=bas_ij_aggregated_tmp,
                ))

        eval_ft = [tmp_ft_evaluator_list[i][0] for i in range(ngpu)]
        _ao_pair_offsets_ft = tmp_ft_evaluator_list[0][1]
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
            auxG_cart_raw = cupy.asarray(
                # ft_ao.ft_ao(auxcell, Gv[g0:g1], sort_cell=False),
                ft_ao.ft_ao(auxcell, Gv[g0:g1]),
                dtype=cupy.complex128,
                order="C",
            )

            auxG_fit_blk = lib.gemm(
                aux_coeff0_c, auxG_cart_raw, buf=auxG_fit_buf, transa='T', transb='H'
            )
            auxG_fit_blk *= coulG[g0:g1]

            auxG_fit_blk_host = cupyx.empty_pinned(
                auxG_fit_blk.shape, dtype=auxG_fit_blk.dtype, order="C"
            )
            auxG_fit_blk.get(out=auxG_fit_blk_host, blocking=True)
            auxG_fit_blocks_cpu.append(auxG_fit_blk_host)
            auxG_cart_raw = None
            auxG_fit_blk = None
            log.timer('cderi_ovL_gamma_point_outcore_gpu4pyscf_Mg, prepare long range part, g_slices:%d/%d' %
                (g0, ngrids), *time0)
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

    file = lib.FileMp(path + '/eris.dat', 'w')
    cderi_file = file.create_dataset('cderi', (nmo_i, nmo_j, naux_fit), 'f8',
                                blksizes=(oblk, vblk, Kblksize))

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

    Mg.mapgpu(lambda: lib.free_all_blocks())

    npair_blk_max = int(numpy.max(ao_pair_offsets[1:] - ao_pair_offsets[:-1]))
    # Accumulate in host memory to reduce device-memory pressure for large MO spaces.
    read_host = [cupyx.empty_pinned((Kblksize, nmo_i, nmo_j), dtype=numpy.float64, order="C") for _ in range(ngpu)]
    write_host = [cupyx.empty_pinned((Kblksize, nmo_i, nmo_j), dtype=numpy.float64, order="C") for _ in range(ngpu)]
    
    j3c_fit_pair_buf = Mg.mapgpu(lambda: cupy.empty((naux_fit, npair_blk_max), dtype=cupy.float64, order="C"))
    pqG_buf = Mg.mapgpu(lambda: cupy.empty((npair_blk_max, Gblksize), dtype=cupy.complex128, order="C"))
    j3c_raw_buf = Mg.mapgpu(lambda: cupy.empty(
        (
            max(
                naux_cart*npair_blk_max,
                int(Kblksize*max_nao_blk*max_nao_blk/2)+1,
                int(Kblksize*max_nao_blk*nmo_j/2)+1,
                    ),
            ),
        dtype=cupy.complex128,
        order="C"))
    auxG_fit_block_buf = Mg.mapgpu(lambda: cupy.empty(
        (
            max(
                naux_cart * max(1, Gblksize),
                int(Kblksize * max_nao_blk * max_nao_blk / 2) + 1,
                int(Kblksize * nmo_j * nmo_i / 2) + 1,
            ),
        ),
        dtype=cupy.complex128,
        order="C",
    ))

    pair_address = Mg.broadcast(pair_address)
    pair_weight = Mg.broadcast(pair_weight)
    mo_coeff_i =  Mg.broadcast(mo_coeff_i)
    mo_coeff_j =  Mg.broadcast(mo_coeff_j)
    aux_coeff = Mg.broadcast(aux_coeff)

    pool_read = [Pool(processes=int(lib.NumFileProcess / 2)) for _ in range(ngpu)]
    pool_write = [Pool(processes=int(lib.NumFileProcess / 2)) for _ in range(ngpu)]

    time0 = log.timer("cderi_ovL_gamma_point_outcore_gpu4pyscf_Mg prepare", *time0)

    def cderi_gen_OVL(_ij_batch_id):
        gid = Mg.getgid()
        time1 = logger.process_clock(), logger.perf_counter()

        if _ij_batch_id >= shl_pair_batches:
            ij_batch_id = 0
        else:
            ij_batch_id = _ij_batch_id

        p0 = int(ao_pair_offsets[ij_batch_id])
        p1 = int(ao_pair_offsets[ij_batch_id + 1])
        if p1 <= p0:
            return

        addr = pair_address[gid][p0:p1]
        npair_blk = int(p1 - p0)
        i = addr // nao
        j = addr % nao
        ij_cat = cupy.concatenate((i, j), axis=0)
        ao_u, inv = cupy.unique(ij_cat, return_inverse=True)
        inv_i = inv[:npair_blk]
        inv_j = inv[npair_blk:]
        nao_blk = int(ao_u.size)
        local_addr = inv_i.astype(cupy.int64) * nao_blk + inv_j.astype(cupy.int64)
        Ci_blk = mo_coeff_i[gid][ao_u]
        Cj_blk = mo_coeff_j[gid][ao_u]
        w = pair_weight[gid][p0:p1]

        # Build the full fit-space pair block once for each AO-pair batch:
        # J_fit(K,pair) = aux_coeff^T * J_sr_cart + auxG_fit * J_lr_G
        j3c_fit_pair = lib.empty_from_buf(j3c_fit_pair_buf[gid], (naux_fit, npair_blk), dtype=cupy.float64)
        j3c_fit_pair[:] = 0

        # SR contribution: evaluate one full auxiliary batch.
        j3c_raw = eval_j3c[gid](shl_pair_batch_id=ij_batch_id, aux_batch_id=0, out=j3c_raw_buf[gid])
        if j3c_raw.dtype != cupy.float64:
            raise TypeError(f"Unexpected j3c_raw dtype: {j3c_raw.dtype}, expected float64")
        if j3c_raw.size > 0:
            j3c_cart = j3c_raw[:, :, 0]
            lib.gemm(aux_coeff[gid], j3c_cart, c=j3c_fit_pair, transa='T', transb='T')
            j3c_cart = None
        j3c_raw = None

        # LR contribution: accumulate in G-blocks to control memory usage.
        if with_long_range:
            for iblk, (g0, g1) in enumerate(g_slices):
                auxG_fit_blk = lib.empty_from_buf(auxG_fit_block_buf[gid], auxG_fit_blocks_cpu[iblk].shape, dtype=cupy.complex128)
                auxG_fit_blk.set(auxG_fit_blocks_cpu[iblk])
                # auxG_fit_blk = cupy.asarray(auxG_fit_blocks_cpu[iblk])
                pqG = eval_ft[gid](Gv[g0:g1], ij_batch_id, out=pqG_buf[gid])
                j3c_fit_pair_lr = lib.gemm(auxG_fit_blk, pqG, buf=j3c_raw_buf[gid], transb='T')
                j3c_fit_pair += j3c_fit_pair_lr.real
                pqG = None
                auxG_fit_blk = None
                j3c_fit_pair_lr = None

        waits_write = None
        # AO2MO in K-blocks to limit GPU memory footprint.
        K_list = [slice(K_start, K_stop) for K_start, K_stop in prange(0, naux_fit, Kblksize)]
        split_n = int(len(K_list) * gid / ngpu)
        K_list = K_list[split_n:] + K_list[:split_n]
        for sK in K_list:
            K0 = sK.start
            K1 = sK.stop
            Kblk = int(K1 - K0)
            
            if ngpu > 1:
                Mg.barrier()
            else:
                cupy.cuda.get_current_stream().synchronize()
            if _ij_batch_id >= shl_pair_batches:
                continue
            out_read_host = lib.empty_from_buf(read_host[gid], (nmo_i, nmo_j, Kblk), 'f8')
            waits_read = cderi_file.getitem(numpy.s_[:, :, K0:K1], pool=pool_read[gid], buf=out_read_host)

            j3c_pair_blk = lib.empty_from_buf(j3c_raw_buf[gid], (Kblk, npair_blk), dtype=cupy.float64)
            j3c_pair_blk[:] = j3c_fit_pair[K0:K1]
            j3c_pair_blk *= w[None, :]
            j3c_tril = lib.empty_from_buf(auxG_fit_block_buf[gid], (Kblk, nao_blk * nao_blk), dtype=cupy.float64)
            j3c_tril[:] = 0
            j3c_tril[:, local_addr] = j3c_pair_blk
            j3c_tril = j3c_tril.reshape(Kblk, nao_blk, nao_blk)
            j3c_tril_trans = lib.empty_from_buf(j3c_raw_buf[gid], (Kblk, nao_blk, nao_blk), dtype=cupy.float64)
            j3c_tril_trans[:] = j3c_tril.transpose(0, 2, 1)
            j3c_tril += j3c_tril_trans

            # Two-step AO2MO: Kij,ip->Kpj; Kpj,jq->Kpq
            # Kij,jq->qKi
            tmp = lib.contraction('Lij', j3c_tril, 'io', Ci_blk, 'Loj', buf=j3c_raw_buf[gid])
            # qKi,ip->pqK
            for p in waits_read:
                p.wait()
            out_tmp = lib.empty_from_buf(auxG_fit_block_buf[gid], (nmo_i, nmo_j, Kblk), 'f8')
            out_tmp.set(out_read_host)
            lib.contraction('Loj', tmp, 'jv', Cj_blk, 'ovL', out_tmp, beta=1.0)

            if waits_write is not None:
                for p in waits_write:
                    p.wait()
            out_write_host = lib.empty_from_buf(write_host[gid], (nmo_i, nmo_j, Kblk), 'f8')
            out_tmp.get(out=out_write_host, blocking=True)
            waits_write = cderi_file.setitem(numpy.s_[:, :, K0:K1], out_write_host, pool=pool_write[gid])

            j3c_pair_blk = None
            j3c_tril = None
            j3c_tril_trans = None
            tmp = None
            out_tmp = None

        if waits_write is not None:
            for p in waits_write:
                p.wait()

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

        if _ij_batch_id < shl_pair_batches:
            log.timer(
                'cderi_ovL_gamma_point_outcore_gpu4pyscf_Mg ij_id:%d/%d on GPU%d' % 
                (ij_batch_id, shl_pair_batches, gid), *time1)


    shl_pair_batches_list = list(range(shl_pair_batches))
    shl_pair_batche_size = len(shl_pair_batches_list)
    padding_size = ngpu - shl_pair_batche_size % ngpu
    shl_pair_batches_list = list(range(shl_pair_batches + padding_size))
    Mg.map(cderi_gen_OVL, shl_pair_batches_list)

    for pool_tmp in pool_read:
        pool_tmp.terminate()
        pool_tmp.join()
    for pool_tmp in pool_write:
        pool_tmp.terminate()
        pool_tmp.join()


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


    Mg.mapgpu(lambda: cupy.cuda.get_current_stream().synchronize())
    Mg.mapgpu(lambda: lib.free_all_blocks())
    gc.collect()
    oslices = [slice(*i) for i in prange(0, nmo_i, oblk)]
    time0 = log.timer("cderi_ovL_gamma_point_outcore_gpu4pyscf_Mg is done", *time0)
    return path, oslices
