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

from functools import reduce
from byteqc.embyte.Setting import RDM_MEMORY_POOL_SIZE
from byteqc import lib
from pyscf.lib import prange
import os
from mpi4py import MPI
from byteqc.embyte.Tools.logger import Logger
import gc
import threading
import multiprocessing
import cupyx
import numpy
import cupy
cupy.cuda.set_pinned_memory_allocator(None)


def make_RDM1_equi_pair_group(PR, LG, fragments, equi_part, equi_pair_group,
                              threshold_list, cheat_th=None, t2_y_buffer_pool=None, if_l2=False):
    '''
    Calculate the 1-RDM for cluster pairs.
    Here the cluster pairs are defined by the equi_pair_group.
    '''

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    LG_RDM = Logger(os.path.join(LG.filepath, 'RDM_logger_node%s.log' % rank))
    LG_RDM.logger.info('Loading Fock matrix.')
    Fock_MO = numpy.load(os.path.join(PR.filepath, 'Fock_MO.npy'))
    nocc_full, nvir_full = PR.recorder['nocc_nvir_full']

    doo_list = []
    dvv_list = []
    if if_l2:
        dov_list = [numpy.zeros((nocc_full, nvir_full))
                    for _ in range(len(threshold_list))]

    Fock_MO_occ = cupy.asarray(Fock_MO[: nocc_full, : nocc_full].copy())
    Fock_MO_vir = cupy.asarray(Fock_MO[nocc_full:, nocc_full:].copy())

    del Fock_MO

    clux = equi_pair_group[0][0][0]
    for equi_frag_group_x in equi_part:
        if clux in equi_frag_group_x:
            for clux_main in equi_frag_group_x:
                if fragments[clux_main]['equivalent_operator'] == ['main']:
                    break
            break

    result_list = numpy.zeros((len(threshold_list), len(equi_pair_group)))
    if cheat_th is not None:
        try:
            threshold_list_x = cheat_th[clux_main]
        except BaseException:
            threshold_list_x = threshold_list
    else:
        threshold_list_x = threshold_list

    if t2_y_buffer_pool is None:
        t2_y_buffer_pool_size = RDM_MEMORY_POOL_SIZE
        t2_y_buffer_pool = cupyx.zeros_pinned(
            (t2_y_buffer_pool_size, ), dtype='float64')
    else:
        t2_y_buffer_pool_size = t2_y_buffer_pool.size

    buff_total = None
    for th_ind in range(len(threshold_list_x)):

        doo = numpy.zeros((nocc_full, nocc_full))
        dvv = numpy.zeros((nvir_full, nvir_full))

        if if_l2:
            l1_g = numpy.load(PR.filepath + f'/l1_{th_ind}_global.npy')

        th_x = threshold_list_x[th_ind]
        th_x_str = 'occ_1e%.1f_vir_1e%.1f' % (numpy.log10(th_x[0]), numpy.log10(th_x[1]))

        LG_RDM.logger.info('loading theta')
        filepath_x = PR.filepath + '/Cluster/Cluster_%s/th_%s' % (clux, th_x_str)
        MO_occ_CLU_occ_x = cupy.asarray(
            numpy.load(filepath_x + '/MO_occ_CLU_occ.npy'))
        MO_vir_CLU_vir_x = cupy.asarray(
            numpy.load(filepath_x + '/MO_vir_CLU_vir.npy'))
        nocc_clux = MO_occ_CLU_occ_x.shape[1]
        nvir_clux = MO_vir_CLU_vir_x.shape[1]
        CLU_occ_FRAG_x_t = cupy.asarray(
            numpy.load(filepath_x + '/CLU_occ_FRAG.npy'))

        nfarg_clux = CLU_occ_FRAG_x_t.shape[1]

        filepath_x_main = PR.filepath + \
            '/Cluster/Cluster_%s/th_%s' % (clux_main, th_x_str)

        theta_size = nfarg_clux * nocc_clux * nvir_clux ** 2
        if theta_size * 8 / (1024 ** 3) > 30:
            slice_theta_len = 1
        else:
            slice_theta_len = nfarg_clux

        slice_theta = [
            slice(
                i[0],
                i[1]) for i in prange(
                0,
                nfarg_clux,
                slice_theta_len)]

        buffer_theta_cpu = cupyx.zeros_pinned(
            (slice_theta_len * nocc_clux, nvir_clux, nvir_clux), dtype='float64')
        pool_r = multiprocessing.Pool(processes=lib.NumFileProcess)
        for s_theta in slice_theta:
            cupy.cuda.get_current_stream().synchronize()
            n_s_theta = s_theta.stop - s_theta.start
            slice_r = slice(
                s_theta.start * nocc_clux,
                s_theta.stop * nocc_clux)
            theta = numpy.ndarray(
                (n_s_theta * nocc_clux,
                 nvir_clux,
                 nvir_clux),
                dtype='f8',
                buffer=buffer_theta_cpu.data)

            file = lib.FileMp(os.path.join(filepath_x_main, 't2'), 'r')
            wait_list = file['t2'].getitem(
                numpy.s_[slice_r], pool=pool_r, buf=theta)
            for w in wait_list:
                w.wait()
            file.close()

            theta = theta.reshape((n_s_theta, nocc_clux, nvir_clux, nvir_clux))
            lib.free_all_blocks()
            gc.collect()

            LG_RDM.logger.info('theta on memory')

            assert t2_y_buffer_pool.size > theta.size
            LG_RDM.logger.info('Transpose theta on CPU')
            theta_T = numpy.ndarray(
                theta.shape,
                dtype='f8',
                buffer=t2_y_buffer_pool.data
            )
            LG_RDM.logger.info('Copy theta to theta_T')
            numpy.copyto(theta_T, theta.transpose(0, 1, 3, 2))
            theta *= 2
            theta -= theta_T
            del theta_T

            LG_RDM.logger.info('Transpose theta done!')

            cupy.cuda.get_current_stream().synchronize()
            lib.free_all_blocks()
            gc.collect()

            if if_l2:
                dov = cupy.zeros((nocc_full, nvir_full))
                theta_gpu = cupy.asarray(theta)
                l1_g_gpu = cupy.asarray(l1_g)
                l1x = reduce(
                    cupy.dot, (MO_occ_CLU_occ_x.T, l1_g_gpu, MO_vir_CLU_vir_x))
                dovx1 = lib.contraction(
                    'xjab', theta_gpu, 'jb', l1x, 'xa', alpha=0.5)
                l1x = cupy.dot(CLU_occ_FRAG_x_t.T, l1x)
                dovx2 = lib.contraction(
                    'xiba', theta_gpu, 'xb', l1x, 'ia', alpha=0.5)

                del theta_gpu

                for equi_frag_group_x in equi_part:
                    if clux in equi_frag_group_x:
                        break
                for clux_i in equi_frag_group_x:
                    filepath_x_i = PR.filepath + \
                        '/Cluster/Cluster_%s/th_%s' % (clux_i, th_x_str)
                    MO_occ_FRAG_x_i = cupy.asarray(numpy.load(
                        filepath_x_i + '/MO_occ_FRAG.npy'))[:, s_theta].copy()
                    MO_occ_CLU_occ_x_i = cupy.asarray(
                        numpy.load(filepath_x_i + '/MO_occ_CLU_occ.npy'))
                    MO_vir_CLU_vir_x_i = cupy.asarray(
                        numpy.load(filepath_x_i + '/MO_vir_CLU_vir.npy'))

                    dovx_tmp = lib.contraction(
                        'xa', dovx1, 'Ox', MO_occ_FRAG_x_i, 'Oa')
                    lib.contraction(
                        'ia',
                        dovx2,
                        'Oi',
                        MO_occ_CLU_occ_x_i,
                        'Oa',
                        dovx_tmp,
                        beta=1.0,
                        alpha=1.0)
                    lib.contraction(
                        'Oa',
                        dovx_tmp,
                        'Va',
                        MO_vir_CLU_vir_x_i,
                        'OV',
                        dov,
                        beta=1.0,
                        alpha=1.0)

                dov_list[th_ind] += dov.get()
                del l1x, l1_g_gpu, dovx1, dovx2, MO_occ_FRAG_x_i, MO_occ_CLU_occ_x_i, MO_vir_CLU_vir_x_i, dovx_tmp, dov
                lib.free_all_blocks()
                gc.collect()

            cluy_main_info = query_t2_y(
                equi_pair_group,
                equi_part,
                fragments,
                cheat_th,
                threshold_list,
                th_ind,
                PR.filepath,
                t2_y_buffer_pool_size,
                if_l2=if_l2
            )
            cupy.cuda.get_current_stream().synchronize()
            theta_gpu = cupy.empty(theta.shape, dtype='float64')
            theta_gpu.set(theta)
            CLU_occ_FRAG_x = CLU_occ_FRAG_x_t[:, s_theta].copy()
            if s_theta.stop >= theta.shape[0]:
                del theta
                lib.free_all_blocks()
                gc.collect()

            load_t2_y_process_pool = [False] * \
                len(cluy_main_info['cluy_main_list'])
            t2_y_path = []
            t2_y_file = [False] * len(cluy_main_info['cluy_main_list'])
            t2_y_list = []
            for clu_y_ind_tmp, clu_y_main_load in enumerate(
                    cluy_main_info['cluy_main_list']):
                clu_y_main_load_info = cluy_main_info[clu_y_main_load]
                t2_y_path.append(clu_y_main_load_info['filepath'])

                buffer_slice = slice(
                    clu_y_main_load_info['start_stop'][0],
                    clu_y_main_load_info['start_stop'][1])
                t2_y_load = numpy.ndarray(
                    clu_y_main_load_info['shape'],
                    dtype='float64',
                    buffer=t2_y_buffer_pool[buffer_slice].data)
                t2_y_list.append(t2_y_load)

                if cluy_main_info[clu_y_main_load]['TakeAfter'] == []:
                    t2_y_file[clu_y_ind_tmp] = lib.FileMp(
                        t2_y_path[clu_y_ind_tmp], 'r')
                    if 'l2' == t2_y_path[clu_y_ind_tmp][-2:]:
                        amp_name = 'l2'
                    else:
                        amp_name = 't2'
                    load_t2_y_process_pool[clu_y_ind_tmp] = \
                        t2_y_file[clu_y_ind_tmp][amp_name].getitem(
                            numpy.s_[:],
                            pool=pool_r,
                            buf=t2_y_list[clu_y_ind_tmp]
                    )
                    cluy_main_info[clu_y_main_load]['status'] = 'Updating'
                    LG_RDM.logger.info(
                        f'Loading t2_y {clu_y_main_load} to CPU.....')

            cluy_match = None
            get_cluy_params = None
            container = None
            for equi_pair_ind, equi_pair in enumerate(equi_pair_group):

                cluy = equi_pair[0][1]

                for equi_frag_group_y in equi_part:
                    if cluy in equi_frag_group_y:
                        for cluy_main in equi_frag_group_y:
                            if fragments[cluy_main]['equivalent_operator'] == [
                                    'main']:
                                break
                        break

                if cluy_match == cluy_main:
                    change_t2y = False
                else:
                    change_t2y = True
                    cluy_match = cluy_main

                doox = cupy.zeros((theta_gpu.shape[1], nocc_full))
                dvvx = cupy.zeros((theta_gpu.shape[2], nvir_full))

                if cheat_th is not None:
                    try:
                        threshold_list_y = cheat_th[cluy_main]
                    except BaseException:
                        threshold_list_y = threshold_list
                else:
                    threshold_list_y = threshold_list
                th_y = threshold_list_y[th_ind]
                th_y_str = 'occ_1e%.1f_vir_1e%.1f' % (numpy.log10(th_y[0]), numpy.log10(th_y[1]))

                LG_RDM.logger.info('loading t2')

                if equi_pair_ind == 0 or get_cluy_params is None:
                    filepath_y = PR.filepath + \
                        '/Cluster/Cluster_%s/th_%s' % (cluy, th_y_str)
                    MO_occ_CLU_occ_y = cupy.asarray(
                        numpy.load(filepath_y + '/MO_occ_CLU_occ.npy'))
                    MO_vir_CLU_vir_y = cupy.asarray(
                        numpy.load(filepath_y + '/MO_vir_CLU_vir.npy'))
                    CLU_occ_FRAG_y = cupy.asarray(
                        numpy.load(filepath_y + '/CLU_occ_FRAG.npy'))
                    MO_occ_FRAG_y = cupy.asarray(
                        numpy.load(filepath_y + '/MO_occ_FRAG.npy'))

                else:
                    get_cluy_params.join()
                    get_cluy_params = None

                    MO_occ_CLU_occ_y, MO_vir_CLU_vir_y, CLU_occ_FRAG_y, MO_occ_FRAG_y = container

                if equi_pair_ind + 1 != len(equi_pair_group):
                    container = []
                    cluy_next_t = equi_pair_group[equi_pair_ind + 1][0][1]

                    for equi_frag_group_y_t in equi_part:
                        if cluy_next_t in equi_frag_group_y_t:
                            for cluy_main_next_t in equi_frag_group_y_t:
                                if fragments[cluy_main_next_t]['equivalent_operator'] == [
                                        'main']:
                                    break
                            break

                    if cheat_th is not None:
                        try:
                            threshold_list_y_next_t = cheat_th[cluy_main_next_t]
                        except BaseException:
                            threshold_list_y_next_t = threshold_list
                    else:
                        threshold_list_y_next_t = threshold_list

                    th_y_next_t = threshold_list_y_next_t[th_ind]
                    th_y_next_t_str = 'occ_1e%.1f_vir_1e%.1f' % (numpy.log10(th_y_next_t[0]), numpy.log10(th_y_next_t[1]))

                    filepath_y_next = PR.filepath + \
                        '/Cluster/Cluster_%s/th_%s' % (cluy_next_t,
                                                       th_y_next_t_str)
                    get_cluy_params = threading.Thread(
                        target=async_read_np_cluy, args=(
                            container, filepath_y_next, LG_RDM.logger))
                    get_cluy_params.start()

                nocc_cluy = MO_occ_CLU_occ_y.shape[1]
                nvir_cluy = MO_vir_CLU_vir_y.shape[1]
                nfrag_cluy = CLU_occ_FRAG_y.shape[1]

                LG_RDM.logger.info(f'Doing ... ... cluy_main - {cluy_main}')
                if change_t2y:

                    cluy_main_ind = cluy_main_info['cluy_main_list'].index(
                        cluy_main)

                    if cluy_main_ind != 0:
                        cluy_main_last = cluy_main_info['cluy_main_list'][cluy_main_ind - 1]
                        cluy_main_info[cluy_main_last]['status'] = 'Used'

                    LG_RDM.logger.info('renew t2')

                    if cluy_main_info[cluy_main]['status'] == 'On-disk':
                        if_load = True
                        for check_cluy_status in cluy_main_info[cluy_main]['TakeAfter']:
                            if_load = if_load and (
                                cluy_main_info[check_cluy_status]['status'] == 'Used')
                        if if_load:
                            t2_y_file[cluy_main_ind] = lib.FileMp(
                                t2_y_path[cluy_main_ind], 'r')
                            if 'l2' == t2_y_path[cluy_main_ind][-2:]:
                                amp_name = 'l2'
                            else:
                                amp_name = 't2'
                            load_t2_y_process_pool[cluy_main_ind] = \
                                t2_y_file[cluy_main_ind][amp_name].getitem(
                                    numpy.s_[:],
                                    pool=pool_r,
                                    buf=t2_y_list[cluy_main_ind]
                            )

                            cluy_main_info[cluy_main]['status'] = 'Updating'
                            LG_RDM.logger.info(
                                f'Loading t2_y {cluy_main} to CPU ......')
                        else:
                            raise KeyboardInterrupt(' ')

                    if cluy_main_info[cluy_main]['status'] == 'Updating':

                        for w in load_t2_y_process_pool[cluy_main_ind]:
                            w.wait()
                        t2_y_file[cluy_main_ind].close()

                        LG_RDM.logger.info(f't2_y {cluy_main} on CPU now !!!')
                    else:
                        raise KeyboardInterrupt(' ')

                    cluy_main_info[cluy_main]['status'] = 'Using'

                    try:
                        cluy_main_info['loading_list'].remove(cluy_main)
                    except BaseException:
                        pass

                    t2_y = t2_y_list[cluy_main_ind]
                    t2_y = t2_y.reshape(
                        (nfrag_cluy, nocc_cluy, nvir_cluy, nvir_cluy))

                    cluy_main_ind_next = cluy_main_ind + 1
                    for cluy_start_load_ind, cluy_start_load in enumerate(
                            cluy_main_info['cluy_main_list'][cluy_main_ind_next:]):
                        if cluy_main_info[cluy_start_load]['status'] == 'On-disk':
                            if_load = True
                            for check_cluy_status in cluy_main_info[cluy_start_load]['TakeAfter']:
                                if_load = if_load and (
                                    cluy_main_info[check_cluy_status]['status'] == 'Used')
                            if if_load:
                                cluy_next_ind_tmp = cluy_start_load_ind + cluy_main_ind_next
                                t2_y_file[cluy_next_ind_tmp] = lib.FileMp(
                                    t2_y_path[cluy_next_ind_tmp], 'r')
                                if 'l2' == t2_y_path[cluy_next_ind_tmp][-2:]:
                                    amp_name = 'l2'
                                else:
                                    amp_name = 't2'
                                load_t2_y_process_pool[cluy_next_ind_tmp] = \
                                    t2_y_file[cluy_next_ind_tmp][amp_name].getitem(
                                        numpy.s_[:],
                                        pool=pool_r,
                                        buf=t2_y_list[cluy_next_ind_tmp]
                                )

                                cluy_main_info[cluy_start_load]['status'] = 'Updating'
                                LG_RDM.logger.info(
                                    f'Loading t2_y {cluy_start_load} to CPU ......')

                LG_RDM.logger.info('t2 on memory')
                cupy.get_default_pinned_memory_pool().free_all_blocks()
                gc.collect()

                rxy_occ = cupy.dot(MO_occ_CLU_occ_x.T, MO_occ_CLU_occ_y)
                rxy_vir = cupy.dot(MO_vir_CLU_vir_x.T, MO_vir_CLU_vir_y)
                cfxy_occ = cupy.dot(rxy_occ, CLU_occ_FRAG_y)
                cfyx_occ = cupy.dot(rxy_occ.T, CLU_occ_FRAG_x)
                ffxy = cupy.dot(CLU_occ_FRAG_x.T, cfxy_occ)

                fx, ix, ax, bx = theta_gpu.shape
                fy, iy, ay, by = t2_y.shape

                # cupy.cuda.nvtx.RangePop()

                # cluster x : FIAB
                # cluster y : fiab
                # doo =
                # (FIAB, fiab, Ii, Aa, Bb -> Ff) +
                # (FIAB, fiab, Ff, Aa, Bb -> Ii) +
                # (FIAB, fiba, If, Ab, Ba -> Fi) +
                # (FIAB, fiba, Fi, Ab, Ba -> If)

                # dvv =
                # (FIAB, fiab, Ff, Ii, Aa -> Bb) +
                # (FIAB, fiab, Ff, Ii, Bb -> Aa) +
                # (FIAB, fiab, Fi, If, Ab -> Ba) +
                # (FIAB, fiab, Fi, If, Ba -> Ab)

                # -------------- Occupied part -----

                min_buff_size = fy * iy * fx * ix * 2 + \
                    fx * fy * ax * ay * 2 + fx * fy * ay * ay

                buff_tmp1_slice_size = max(
                    ix * ay * bx,
                    iy * ax * bx,
                )

                buff_tmp2_slice_size = max(
                    ix * ay * by,
                    iy * ay * bx,
                    iy * ax * by,
                )

                buff_tmp_y_slice_size = iy * ay * by

                if buff_total is not None:
                    buff_total[:] = 0
                else:
                    buff_total_size = int(lib.gpu_avail_bytes() / 8)
                    buff_total = cupy.zeros(
                        (buff_total_size, ), dtype='float64')
                    buff_total[:] = 0
                    t2_y_set_stream = cupy.cuda.Stream(
                        null=False, non_blocking=True, ptds=False)

                assert buff_total_size > min_buff_size
                remain_size = buff_total.size - fy * fx * ay * by - fx * fy * ax * ay \
                    - fy * iy * fx * ix * 2 - fx * fy * ax * ay

                slice_len = int(remain_size
                                / (buff_tmp1_slice_size
                                   + buff_tmp2_slice_size
                                   + buff_tmp_y_slice_size))
                while slice_len ** 2 * ix * iy > buff_tmp1_slice_size * slice_len:
                    slice_len -= 1
                assert slice_len > 0

                buff_shift = 0
                tmp_fFab = cupy.ndarray(
                    (fy, fx, ay, by), memptr=buff_total[buff_shift: buff_shift + fy * fx * ay * by].data)
                buff_shift += fy * fx * ay * by
                tmp_FfBb = cupy.ndarray(
                    (fx, fy, ax, ay), memptr=buff_total[buff_shift: buff_shift + fx * fy * ax * ay].data)
                buff_shift += fx * fy * ax * ay
                tmp_fiFI = cupy.ndarray(
                    (fy, iy, fx, ix), memptr=buff_total[buff_shift: buff_shift + fy * iy * fx * ix].data)
                buff_shift += fy * iy * fx * ix
                tmp_fiFI_T = cupy.ndarray(
                    (fy, iy, fx, ix), memptr=buff_total[buff_shift: buff_shift + fy * iy * fx * ix].data)
                buff_shift += fy * iy * fx * ix

                buff_shift_pin = buff_shift

                buff_ffvv = buff_total[buff_shift: buff_shift
                                       + fx * fy * ax * ay]
                buff_shift += fx * fy * ax * ay

                buff_tmp1 = buff_total[buff_shift: buff_shift
                                       + buff_tmp1_slice_size * min(slice_len, fx)]
                assert buff_tmp1.size == buff_tmp1_slice_size * min(slice_len, fx)
                buff_shift += buff_tmp1_slice_size * min(slice_len, fx)

                buff_tmp2 = buff_total[buff_shift: buff_shift
                                       + buff_tmp2_slice_size * min(slice_len, fx)]
                assert buff_tmp2.size == buff_tmp2_slice_size * min(slice_len, fx)
                buff_shift += buff_tmp2_slice_size * min(slice_len, fx)

                buff_tmp_y = buff_total[buff_shift: buff_shift
                                        + buff_tmp_y_slice_size * min(slice_len, fy)]
                assert buff_tmp_y.size == buff_tmp_y_slice_size * min(slice_len, fy)
                buff_shift += buff_tmp_y_slice_size * min(slice_len, fy)

                assert buff_total.size >= buff_shift

                slice_t2_y = [slice(i, i + slice_len)
                              for i in range(0, fy, slice_len)]
                slice_t2_x = [slice(i, i + slice_len)
                              for i in range(0, fx, slice_len)]
                for sy in slice_t2_y:
                    sy_len = min(sy.stop - sy.start, fy - sy.start)
                    cupy.cuda.get_current_stream().synchronize()
                    assert buff_tmp_y.size >= sy_len * iy * ay * by
                    assert sy_len > 0
                    t2_y_tmp_gpu = cupy.ndarray(
                        (sy_len, iy, ay, by), memptr=buff_tmp_y.data)
                    assert t2_y_tmp_gpu.size >= t2_y[sy].size
                    t2_y_tmp_gpu.set(t2_y[sy], stream=t2_y_set_stream)
                    for sx in slice_t2_x:
                        sx_len = min(sx.stop - sx.start, fx - sx.start)

                        t2_x_tmp_gpu = lib.contraction(
                            'FIAB', theta_gpu[sx], 'Aa', rxy_vir, 'FIaB', buf=buff_tmp1)
                        t2_x_tmp_gpu = lib.contraction(
                            'FIaB', t2_x_tmp_gpu, 'Bb', rxy_vir, 'FIab', buf=buff_tmp2)

                        t2_y_set_stream.synchronize()
                        assert buff_tmp1.size >= sx_len * sy_len * iy * ix
                        tmp_fiFI[sy, :, sx, :] = lib.contraction(
                            'fiab', t2_y_tmp_gpu, 'FIab', t2_x_tmp_gpu, 'fiFI', buf=buff_tmp1)
                        tmp_fiFI_T[sy, :, sx, :] = lib.contraction(
                            'fiab', t2_y_tmp_gpu, 'FIba', t2_x_tmp_gpu, 'fiFI', buf=buff_tmp1)

                        t2_x_tmp_gpu1 = lib.contraction(
                            'FIAB', theta_gpu[sx], 'Ii', rxy_occ, 'FiAB', buf=buff_tmp1)
                        t2_x_tmp_gpu2 = lib.contraction(
                            'FiAB', t2_x_tmp_gpu1, 'Aa', rxy_vir, 'FiaB', buf=buff_tmp2)

                        tmp_FfBb[sx,
                                 sy,
                                 :,
                                 :] += lib.contraction('FiaB',
                                                       t2_x_tmp_gpu2,
                                                       'fiab',
                                                       t2_y_tmp_gpu,
                                                       'FfBb',
                                                       buf=buff_ffvv)

                        t2_x_tmp_gpu2 = lib.contraction(
                            'FiAB', t2_x_tmp_gpu1, 'Bb', rxy_vir, 'FiAb', buf=buff_tmp2)

                        tmp_FfBb[sx,
                                 sy,
                                 :,
                                 :] += lib.contraction('FiAb',
                                                       t2_x_tmp_gpu2,
                                                       'fiab',
                                                       t2_y_tmp_gpu,
                                                       'FfAa',
                                                       buf=buff_ffvv)

                    lib.contraction(
                        'fiab',
                        t2_y_tmp_gpu,
                        'iF',
                        cfyx_occ,
                        'fFab',
                        tmp_fFab[sy])

                del buff_tmp_y, buff_tmp1, buff_tmp2, t2_x_tmp_gpu
                del t2_y_tmp_gpu, buff_ffvv, t2_x_tmp_gpu1, t2_x_tmp_gpu2

                assert buff_total.size - buff_shift_pin >= max(fx * fy + fx * MO_occ_FRAG_y.shape[0], ix * iy,
                                                               fx * iy + fx * MO_occ_CLU_occ_y.shape[0], ix * fy)

                buff_shift = buff_shift_pin
                buff_oo = buff_total[buff_shift: buff_shift + fx * fy]
                buff_shift += buff_oo.size
                tmp_oox = lib.contraction(
                    'fiFI',
                    tmp_fiFI,
                    'Ii',
                    rxy_occ,
                    'Ff',
                    alpha=-0.25,
                    buf=buff_oo)
                buff_oo2 = buff_total[buff_shift: buff_shift
                                      + fx * MO_occ_FRAG_y.shape[0]]

                tmp_oox = lib.gemm(
                    tmp_oox, MO_occ_FRAG_y, transb='T', buf=buff_oo2)

                lib.gemm(CLU_occ_FRAG_x, tmp_oox, c=doox, beta=1.0)

                buff_shift = buff_shift_pin
                buff_oo = buff_total[buff_shift: buff_shift + ix * iy]
                tmp_oox = lib.contraction(
                    'fiFI',
                    tmp_fiFI,
                    'Ff',
                    ffxy,
                    'Ii',
                    alpha=-0.25,
                    buf=buff_oo)
                lib.gemm(
                    tmp_oox,
                    MO_occ_CLU_occ_y,
                    c=doox,
                    transb='T',
                    beta=1.0)

                buff_shift = buff_shift_pin
                buff_oo = buff_total[buff_shift: buff_shift + fx * iy]
                buff_shift += buff_oo.size
                tmp_oox = lib.contraction(
                    'fiFI',
                    tmp_fiFI_T,
                    'If',
                    cfxy_occ,
                    'Fi',
                    alpha=-0.25,
                    buf=buff_oo)
                buff_oo2 = buff_total[buff_shift: buff_shift
                                      + fx * MO_occ_CLU_occ_y.shape[0]]
                tmp_oox = lib.gemm(
                    tmp_oox,
                    MO_occ_CLU_occ_y,
                    transb='T',
                    buf=buff_oo2)
                lib.gemm(CLU_occ_FRAG_x, tmp_oox, c=doox, beta=1.0)

                buff_shift = buff_shift_pin
                buff_oo = buff_total[buff_shift: buff_shift + ix * fy]
                tmp_oox = lib.contraction(
                    'fiFI',
                    tmp_fiFI_T,
                    'iF',
                    cfyx_occ,
                    'If',
                    alpha=-0.25,
                    buf=buff_oo)
                lib.gemm(tmp_oox, MO_occ_FRAG_y, c=doox, transb='T', beta=1.0)

                buff_shift_pin -= tmp_fiFI.size * 2

                del tmp_oox, buff_oo, buff_oo2, tmp_fiFI, tmp_fiFI_T

                # -------------- Virtual part -----

                assert buff_total.size - buff_shift_pin >= ax * ay

                buff_shift = buff_shift_pin
                buff_vv = buff_total[buff_shift: buff_shift + ax * ay]

                tmp_vv = lib.contraction(
                    'FfBb',
                    tmp_FfBb,
                    'Ff',
                    ffxy,
                    'Bb',
                    buf=buff_vv,
                    alpha=0.25)
                lib.gemm(
                    tmp_vv,
                    MO_vir_CLU_vir_y,
                    transb='T',
                    c=dvvx,
                    beta=1.0)

                buff_shift_pin -= tmp_FfBb.size
                del buff_vv, tmp_vv, tmp_FfBb

                assert buff_total.size - buff_shift_pin >= fx * \
                    fy * ax * bx + fy * fx * ay * ax + ax * ay

                buff_shift = buff_shift_pin
                buff_FfAB = buff_total[buff_shift: buff_shift
                                       + fx * fy * ax * bx]
                tmp_FfAB = cupy.ndarray(
                    (fx, fy, ax, bx), memptr=buff_FfAB.data)
                tmp_FfAB[:] = 0
                lib.contraction(
                    'FIAB',
                    theta_gpu,
                    'If',
                    cfxy_occ,
                    'FfAB',
                    tmp_FfAB)

                buff_shift += tmp_FfAB.size
                buff_tmp = buff_total[buff_shift: buff_shift
                                      + fy * fx * ay * ax]
                buff_shift += buff_tmp.size
                buff_vv = buff_total[buff_shift: buff_shift + ax * ay]
                tmp_vv = cupy.ndarray((ax, by), memptr=buff_vv.data)
                tmp_vv[:] = 0

                tmp_fFaA = lib.contraction(
                    'fFab', tmp_fFab, 'Ab', rxy_vir, 'fFaA', buf=buff_tmp)
                lib.contraction(
                    'fFaA',
                    tmp_fFaA,
                    'FfAB',
                    tmp_FfAB,
                    'Ba',
                    tmp_vv,
                    alpha=0.25,
                    beta=1.0)

                tmp_fFBb = lib.contraction(
                    'fFab', tmp_fFab, 'Ba', rxy_vir, 'fFBb', buf=buff_tmp)
                lib.contraction(
                    'fFBb',
                    tmp_fFBb,
                    'FfAB',
                    tmp_FfAB,
                    'Ab',
                    tmp_vv,
                    alpha=0.25,
                    beta=1.0)

                lib.gemm(
                    tmp_vv,
                    MO_vir_CLU_vir_y,
                    transb='T',
                    c=dvvx,
                    beta=1.0)

                del tmp_fFaA, tmp_fFBb, tmp_fFab, tmp_FfAB, tmp_vv, buff_tmp, buff_vv, buff_FfAB

                doox = lib.gemm(MO_occ_CLU_occ_x, doox)
                dvvx = lib.gemm(MO_vir_CLU_vir_x, dvvx)

                print(f'Fock_MO_vir.shape : {Fock_MO_vir.shape}')
                print(f'dvvx.shape : {dvvx.shape}')
                corr_energy = lib. gemm(Fock_MO_occ.reshape((1, -1)), doox.reshape((-1, 1)))[0, 0] \
                    + lib.gemm(Fock_MO_vir.reshape((1, -1)), dvvx.reshape((-1, 1)))[0, 0] \
                    + lib.gemm(Fock_MO_occ.reshape((1, -1)), doox.T.reshape((-1, 1)))[0, 0] \
                    + lib.gemm(Fock_MO_vir.reshape((1, -1)),
                               dvvx.T.reshape((-1, 1)))[0, 0]

                LG_RDM.logger.info(
                    '---- corr_energy is %s for clux-cluy pair %s with slice F [%s, %s] - %s threshold x-y %s - %s' %
                    (corr_energy, clux, s_theta.start, s_theta.stop, cluy, th_x, th_y))

                result_list[th_ind, equi_pair_ind] += corr_energy

                doox *= len(equi_pair)
                dvvx *= len(equi_pair)
                doo += doox.get()
                dvv += dvvx.get()

            theta_gpu = buff_total = t2_y = None
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
            gc.collect()

        doo_list.append(doo)
        dvv_list.append(dvv)
        pool_r.close()
        pool_r.join()

    if if_l2:
        return result_list, doo_list, dvv_list, dov_list
    else:
        return result_list, doo_list, dvv_list, None


def async_read_np_cluy(container, file_path, logger):
    container.append(
        cupy.asarray(
            numpy.load(
                file_path
                + '/MO_occ_CLU_occ.npy')))
    container.append(
        cupy.asarray(
            numpy.load(
                file_path
                + '/MO_vir_CLU_vir.npy')))
    container.append(cupy.asarray(numpy.load(file_path + '/CLU_occ_FRAG.npy')))
    container.append(cupy.asarray(numpy.load(file_path + '/MO_occ_FRAG.npy')))
    logger.info(f'Finish load next cluster y information.')


def query_t2_y(equi_pair_group, equi_part, fragments, cheat_th,
               threshold_list, th_ind, PR_filepath, t2_y_buffer_pool_size, if_l2=False):

    if if_l2:
        amp_name = 'l2'
    else:
        amp_name = 't2'
    cluy_main_list = []
    for equi_pair in equi_pair_group:
        cluy = equi_pair[0][1]
        for equi_frag_group_y in equi_part:
            if cluy in equi_frag_group_y:
                for cluy_main in equi_frag_group_y:
                    if fragments[cluy_main]['equivalent_operator'] == ['main']:
                        break
                break
        cluy_main_list.append(cluy_main)

    cluy_main_list = list(set(cluy_main_list))
    cluy_main_list.sort()

    cluy_main_dict = {}
    cluy_main_dict['total_size'] = 0
    cluy_main_dict['slice_cluy'] = []
    cluy_main_dict['cluy_main_list'] = cluy_main_list
    cluy_main_dict['loading_list'] = []
    slice_cluy = []
    slice_cluy_size = 0
    for cluy_main_ind, cluy_main in enumerate(cluy_main_list):
        cluy_main_dict[cluy_main] = {}

        try:
            threshold_list_y = cheat_th[cluy_main]
        except BaseException:
            threshold_list_y = threshold_list
        th_y = threshold_list_y[th_ind]
        th_y_str = 'occ_1e%.1f_vir_1e%.1f' % (numpy.log10(th_y[0]), numpy.log10(th_y[1]))

        cluy_main_dict[cluy_main]['TakeAfter'] = []
        cluy_main_dict[cluy_main]['thereshold'] = th_y

        cluy_main_dict[cluy_main]['filepath'] = PR_filepath + \
            '/Cluster/Cluster_%s/th_%s' % (cluy_main, th_y_str) + f'/{amp_name}'
        cluy_main_dict[cluy_main]['filepath_t'] = PR_filepath + \
            '/Cluster/Cluster_%s/th_%s' % (cluy_main, th_y_str)
        cluy_main_dict[cluy_main]['status'] = 'On-disk'

        with lib.FileMp(cluy_main_dict[cluy_main]['filepath'], 'r') as f:
            cluy_main_dict[cluy_main]['shape'] = f[amp_name].shape
            amp_size = numpy.prod(f[amp_name].shape)
            cluy_main_dict[cluy_main]['size'] = amp_size
            cluy_main_dict['total_size'] += amp_size
            assert amp_size <= t2_y_buffer_pool_size
            if slice_cluy_size + amp_size <= t2_y_buffer_pool_size:
                slice_cluy.append(cluy_main)
                cluy_main_dict[cluy_main]['start_stop'] = [
                    slice_cluy_size, slice_cluy_size + amp_size]
                slice_cluy_size += amp_size
            else:
                cluy_main_dict['slice_cluy'].append(slice_cluy)
                slice_cluy = [cluy_main]
                cluy_main_dict[cluy_main]['start_stop'] = [0, amp_size]
                slice_cluy_size = amp_size

            if cluy_main_ind + 1 == len(cluy_main_list):
                assert slice_cluy != []
                cluy_main_dict['slice_cluy'].append(slice_cluy)

    for slice_ind, slice_cluy in enumerate(cluy_main_dict['slice_cluy']):
        if slice_ind == 0:
            continue

        else:
            cluy_main_start = 0
            for cluy_main in slice_cluy:
                cluy_main_stop = cluy_main_start + \
                    cluy_main_dict[cluy_main]['size']
                for slice_cluy_before in cluy_main_dict['slice_cluy'][: slice_ind]:
                    cluy_main_before_start = 0
                    for cluy_main_before in slice_cluy_before:
                        cluy_main_before_stop = cluy_main_before_start + \
                            cluy_main_dict[cluy_main_before]['size']

                        judge = cluy_main_before_start < cluy_main_stop and cluy_main_before_stop > cluy_main_start

                        if judge:
                            cluy_main_dict[cluy_main]['TakeAfter'].append(
                                cluy_main_before)
                        cluy_main_before_start = cluy_main_before_stop

                cluy_main_start = cluy_main_stop

    return cluy_main_dict
