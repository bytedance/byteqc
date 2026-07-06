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

from ..Setting import MIN_GROUP_SIZE, TEMP_DIR
import cupyx
import uuid
from functools import reduce
from pyscf.lib import prange
from byteqc.embyte.Tools.tool_lib import take_pointer_data_from_array, get_array_from_pointer_data
from byteqc.lib.linalg import solve_triangular
from byteqc import lib
from byteqc.cuobc.lib.int3c import VHFOpt3c, get_int2c, get_int3c
import h5py
import os
import gc
import threading
import multiprocessing
import numpy
import cupy
cupy.cuda.set_pinned_memory_allocator(None)


def generate_tmpfile_name():
    unique_name = str(uuid.uuid4())
    return unique_name


def build_vhfopt3c(mol, auxmol, blksize=MIN_GROUP_SIZE):
    vhfopt = VHFOpt3c(mol, auxmol, 'int2e')
    vhfopt.build(
        group_size=blksize,
        aux_group_size=blksize,
    )
    return vhfopt, blksize


def get_j2c(logger, mol=None, auxmol=None, vhfopt3c=None,
            j2c_eig_always=False, linear_dep_threshold=1e-13):
    '''
    Return j2c for the cderi generation.
    '''
    if vhfopt3c is None:
        logger.info('----- Build VHFOpt3c')
        vhfopt, _ = build_vhfopt3c(mol, auxmol)
        logger.info('----- Build VHFOpt3c is done.')

    j2c = get_int2c(vhfopt, j2c_eig_always, linear_dep_threshold)
    naux_cart, naux = vhfopt.auxcoeff.shape
    j2c_h = cupyx.empty_pinned(j2c.shape, dtype='f8')
    j2c.get(out=j2c_h, blocking=True)

    return j2c_h, naux_cart, naux


def eri_OVL_SIE_MP2(mol, auxmol, mo_coeff_occ1, mo_coeff_unocc1,
                    mo_coeff_occ2, mo_coeff_unocc2, j2c, logger, vhfopt=None):
    '''
    On-the-flying calculate 2 cderis in shape of (nocc1, nvir1, naux)
    and (nvir2, nocc2, naux) corresponding to mo_coeff 1 and 2.
    The cderis are used in MP2 to generate BNOs.
    '''

    mo_coeff_occ1 = cupy.asarray(mo_coeff_occ1)
    mo_coeff_occ2 = cupy.asarray(mo_coeff_occ2)
    mo_coeff_unocc1 = cupy.asarray(mo_coeff_unocc1)
    mo_coeff_unocc2 = cupy.asarray(mo_coeff_unocc2)
    nocc1 = mo_coeff_occ1.shape[1]
    nvir1 = mo_coeff_unocc1.shape[1]
    nocc2 = mo_coeff_occ2.shape[1]
    nvir2 = mo_coeff_unocc2.shape[1]

    lib.free_all_blocks()
    gc.collect()
    blksize = MIN_GROUP_SIZE
    if vhfopt is None:
        logger.info('----- Build VHFOpt3c')
        vhfopt, blksize = build_vhfopt3c(mol, auxmol, blksize)
        logger.info('----- Build VHFOpt3c is done.')

    assert isinstance(
        j2c, str), 'Seems the j2c build in low_level_process is missed, please delete the logpath file and rerun.'

    f_j2c = h5py.File(j2c, 'r')
    j2c = cupyx.zeros_pinned(f_j2c['j2c'].shape, dtype='f8', order='F')

    def load_j2c(file_obj, j2c_buffer):
        j2c_buffer[:] = file_obj['j2c']

    thread_load_j2c = threading.Thread(target=load_j2c, args=(f_j2c, j2c, ))
    thread_load_j2c.start()

    mo_coeff_occ1 = reduce(
        cupy.dot, (cupy.asarray(
            vhfopt.coeff), mo_coeff_occ1))
    mo_coeff_unocc1 = reduce(
        cupy.dot, (cupy.asarray(
            vhfopt.coeff), mo_coeff_unocc1))
    mo_coeff_occ2 = reduce(
        cupy.dot, (cupy.asarray(
            vhfopt.coeff), mo_coeff_occ2))
    mo_coeff_unocc2 = reduce(
        cupy.dot, (cupy.asarray(
            vhfopt.coeff), mo_coeff_unocc2))

    naux_cart, naux = vhfopt.auxcoeff.shape
    if isinstance(vhfopt.auxcoeff, cupy.ndarray):
        auxcoeff_cpu = cupyx.zeros_pinned(
            vhfopt.auxcoeff.shape, dtype='f8')
        vhfopt.auxcoeff.get(out=auxcoeff_cpu, blocking=True)
        vhfopt.auxcoeff = auxcoeff_cpu
        del auxcoeff_cpu

    if isinstance(vhfopt.coeff, cupy.ndarray):
        coeff_cpu = cupyx.zeros_pinned(
            vhfopt.coeff.shape, dtype='f8')
        vhfopt.coeff.get(out=coeff_cpu, blocking=True)
        vhfopt.coeff = coeff_cpu
        del coeff_cpu

    lib.free_all_blocks()
    gc.collect()

    buff_tmp = cupy.empty((blksize, max(nocc1, nvir2) * blksize), dtype='f8')
    buff_eri_ov1 = cupy.empty((blksize, nocc1 * nvir1), dtype='f8')
    buff_eri_ov2 = cupy.empty((blksize, nocc2 * nvir2), dtype='f8')
    buff_int3c = cupy.empty((blksize, blksize ** 2), dtype='f8')
    unLov1 = cupyx.empty_pinned((naux_cart, nocc1 * nvir1), dtype='f8')
    unLvo2 = cupyx.empty_pinned((naux_cart, nocc2 * nvir2), dtype='f8')

    GPU_buf_size = (buff_tmp.size + buff_eri_ov1.size
                    + buff_eri_ov2.size + buff_int3c.size) * 8 / (1024 ** 3)
    CPU_buf_size = (unLov1.size + unLvo2.size) * 8 / (1024 ** 3)
    logger.info(f'GPU buffer size {GPU_buf_size} GB.')
    logger.info(f'CPU buffer size {CPU_buf_size} GB.')
    nauxid = len(vhfopt.aux_log_qs)
    kslices = []
    kextents = []

    for cp_aux_id in range(nauxid):
        k0, k1 = vhfopt.auxmol.ao_loc[vhfopt.aux_l_ctr_offsets
                                      [cp_aux_id: cp_aux_id + 2]]
        kslices.append(slice(k0, k1))
        kextents.append(k1 - k0)

    for cp_aux_id in range(nauxid):
        logger.info(
            'eri_OVL_SIE_MP2, get int3c, cp_aux_id:%d/%d' %
            (cp_aux_id + 1, nauxid))
        sLov1 = lib.empty_from_buf(
            buff_eri_ov1, (kextents[cp_aux_id], nocc1, nvir1), 'f8')
        sLvo2 = lib.empty_from_buf(
            buff_eri_ov2, (kextents[cp_aux_id], nvir2, nocc2), 'f8')
        sLov1[:] = 0
        sLvo2[:] = 0
        for cp_ij_id in range(len(vhfopt.log_qs)):
            si, sj, sk, int3c = get_int3c(
                cp_ij_id, cp_aux_id, vhfopt, buf=buff_int3c)
            tmp = lib.contraction(
                'ijL',
                int3c,
                'io',
                mo_coeff_occ1[si],
                'Loj',
                buf=buff_tmp)
            lib.contraction(
                'Loj',
                tmp,
                'jv',
                mo_coeff_unocc1[sj],
                'Lov',
                sLov1,
                beta=1.0)
            if si != sj:
                tmp = lib.contraction(
                    'ijL',
                    int3c,
                    'jo',
                    mo_coeff_occ1[sj],
                    'Loi',
                    buf=buff_tmp)
                lib.contraction(
                    'Loi',
                    tmp,
                    'iv',
                    mo_coeff_unocc1[si],
                    'Lov',
                    sLov1,
                    beta=1.0)

            tmp = lib.contraction(
                'ijL',
                int3c,
                'iv',
                mo_coeff_unocc2[si],
                'Lvj',
                buf=buff_tmp)
            lib.contraction(
                'Lvj',
                tmp,
                'jo',
                mo_coeff_occ2[sj],
                'Lvo',
                sLvo2,
                beta=1.0)
            if si != sj:
                tmp = lib.contraction(
                    'ijL',
                    int3c,
                    'jv',
                    mo_coeff_unocc2[sj],
                    'Lvi',
                    buf=buff_tmp)
                lib.contraction(
                    'Lvi',
                    tmp,
                    'io',
                    mo_coeff_occ2[si],
                    'Lvo',
                    sLvo2,
                    beta=1.0)

        sLov1.reshape(kextents[cp_aux_id], -1
                      ).get(out=unLov1[kslices[cp_aux_id]], blocking=True)
        sLvo2.reshape(kextents[cp_aux_id], -1
                      ).get(out=unLvo2[kslices[cp_aux_id]], blocking=True)

    del int3c, buff_int3c, buff_tmp, tmp, sLov1, sLvo2, buff_eri_ov1, buff_eri_ov2
    del mo_coeff_occ1, mo_coeff_occ2, mo_coeff_unocc1, mo_coeff_unocc2
    lib.free_all_blocks()
    gc.collect()

    thread_load_j2c.join()
    f_j2c.close()

    if logger is not None:
        logger.info('----- solve to get cderi -----')

    auxcoeff = cupy.asarray(vhfopt.auxcoeff)
    del vhfopt
    lib.free_all_blocks()
    gc.collect()

    j2c = cupy.asarray(j2c)

    avail_gpu_memory = lib.gpu_avail_bytes() / 8
    slice_len_ov = min(max(nocc1 * nvir1, nocc2 * nvir2),
                       int(avail_gpu_memory // (naux_cart + naux)))
    logger.info(
        '----- The slice_len_ov for cupy.linalg.solve: %s' %
        slice_len_ov)

    buff = cupy.empty((naux_cart, slice_len_ov), dtype='f8')
    buff2 = cupy.empty((naux * slice_len_ov), dtype='f8')
    ovL1 = cupyx.zeros_pinned((nocc1 * nvir1, naux), dtype='f8')
    voL2 = cupyx.zeros_pinned((nocc2 * nvir2, naux), dtype='f8')
    trans_buff_h = cupyx.empty_pinned(buff.shape, dtype='f8')

    logger.info(
        f'CPU buffer size : {(trans_buff_h.size + ovL1.size + voL2.size) * 8 / (1024 ** 3)} GB')
    logger.info(
        f'GPU buffer size : {(buff.size + buff2.size + j2c.size) * 8 / (1024 ** 3)} GB')
    sov1_list = [
        slice(
            i[0],
            i[1]) for i in prange(
            0,
            nocc1 * nvir1,
            slice_len_ov)]
    sov2_list = [
        slice(
            i[0],
            i[1]) for i in prange(
            0,
            nocc2 * nvir2,
            slice_len_ov)]
    for sov_ind, sov in enumerate(sov1_list):
        logger.info(
            'eri_OVL_SIE_MP2, linalg.solve, for ovL1, sov1:%d/%d' %
            (sov_ind + 1, len(sov1_list)))
        nov = sov.stop - sov.start
        unLovs = lib.empty_from_buf(buff, (naux_cart, nov), 'f8')

        unLovs_h = lib.empty_from_buf(trans_buff_h, unLovs.shape, 'f8')
        numpy.copyto(unLovs_h, unLov1[:, sov])
        unLovs.set(unLovs_h)

        ovs_L = lib.gemm(unLovs, auxcoeff, buf=buff2, transa='T', transb='N')

        solve_triangular(j2c, ovs_L.T, lower=True, overwrite_b=True,)

        ovs_L.get(out=ovL1[sov], blocking=True)

    for sov_ind, sov in enumerate(sov2_list):
        logger.info(
            'eri_OVL_SIE_MP2, linalg.solve, for voL2, sov2:%d/%d' %
            (sov_ind + 1, len(sov2_list)))
        nov = sov.stop - sov.start
        unLovs = lib.empty_from_buf(buff, (naux_cart, nov), 'f8')

        unLovs_h = lib.empty_from_buf(trans_buff_h, unLovs.shape, 'f8')
        numpy.copyto(unLovs_h, unLvo2[:, sov])
        unLovs.set(unLovs_h)

        ovs_L = lib.gemm(unLovs, auxcoeff, buf=buff2, transa='T', transb='N')

        solve_triangular(j2c, ovs_L.T, lower=True, overwrite_b=True,)

        assert ovs_L.flags.c_contiguous

        ovs_L.get(out=voL2[sov], blocking=True)

    del unLov1, unLvo2, unLovs, unLovs_h, ovs_L, buff, buff2, trans_buff_h, j2c
    lib.free_all_blocks()
    gc.collect()

    return ovL1.reshape(nocc1, nvir1, -1), voL2.reshape(nvir2, nocc2, -1)


def eri_ondisk_OVL_SIE_MP2(mol, cderi_AO_file, mo_coeff_occ1,
                           mo_coeff_unocc1, mo_coeff_occ2, mo_coeff_unocc2, logger):
    '''
    Calculate 2 cderis based on full cderi on the disk.
    The 2 cderis are in shape of (nocc1, nvir1, naux) and
    (nvir2, nocc2, naux) corresponding to mo_coeff 1 and 2.
    The cderis are used in MP2 to generate BNOs.
    '''

    lib.free_all_blocks()
    gc.collect()

    load_process_num = lib.NumFileProcess

    mo_coeff_occ1 = cupy.asarray(mo_coeff_occ1)
    mo_coeff_unocc1 = cupy.asarray(mo_coeff_unocc1)
    mo_coeff_occ2 = cupy.asarray(mo_coeff_occ2)
    mo_coeff_unocc2 = cupy.asarray(mo_coeff_unocc2)

    if os.path.isdir(cderi_AO_file):
        j3c_type = 0
        cderi_AO_file_tmp = os.path.join(cderi_AO_file, '0-0')
        with h5py.File(cderi_AO_file_tmp, 'r') as f:
            naux = f['j3c'].shape[0]
    else:
        if getattr(mol, 'pbc_intor', None):
            j3c_type = 1
            with h5py.File(cderi_AO_file, 'r') as f:
                naux = f['j3c/0/0'].shape[0]
        else:
            j3c_type = 2
            with h5py.File(cderi_AO_file, 'r') as f:
                naux = f['j3c/0'].shape[0]

    nmo = mo_coeff_occ1.shape[0]
    nocc1 = mo_coeff_occ1.shape[1]
    nvir1 = mo_coeff_unocc1.shape[1]
    nocc2 = mo_coeff_occ2.shape[1]
    nvir2 = mo_coeff_unocc2.shape[1]

    free_size = lib.gpu_avail_bytes() / 8

    nmo_pair = int(nmo * (nmo - 1) / 2 + nmo)

    naux_slice_len = min(naux,
                         free_size / (nmo ** 2 + max(nmo_pair,
                                                     nmo * max(nocc1,
                                                               nvir2)) + max(nocc1 * nvir1,
                                                                             nocc2 * nvir2)))
    naux_slice_len = int(naux_slice_len)

    auxslice = [slice(i[0], i[1]) for i in prange(0, naux, naux_slice_len)]

    max_num_pair_slice_len = 0
    if j3c_type == 0:
        slice_ind_list = []
        for _, _, files in os.walk(cderi_AO_file):
            for filename in files:
                assert 'swap' not in filename
                assert '0' == filename[0]

                slice_ind_list.append(int(filename.split('-')[-1]))

                with h5py.File(os.path.join(cderi_AO_file, filename), 'r') as f:
                    max_num_pair_slice_len = max(
                        f['j3c'].shape[1], max_num_pair_slice_len)
        slice_ind_list.sort()
    elif j3c_type == 1:
        with h5py.File(cderi_AO_file, 'r') as f:
            assert len(list(f['j3c'].keys())) == 1
            j3c_0_slice = list(f['j3c/0'].keys())
            for s_j3c_0 in j3c_0_slice:
                max_num_pair_slice_len = max(
                    max_num_pair_slice_len, f[f'j3c/0/{s_j3c_0}'].shape[1])
        slice_ind_list = None

    elif j3c_type == 2:
        with h5py.File(cderi_AO_file, 'r') as f:
            j3c_0_slice = list(f['j3c'].keys())
            for s_j3c_0 in j3c_0_slice:
                max_num_pair_slice_len = max(
                    max_num_pair_slice_len, f[f'j3c/{s_j3c_0}'].shape[1])
        slice_ind_list = None
    else:
        assert False

    buffer_CPU_eri_load = cupyx.empty_pinned(
        (naux_slice_len * nmo_pair), dtype='f8')
    buffer_CPU_eri_load2 = cupyx.empty_pinned(
        (naux_slice_len * max(nocc1 * nvir1, nocc2 * nvir2)), dtype='f8')
    eri_load_tmp = cupyx.empty_pinned(
        (naux_slice_len, max_num_pair_slice_len), dtype='f8')
    buffer_gpu_cderi = cupy.empty((naux_slice_len, nmo * nmo), dtype='f8')
    buffer_gpu_tmp = cupy.empty(
        (naux_slice_len * max(nmo_pair, nmo * max(nocc1, nvir2))), dtype='f8')
    buffer_gpu_save = cupy.empty(
        (naux_slice_len, max(nocc1 * nvir1, nocc2 * nvir2)), dtype='f8')

    ovL1 = cupyx.zeros_pinned((nocc1, nvir1, naux), dtype='f8')
    voL2 = cupyx.zeros_pinned((nvir2, nocc2, naux), dtype='f8')

    read_pool = multiprocessing.Pool(processes=load_process_num)
    cderi_next = None
    for sL_ind, sL in enumerate(auxslice):
        logger.info(
            'eri_OVL_SIE_MP2, get int3c, cp_aux_id:%d/%d' %
            (sL_ind + 1, len(auxslice)))
        sL_len = sL.stop - sL.start
        if sL_ind == 0:
            cderi = lib.empty_from_buf(
                buffer_CPU_eri_load, (sL_len, nmo_pair), 'f8')
            process_r_cderi = []
            norb_taken_tmp = max(int(sL_len / load_process_num), 1)
            slice_load = [slice(i, min(i + norb_taken_tmp, sL.start + sL_len))
                          for i in range(sL.start, (sL.start + sL_len), norb_taken_tmp)]
            slice_load_r = [slice(i, min(i + norb_taken_tmp, sL_len))
                            for i in range(0, sL_len, norb_taken_tmp)]

            for sload_ind in range(len(slice_load)):
                s_load = slice_load[sload_ind]
                s_load_r = slice_load_r[sload_ind]
                process = read_pool.apply_async(
                    read_async_PBC, (j3c_type, cderi_AO_file, s_load, take_pointer_data_from_array(
                        cderi[s_load_r]), take_pointer_data_from_array(
                        eri_load_tmp[s_load_r]), slice_ind_list))
                process_r_cderi.append(process)

            for p in process_r_cderi:
                p.get()
        else:
            for p in process_r_cderi:
                p.get()

            cderi = cderi_next

        cderi_pack = lib.empty_from_buf(buffer_gpu_tmp, cderi.shape, 'f8')
        cderi_pack.set(cderi)
        cupy.cuda.get_current_stream().synchronize()
        if sL_ind + 1 < len(auxslice):
            sL_next = auxslice[sL_ind + 1]
            sL_len_next = min(
                naux - sL_next.start,
                sL_next.stop - sL_next.start)
            cderi_next = lib.empty_from_buf(
                buffer_CPU_eri_load, (sL_len_next, nmo_pair), 'f8')

            process_r_cderi = []
            norb_taken_tmp = max(int(sL_len_next / load_process_num), 1)
            slice_load = [slice(i, min(i + norb_taken_tmp, sL_next.start + sL_len_next))
                          for i in range(sL_next.start, (sL_next.start + sL_len_next), norb_taken_tmp)]
            slice_load_r = [slice(i, min(i + norb_taken_tmp, sL_len_next))
                            for i in range(0, sL_len_next, norb_taken_tmp)]

            for sload_ind in range(len(slice_load)):
                s_load = slice_load[sload_ind]
                s_load_r = slice_load_r[sload_ind]
                process = read_pool.apply_async(
                    read_async_PBC, (j3c_type, cderi_AO_file, s_load, take_pointer_data_from_array(
                        cderi_next[s_load_r]), take_pointer_data_from_array(
                        eri_load_tmp[s_load_r]), slice_ind_list))
                process_r_cderi.append(process)

        cderi = lib.empty_from_buf(buffer_gpu_cderi, (sL_len, nmo, nmo), 'f8')

        lib.unpack_tril(cderi_pack, out=cderi)

        cderi_save_tmp = lib.empty_from_buf(
            buffer_CPU_eri_load2, (nocc1, nvir1, sL_len))
        tmp = lib.contraction(
            'Lij',
            cderi,
            'io',
            mo_coeff_occ1,
            'ojL',
            buf=buffer_gpu_tmp)
        ov_sL = lib.contraction(
            'ojL',
            tmp,
            'jv',
            mo_coeff_unocc1,
            'ovL',
            buf=buffer_gpu_save)
        ov_sL.get(out=cderi_save_tmp, blocking=True)
        ovL1[:, :, sL] = cderi_save_tmp

        cderi_save_tmp = lib.empty_from_buf(
            buffer_CPU_eri_load2, (nvir2, nocc2, sL_len))
        tmp = lib.contraction(
            'Lij',
            cderi,
            'jv',
            mo_coeff_unocc2,
            'viL',
            buf=buffer_gpu_tmp)
        vo_sL = lib.contraction(
            'viL',
            tmp,
            'io',
            mo_coeff_occ2,
            'voL',
            buf=buffer_gpu_save)
        vo_sL.get(out=cderi_save_tmp, blocking=True)
        voL2[:, :, sL] = cderi_save_tmp

    read_pool.close()
    read_pool.join()
    buffer_CPU_eri_load = buffer_CPU_eri_load2 = eri_load_tmp = buffer_gpu_cderi = buffer_gpu_tmp = buffer_gpu_save = None
    vo_sL = cderi_save_tmp = tmp = ov_sL = cderi = cderi_next = cderi_pack = None

    return ovL1, voL2


def read_async_PBC(j3c_type, filepath, sv, x_pointer, buf_tmp_pointer,
                   slice_ind_list=None):
    x = get_array_from_pointer_data(x_pointer)
    buf_tmp = get_array_from_pointer_data(buf_tmp_pointer)
    if j3c_type == 0:
        assert slice_ind_list is not None
        start_j3c = 0
        for s_j3c_0 in slice_ind_list:
            s_j3c_0 = str(s_j3c_0)
            with h5py.File(os.path.join(filepath, f'0-{s_j3c_0}'), 'r') as f:
                s_j3c_0_len = f['j3c'].shape[1]
                x_tmp = lib.empty_from_buf(buf_tmp, (x.shape[0], s_j3c_0_len))
                f['j3c'].read_direct(x_tmp, source_sel=sv)
                s_j3c_w = slice(start_j3c, start_j3c + s_j3c_0_len)
                x[:, s_j3c_w] = x_tmp
                start_j3c += s_j3c_0_len

    elif j3c_type == 1:
        with h5py.File(filepath, 'r') as f:
            j3c_0_slice = list(f['j3c/0'].keys())
            j3c_0_slice = numpy.asarray([int(i) for i in j3c_0_slice])
            j3c_0_slice = j3c_0_slice[j3c_0_slice.argsort()]
            start_j3c = 0
            for s_j3c_0 in j3c_0_slice:
                s_j3c_0 = str(s_j3c_0)
                s_j3c_0_len = f[f'j3c/0/{s_j3c_0}'].shape[1]
                x_tmp = lib.empty_from_buf(buf_tmp, (x.shape[0], s_j3c_0_len))
                f[f'j3c/0/{s_j3c_0}'].read_direct(x_tmp, source_sel=sv)
                s_j3c_w = slice(start_j3c, start_j3c + s_j3c_0_len)
                x[:, s_j3c_w] = x_tmp
                start_j3c += s_j3c_0_len

    elif j3c_type == 2:
        with h5py.File(filepath, 'r') as f:
            j3c_0_slice = list(f['j3c'].keys())
            j3c_0_slice = numpy.asarray([int(i) for i in j3c_0_slice])
            j3c_0_slice = j3c_0_slice[j3c_0_slice.argsort()]
            start_j3c = 0
            for s_j3c_0 in j3c_0_slice:
                s_j3c_0 = str(s_j3c_0)
                s_j3c_0_len = f[f'j3c/{s_j3c_0}'].shape[1]
                x_tmp = lib.empty_from_buf(buf_tmp, (x.shape[0], s_j3c_0_len))
                f[f'j3c/{s_j3c_0}'].read_direct(x_tmp, source_sel=sv)
                s_j3c_w = slice(start_j3c, start_j3c + s_j3c_0_len)
                x[:, s_j3c_w] = x_tmp
                start_j3c += s_j3c_0_len
    else:
        assert False


def eri_high_level_solver_incore(mol, auxmol, mo_coeff_occ, mo_coeff_unocc,
                                 j2c, logger, solver_type='MP2', vhfopt=None, svd_tol=1e-4):
    '''
    On-the-flying generate cderi for cluster which is under high-level solver processing.
    '''
    mo_coeff_occ = cupy.asarray(mo_coeff_occ)
    mo_coeff_unocc = cupy.asarray(mo_coeff_unocc)
    nocc = mo_coeff_occ.shape[1]
    nvir = mo_coeff_unocc.shape[1]

    lib.free_all_blocks()
    gc.collect()
    blksize = MIN_GROUP_SIZE
    if vhfopt is None:
        logger.info('----- Build VHFOpt3c')
        vhfopt, blksize = build_vhfopt3c(mol, auxmol, blksize)
        logger.info('----- Build VHFOpt3c is done.')

    assert isinstance(
        j2c, str), 'Seems the j2c build in low_level_process is missed, please delete the logpath file and rerun.'

    f_j2c = h5py.File(j2c, 'r')
    j2c = cupyx.zeros_pinned(f_j2c['j2c'].shape, dtype='f8', order='F')

    def load_j2c(file_obj, j2c_buffer):
        j2c_buffer[:] = file_obj['j2c']

    thread_load_j2c = threading.Thread(target=load_j2c, args=(f_j2c, j2c, ))
    thread_load_j2c.start()

    mo_coeff_occ = reduce(cupy.dot, (cupy.asarray(vhfopt.coeff), mo_coeff_occ))
    mo_coeff_unocc = reduce(
        cupy.dot, (cupy.asarray(
            vhfopt.coeff), mo_coeff_unocc))

    naux_cart, naux = vhfopt.auxcoeff.shape
    if isinstance(vhfopt.auxcoeff, cupy.ndarray):
        auxcoeff_cpu = cupyx.zeros_pinned(
            vhfopt.auxcoeff.shape, dtype='f8')
        vhfopt.auxcoeff.get(out=auxcoeff_cpu, blocking=True)
        vhfopt.auxcoeff = auxcoeff_cpu
        del auxcoeff_cpu

    if isinstance(vhfopt.coeff, cupy.ndarray):
        coeff_cpu = cupyx.zeros_pinned(
            vhfopt.coeff.shape, dtype='f8')
        vhfopt.coeff.get(out=coeff_cpu, blocking=True)
        vhfopt.coeff = coeff_cpu
        del coeff_cpu

    lib.free_all_blocks()
    gc.collect()

    buff_tmp = cupy.empty((blksize, nocc * blksize), dtype='f8')
    buff_eri_ov = cupy.empty((blksize, nocc * nvir), dtype='f8')
    buff_int3c = cupy.empty((blksize, blksize ** 2), dtype='f8')
    unLov = cupyx.empty_pinned((naux_cart, nocc * nvir), dtype='f8')

    GPU_buf_size = (buff_tmp.size + buff_eri_ov.size
                    + buff_int3c.size) * 8 / (1024 ** 3)
    CPU_buf_size = unLov.size * 8 / (1024 ** 3)
    logger.info(f'GPU buffer size {GPU_buf_size} GB.')
    logger.info(f'CPU buffer size {CPU_buf_size} GB.')
    nauxid = len(vhfopt.aux_log_qs)
    kslices = []
    kextents = []

    for cp_aux_id in range(nauxid):
        k0, k1 = vhfopt.auxmol.ao_loc[vhfopt.aux_l_ctr_offsets
                                      [cp_aux_id: cp_aux_id + 2]]
        kslices.append(slice(k0, k1))
        kextents.append(k1 - k0)

    for cp_aux_id in range(nauxid):
        logger.info(
            'eri_high_level_solver_incore, get int3c, cp_aux_id:%d/%d' %
            (cp_aux_id + 1, nauxid))
        sLov = lib.empty_from_buf(
            buff_eri_ov, (kextents[cp_aux_id], nocc, nvir), 'f8')
        sLov[:] = 0
        for cp_ij_id in range(len(vhfopt.log_qs)):
            si, sj, sk, int3c = get_int3c(
                cp_ij_id, cp_aux_id, vhfopt, buf=buff_int3c)
            tmp = lib.contraction(
                'ijL',
                int3c,
                'io',
                mo_coeff_occ[si],
                'Loj',
                buf=buff_tmp)
            lib.contraction(
                'Loj',
                tmp,
                'jv',
                mo_coeff_unocc[sj],
                'Lov',
                sLov,
                beta=1.0)
            if si != sj:
                tmp = lib.contraction(
                    'ijL', int3c, 'jo', mo_coeff_occ[sj], 'Loi', buf=buff_tmp)
                lib.contraction(
                    'Loi',
                    tmp,
                    'iv',
                    mo_coeff_unocc[si],
                    'Lov',
                    sLov,
                    beta=1.0)

        sLov.reshape(kextents[cp_aux_id], -1
                     ).get(out=unLov[kslices[cp_aux_id]], blocking=True)

    int3c = buff_int3c = buff_tmp = tmp = sLov = buff_eri_ov = mo_coeff_occ = mo_coeff_unocc = None
    lib.free_all_blocks()
    gc.collect()

    thread_load_j2c.join()
    f_j2c.close()

    if logger is not None:
        logger.info('----- solve to get cderi -----')

    auxcoeff = cupy.asarray(vhfopt.auxcoeff)
    del vhfopt
    lib.free_all_blocks()
    gc.collect()

    j2c = cupy.asarray(j2c)
    LL_svd = cupy.zeros(j2c.shape, order='F')
    avail_gpu_memory = lib.gpu_avail_bytes() / 8
    slice_len_ov = min(nocc * nvir,
                       int(avail_gpu_memory // (naux_cart + naux)))
    logger.info(
        '----- The slice_len_ov for cupy.linalg.solve: %s' %
        slice_len_ov)

    buff = cupy.empty((naux_cart, slice_len_ov), dtype='f8')
    buff2 = cupy.empty((naux * slice_len_ov), dtype='f8')
    ovL = cupyx.zeros_pinned((nocc * nvir, naux), dtype='f8')
    trans_buff_h = cupyx.empty_pinned(buff.shape, dtype='f8')

    logger.info(
        f'CPU buffer size : {(trans_buff_h.size + ovL.size) * 8 / (1024 ** 3)} GB')
    logger.info(
        f'GPU buffer size : {(buff.size + buff2.size + j2c.size) * 8 / (1024 ** 3)} GB')
    sov_list = [
        slice(
            i[0],
            i[1]) for i in prange(
            0,
            nocc * nvir,
            slice_len_ov)]
    for sov_ind, sov in enumerate(sov_list):
        logger.info(
            'eri_high_level_solver_incore, linalg.solve, for ovL, sov:%d/%d' %
            (sov_ind + 1, len(sov_list)))
        nov = sov.stop - sov.start
        unLovs = lib.empty_from_buf(buff, (naux_cart, nov), 'f8')

        unLovs_h = lib.empty_from_buf(trans_buff_h, unLovs.shape, 'f8')
        numpy.copyto(unLovs_h, unLov[:, sov])
        unLovs.set(unLovs_h)

        cupy.cuda.get_current_stream().synchronize()
        ovs_L = lib.gemm(unLovs, auxcoeff, buf=buff2, transa='T', transb='N')

        solve_triangular(j2c, ovs_L.T, lower=True, overwrite_b=True,)

        ovs_L.get(out=ovL[sov])

        lib.gemm(ovs_L, ovs_L, LL_svd, transa='T', transb='N', beta=1.0)

    cupy.cuda.get_current_stream().synchronize()
    unLov = unLovs = unLovs_h = ovs_L = buff = buff2 = trans_buff_h = j2c = None
    lib.free_all_blocks()
    gc.collect()

    logger.info('----- Start SVD to cut aux-basis')
    S, U_svd = cupy.linalg._eigenvalue._syevd(
        LL_svd, 'L', with_eigen_vector=True, overwrite_a=True)
    S = cupy.sqrt(abs(S))
    sort_ind = S.argsort()[::-1]
    S = S[sort_ind]
    U_svd = U_svd[:, sort_ind]
    newind = cupy.where(S > svd_tol)[0]
    assert newind.size > 0
    naux_cut = newind.size
    U_svd = cupy.ascontiguousarray(U_svd[:, newind])
    L_cd = lib.gemm(U_svd, U_svd, transa='T', transb='N')
    lib.linalg.cholesky(L_cd, overwrite=True)
    U_svd = lib.gemm(U_svd, L_cd, transa='N', transb='N')
    L_cd = S = LL_svd = None

    logger.info(f'----- SVD cut aux basis size from : {naux} to : {naux_cut}')

    avail_gpu_memory = lib.gpu_avail_bytes() / 8
    slice_len_ov = int(avail_gpu_memory / (naux_cut + naux))
    slice_len_ov = min(slice_len_ov, nocc * nvir)
    logger.info('----- The slice_len_ov for SVD cut: %s' % slice_len_ov)
    sov_list = [
        slice(
            i[0],
            i[1]) for i in prange(
            0,
            nocc * nvir,
            slice_len_ov)]

    if solver_type == 'MP2':
        cderi_cut = cupyx.empty_pinned((nocc * nvir, naux_cut), dtype='f8')
    elif 'CC' in solver_type:
        cderi_cut = cupyx.empty_pinned((naux_cut, nocc * nvir), dtype='f8')
    else:
        assert False, f'solver_type: {solver_type} is not supported'

    buff_ovL = cupy.empty((naux * slice_len_ov), dtype='f8')
    buff_cderi_cut = cupy.empty((naux_cut * slice_len_ov), dtype='f8')
    if 'CC' in solver_type:
        buff_cderi_cut_h = cupyx.empty_pinned(buff_cderi_cut.shape, dtype='f8')

    for sov_ind, sov in enumerate(sov_list):
        logger.info(
            'eri_high_level_solver_incore, SVD cut, for cderi_cut, sov:%d/%d' %
            (sov_ind + 1, len(sov_list)))
        nov = sov.stop - sov.start
        ovs_L = lib.empty_from_buf(buff_ovL, (nov, naux), 'f8')
        ovs_L.set(ovL[sov])

        if solver_type == 'MP2':
            cupy.cuda.get_current_stream().synchronize()
            cderi_cut_s = lib.gemm(
                ovs_L,
                U_svd,
                buf=buff_cderi_cut,
                transa='N',
                transb='N')
            cderi_cut_s.get(out=cderi_cut[sov])
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

    cupy.cuda.get_current_stream().synchronize()
    U_svd = buff_ovL = buff_cderi_cut = buff_cderi_cut_h = ovL = ovs_L = cderi_cut_s = cderi_cut_s_h = None
    lib.free_all_blocks()
    gc.collect()

    return cderi_cut


def eri_ondisk_high_level_solver_incore(
        mol, cderi_AO_file, mo_coeff_occ, mo_coeff_unocc, logger, solver_type='MP2', svd_tol=1e-4):
    '''
    Generate cderi for cluster which is under high-level solver
    processing by using the full system cderi on the disk.
    '''

    lib.free_all_blocks()
    gc.collect()

    load_process_num = lib.NumFileProcess

    mo_coeff_occ = cupy.asarray(mo_coeff_occ)
    mo_coeff_unocc = cupy.asarray(mo_coeff_unocc)

    if os.path.isdir(cderi_AO_file):
        j3c_type = 0
        cderi_AO_file_tmp = os.path.join(cderi_AO_file, '0-0')
        with h5py.File(cderi_AO_file_tmp, 'r') as f:
            naux = f['j3c'].shape[0]
    else:
        if getattr(mol, 'pbc_intor', None):
            j3c_type = 1
            with h5py.File(cderi_AO_file, 'r') as f:
                naux = f['j3c/0/0'].shape[0]
        else:
            j3c_type = 2
            with h5py.File(cderi_AO_file, 'r') as f:
                naux = f['j3c/0'].shape[0]

    nmo = mo_coeff_occ.shape[0]
    nocc = mo_coeff_occ.shape[1]
    nvir = mo_coeff_unocc.shape[1]

    free_size = lib.gpu_avail_bytes() / 8

    nmo_pair = int(nmo * (nmo - 1) / 2 + nmo)

    naux_slice_len = min(naux, free_size / (nmo ** 2
                                            + max(nmo_pair, nmo * nocc) + nocc * nvir))
    naux_slice_len = int(naux_slice_len)

    auxslice = [slice(i[0], i[1]) for i in prange(0, naux, naux_slice_len)]

    max_num_pair_slice_len = 0
    if j3c_type == 0:
        slice_ind_list = []
        for _, _, files in os.walk(cderi_AO_file):
            for filename in files:
                assert 'swap' not in filename
                assert '0' == filename[0]

                slice_ind_list.append(int(filename.split('-')[-1]))

                with h5py.File(os.path.join(cderi_AO_file, filename), 'r') as f:
                    max_num_pair_slice_len = max(
                        f['j3c'].shape[1], max_num_pair_slice_len)
        slice_ind_list.sort()
    elif j3c_type == 1:
        with h5py.File(cderi_AO_file, 'r') as f:
            assert len(list(f['j3c'].keys())) == 1
            j3c_0_slice = list(f['j3c/0'].keys())
            for s_j3c_0 in j3c_0_slice:
                max_num_pair_slice_len = max(
                    max_num_pair_slice_len, f[f'j3c/0/{s_j3c_0}'].shape[1])
        slice_ind_list = None

    elif j3c_type == 2:
        with h5py.File(cderi_AO_file, 'r') as f:
            j3c_0_slice = list(f['j3c'].keys())
            for s_j3c_0 in j3c_0_slice:
                max_num_pair_slice_len = max(
                    max_num_pair_slice_len, f[f'j3c/{s_j3c_0}'].shape[1])
        slice_ind_list = None
    else:
        assert False

    buffer_CPU_eri_load = cupyx.empty_pinned(
        (naux_slice_len * nmo_pair), dtype='f8')
    buffer_CPU_eri_load2 = cupyx.empty_pinned(
        (naux_slice_len * nocc * nvir), dtype='f8')
    eri_load_tmp = cupyx.empty_pinned(
        (naux_slice_len, max_num_pair_slice_len), dtype='f8')
    buffer_gpu_cderi = cupy.empty((naux_slice_len, nmo * nmo), dtype='f8')
    buffer_gpu_tmp = cupy.empty(
        (naux_slice_len * max(nmo_pair, nmo * nocc)), dtype='f8')
    buffer_gpu_save = cupy.empty((naux_slice_len, nocc * nvir), dtype='f8')

    ovL = cupyx.zeros_pinned((nocc * nvir, naux), dtype='f8')

    read_pool = multiprocessing.Pool(processes=load_process_num)
    cderi_next = None
    for sL_ind, sL in enumerate(auxslice):
        logger.info(
            'eri_ondisk_high_level_solver_incore, get int3c, cp_aux_id:%d/%d' %
            (sL_ind + 1, len(auxslice)))
        sL_len = sL.stop - sL.start
        if sL_ind == 0:
            cderi = lib.empty_from_buf(
                buffer_CPU_eri_load, (sL_len, nmo_pair), 'f8')
            process_r_cderi = []
            norb_taken_tmp = max(int(sL_len / load_process_num), 1)
            slice_load = [slice(i, min(i + norb_taken_tmp, sL.start + sL_len))
                          for i in range(sL.start, (sL.start + sL_len), norb_taken_tmp)]
            slice_load_r = [slice(i, min(i + norb_taken_tmp, sL_len))
                            for i in range(0, sL_len, norb_taken_tmp)]

            for sload_ind in range(len(slice_load)):
                s_load = slice_load[sload_ind]
                s_load_r = slice_load_r[sload_ind]
                process = read_pool.apply_async(
                    read_async_PBC, (j3c_type, cderi_AO_file, s_load, take_pointer_data_from_array(
                        cderi[s_load_r]), take_pointer_data_from_array(
                        eri_load_tmp[s_load_r]), slice_ind_list))
                process_r_cderi.append(process)

            for p in process_r_cderi:
                p.get()
        else:
            for p in process_r_cderi:
                p.get()

            cderi = cderi_next

        cderi_pack = lib.empty_from_buf(buffer_gpu_tmp, cderi.shape, 'f8')
        cderi_pack.set(cderi)
        cupy.cuda.get_current_stream().synchronize()

        if sL_ind + 1 < len(auxslice):
            sL_next = auxslice[sL_ind + 1]
            sL_len_next = min(
                naux - sL_next.start,
                sL_next.stop - sL_next.start)
            cderi_next = lib.empty_from_buf(
                buffer_CPU_eri_load, (sL_len_next, nmo_pair), 'f8')

            process_r_cderi = []
            norb_taken_tmp = max(int(sL_len_next / load_process_num), 1)
            slice_load = [slice(i, min(i + norb_taken_tmp, sL_next.start + sL_len_next))
                          for i in range(sL_next.start, (sL_next.start + sL_len_next), norb_taken_tmp)]
            slice_load_r = [slice(i, min(i + norb_taken_tmp, sL_len_next))
                            for i in range(0, sL_len_next, norb_taken_tmp)]

            for sload_ind in range(len(slice_load)):
                s_load = slice_load[sload_ind]
                s_load_r = slice_load_r[sload_ind]
                process = read_pool.apply_async(
                    read_async_PBC, (j3c_type, cderi_AO_file, s_load, take_pointer_data_from_array(
                        cderi_next[s_load_r]), take_pointer_data_from_array(
                        eri_load_tmp[s_load_r]), slice_ind_list))
                process_r_cderi.append(process)

        cderi = lib.empty_from_buf(buffer_gpu_cderi, (sL_len, nmo, nmo), 'f8')
        cderi_tmp_h = lib.empty_from_buf(
            buffer_CPU_eri_load2, (nocc * nvir, sL_len))

        lib.unpack_tril(cderi_pack, out=cderi)

        tmp = lib.contraction(
            'Lij',
            cderi,
            'io',
            mo_coeff_occ,
            'ojL',
            buf=buffer_gpu_tmp)
        ov_sL = lib.contraction(
            'ojL', tmp, 'jv', mo_coeff_unocc, 'ovL', buf=buffer_gpu_save).reshape(
            (nocc * nvir, sL_len))
        ov_sL.get(out=cderi_tmp_h, blocking=True)

        ovL[:, sL] = cderi_tmp_h

    read_pool.close()
    read_pool.join()
    buffer_CPU_eri_load = eri_load_tmp = buffer_gpu_cderi = buffer_gpu_tmp = buffer_gpu_save = None
    tmp = ov_sL = cderi = cderi_next = cderi_pack = None

    lib.free_all_blocks()
    gc.collect()

    logger.info('----- Start SVD to cut aux-basis')
    LL_svd = cupy.zeros((naux, naux), dtype='f8', order='F')
    avail_gpu_memory = lib.gpu_avail_bytes() / 8
    slice_len_ov = int(avail_gpu_memory / naux)
    slice_len_ov = min(slice_len_ov, nocc * nvir)

    ovslice_list = [
        slice(
            i[0],
            i[1]) for i in prange(
            0,
            nocc * nvir,
            slice_len_ov)]

    buffer_ovL = cupy.empty((slice_len_ov, naux), dtype='f8')

    for sov in ovslice_list:
        sov_len = sov.stop - sov.start
        sov_L = lib.empty_from_buf(buffer_ovL, (sov_len, naux), 'f8')
        sov_L.set(ovL[sov])
        lib.gemm(sov_L, sov_L, transa='T', c=LL_svd, beta=1.0)

    buffer_ovL = sov_L = None

    lib.free_all_blocks()
    gc.collect()

    S, U_svd = cupy.linalg._eigenvalue._syevd(
        LL_svd, 'L', with_eigen_vector=True, overwrite_a=True)
    S = cupy.sqrt(abs(S))
    sort_ind = S.argsort()[::-1]
    S = S[sort_ind]
    U_svd = U_svd[:, sort_ind]
    newind = cupy.where(S > svd_tol)[0]
    assert newind.size > 0
    naux_cut = newind.size
    U_svd = cupy.ascontiguousarray(U_svd[:, newind])
    L_cd = lib.gemm(U_svd, U_svd, transa='T', transb='N')
    lib.linalg.cholesky(L_cd, overwrite=True)
    U_svd = lib.gemm(U_svd, L_cd, transa='N', transb='N')
    L_cd = S = LL_svd = None

    logger.info(f'----- SVD cut aux basis size from : {naux} to : {naux_cut}')

    avail_gpu_memory = lib.gpu_avail_bytes() / 8
    slice_len_ov = int(avail_gpu_memory / (naux_cut + naux))
    slice_len_ov = min(slice_len_ov, nocc * nvir)
    logger.info('----- The slice_len_ov for SVD cut: %s' % slice_len_ov)
    sov_list = [
        slice(
            i[0],
            i[1]) for i in prange(
            0,
            nocc * nvir,
            slice_len_ov)]

    if solver_type == 'MP2':
        cderi_cut = cupyx.empty_pinned((nocc * nvir, naux_cut), dtype='f8')
    elif 'CC' in solver_type:
        cderi_cut = cupyx.empty_pinned((naux_cut, nocc * nvir), dtype='f8')
    else:
        assert False, f'solver_type: {solver_type} is not supported'

    buff_ovL = cupy.empty((naux * slice_len_ov), dtype='f8')
    buff_cderi_cut = cupy.empty((naux_cut * slice_len_ov), dtype='f8')
    if 'CC' in solver_type:
        buff_cderi_cut_h = cupyx.empty_pinned(buff_cderi_cut.shape, dtype='f8')

    for sov_ind, sov in enumerate(sov_list):
        logger.info(
            'eri_high_level_solver_incore, SVD cut, for cderi_cut, sov:%d/%d' %
            (sov_ind + 1, len(sov_list)))
        nov = sov.stop - sov.start
        ovs_L = lib.empty_from_buf(buff_ovL, (nov, naux), 'f8')
        ovs_L.set(ovL[sov])

        if solver_type == 'MP2':
            cupy.cuda.get_current_stream().synchronize()
            cderi_cut_s = lib.gemm(
                ovs_L,
                U_svd,
                buf=buff_cderi_cut,
                transa='N',
                transb='N')
            cderi_cut_s.get(out=cderi_cut[sov])
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

    cupy.cuda.get_current_stream().synchronize()
    U_svd = buff_ovL = buff_cderi_cut = buff_cderi_cut_h = ovL = ovs_L = cderi_cut_s = cderi_cut_s_h = None
    lib.free_all_blocks()
    gc.collect()

    return cderi_cut


def eri_high_level_solver_incore_with_jk(
        mol, auxmol, mo_coeff, j2c, logger, rdm1_core_coeff, vhfopt=None, svd_tol=1e-4):
    '''
    On-the-flying generate the cderi for CCSD, and obtain j and k at the meantime.
    '''
    mo_coeff = cupy.asarray(mo_coeff)
    nmo = mo_coeff.shape[1]
    rdm1_core_coeff = cupy.asarray(rdm1_core_coeff)
    nrdm_index = rdm1_core_coeff.shape[1]

    lib.free_all_blocks()
    gc.collect()
    blksize = MIN_GROUP_SIZE
    if vhfopt is None:
        logger.info('----- Build VHFOpt3c')
        vhfopt, blksize = build_vhfopt3c(mol, auxmol, blksize)
        logger.info('----- Build VHFOpt3c is done.')

    assert isinstance(
        j2c, str), 'Seems the j2c build in low_level_process is missed, please delete the logpath file and rerun.'

    f_j2c = h5py.File(j2c, 'r')
    j2c = cupyx.zeros_pinned(f_j2c['j2c'].shape, dtype='f8', order='F')

    def load_j2c(file_obj, j2c_buffer):
        j2c_buffer[:] = file_obj['j2c']

    thread_load_j2c = threading.Thread(target=load_j2c, args=(f_j2c, j2c, ))
    thread_load_j2c.start()

    mo_coeff = reduce(cupy.dot, (cupy.asarray(vhfopt.coeff), mo_coeff))
    rdm1_core_coeff = reduce(
        cupy.dot, (cupy.asarray(
            vhfopt.coeff), rdm1_core_coeff))

    naux_cart, naux = vhfopt.auxcoeff.shape
    if isinstance(vhfopt.auxcoeff, cupy.ndarray):
        auxcoeff_cpu = cupyx.zeros_pinned(
            vhfopt.auxcoeff.shape, dtype='f8')
        vhfopt.auxcoeff.get(out=auxcoeff_cpu, blocking=True)
        vhfopt.auxcoeff = auxcoeff_cpu
        del auxcoeff_cpu

    if isinstance(vhfopt.coeff, cupy.ndarray):
        coeff_cpu = cupyx.zeros_pinned(
            vhfopt.coeff.shape, dtype='f8')
        vhfopt.coeff.get(out=coeff_cpu, blocking=True)
        vhfopt.coeff = coeff_cpu
        del coeff_cpu

    lib.free_all_blocks()
    gc.collect()

    nauxid = len(vhfopt.aux_log_qs)
    kslices = []
    kextents = []
    for cp_aux_id in range(nauxid):
        k0, k1 = vhfopt.auxmol.ao_loc[vhfopt.aux_l_ctr_offsets
                                      [cp_aux_id: cp_aux_id + 2]]
        kslices.append(slice(k0, k1))
        kextents.append(k1 - k0)

    auxblk = max(kextents)

    buff_tmp = cupy.empty((blksize, nmo * blksize), dtype='f8')
    buff_rdm_tmp = cupy.empty((blksize * blksize), dtype='f8')
    buff_eri_d = cupy.empty((blksize, nmo ** 2), dtype='f8')
    buff_eri_vk_d = cupy.empty((blksize, nmo * nrdm_index), dtype='f8')
    buff_int3c = cupy.empty((blksize, blksize ** 2), dtype='f8')
    buff_eri_h = cupyx.empty_pinned((blksize, nmo ** 2), dtype='f8')
    buff_eri_vk_h = cupyx.empty_pinned((blksize, nmo * nrdm_index), dtype='f8')

    eri_vj = cupy.zeros((naux_cart), dtype='f8')
    tmpfile_path = os.path.join(TEMP_DIR, generate_tmpfile_name())
    os.makedirs(tmpfile_path)
    tmp_eri_file = lib.FileMp(os.path.join(tmpfile_path, 'tmperi.dat'), 'a')
    ij_unL_f = tmp_eri_file.create_dataset(
        'ijunL', (nmo ** 2, naux_cart), 'f8', blksizes=(nmo ** 2, auxblk))
    eri_vk_f = tmp_eri_file.create_dataset(
        'eri_vk',
        (nmo * nrdm_index,
         naux_cart),
        'f8',
        blksizes=(
            nmo * nrdm_index,
            auxblk))

    GPU_buf_size = (buff_tmp.size + buff_eri_d.size + buff_eri_vk_d.size
                    + eri_vj.size + buff_int3c.size) * 8 / (1024 ** 3)
    CPU_buf_size = (buff_eri_h.size + buff_eri_vk_h.size) * 8 / (1024 ** 3)
    logger.info(f'GPU buffer size {GPU_buf_size} GB.')
    logger.info(f'CPU buffer size {CPU_buf_size} GB.')

    # pool_w1 = multiprocessing.Pool(processes=1)
    # pool_w2 = multiprocessing.Pool(processes=1)
    w1 = w2 = None
    for cp_aux_id in range(nauxid):
        logger.info(
            'eri_high_level_solver_incore_with_jk, get int3c, cp_aux_id:%d/%d' %
            (cp_aux_id + 1, nauxid))
        ij_sL = lib.empty_from_buf(
            buff_eri_d, (nmo, nmo, kextents[cp_aux_id]), 'f8')
        eri_vk_s = lib.empty_from_buf(
            buff_eri_vk_d, (nrdm_index, nmo, kextents[cp_aux_id]), 'f8')
        ij_sL[:] = 0
        eri_vk_s[:] = 0
        for cp_ij_id in range(len(vhfopt.log_qs)):
            sp, sq, sk, int3c = get_int3c(
                cp_ij_id, cp_aux_id, vhfopt, buf=buff_int3c)
            tmp = lib.contraction(
                'pqL',
                int3c,
                'pi',
                mo_coeff[sp],
                'iqL',
                buf=buff_tmp)
            lib.contraction(
                'iqL',
                tmp,
                'qj',
                mo_coeff[sq],
                'ijL',
                ij_sL,
                beta=1.0)
            lib.contraction(
                'iqL',
                tmp,
                'qj',
                rdm1_core_coeff[sq],
                'jiL',
                eri_vk_s,
                beta=1.0)
            rdm1_tmp = lib.contraction(
                'pi',
                rdm1_core_coeff[sp],
                'qi',
                rdm1_core_coeff[sq],
                'pq',
                buf=buff_rdm_tmp)

            if sp != sq:
                lib.contraction(
                    'pqL',
                    int3c,
                    'pq',
                    rdm1_tmp,
                    'L',
                    eri_vj[sk],
                    beta=1.0,
                    alpha=2.0)
            else:
                lib.contraction(
                    'pqL',
                    int3c,
                    'pq',
                    rdm1_tmp,
                    'L',
                    eri_vj[sk],
                    beta=1.0,
                    alpha=1.0)

            if sp != sq:
                tmp = lib.contraction(
                    'pqL', int3c, 'qi', mo_coeff[sq], 'ipL', buf=buff_tmp)
                lib.contraction(
                    'ipL',
                    tmp,
                    'pj',
                    mo_coeff[sp],
                    'ijL',
                    ij_sL,
                    beta=1.0)
                lib.contraction(
                    'ipL',
                    tmp,
                    'pj',
                    rdm1_core_coeff[sp],
                    'jiL',
                    eri_vk_s,
                    beta=1.0)

        ij_sL_h = lib.empty_from_buf(buff_eri_h, ij_sL.shape, 'f8')
        eri_vk_s_h = lib.empty_from_buf(buff_eri_vk_h, eri_vk_s.shape, 'f8')

        if w1 is not None:
            for w in w1:
                w.wait()
        if w2 is not None:
            for w in w2:
                w.wait()

        ij_sL.get(out=ij_sL_h, blocking=True)
        eri_vk_s.get(out=eri_vk_s_h, blocking=True)

        ij_sL_h = ij_sL_h.reshape(-1, kextents[cp_aux_id])
        eri_vk_s_h = eri_vk_s_h.reshape(-1, kextents[cp_aux_id])

        # w1 = ij_unL_f.setitem(numpy.s_[:, sk], ij_sL_h, pool=pool_w1)
        # w2 = eri_vk_f.setitem(numpy.s_[:, sk], eri_vk_s_h, pool=pool_w2)
        w1 = ij_unL_f.setitem(numpy.s_[:, sk], ij_sL_h)
        w2 = eri_vk_f.setitem(numpy.s_[:, sk], eri_vk_s_h)

    for w in w1:
        w.wait()
    for w in w2:
        w.wait()

    # pool_w1.close()
    # pool_w1.join()
    # pool_w2.close()
    # pool_w2.join()

    buff_tmp = buff_rdm_tmp = buff_eri_d = buff_eri_vk_d = buff_int3c = buff_eri_h = buff_eri_vk_h = None
    ij_sL = eri_vk_s = int3c = tmp = rdm1_tmp = ij_sL_h = eri_vk_s_h = None

    lib.free_all_blocks()
    gc.collect()

    thread_load_j2c.join()
    f_j2c.close()

    if logger is not None:
        logger.info('----- solve to get cderi -----')

    auxcoeff = cupy.asarray(vhfopt.auxcoeff)
    eri_vj = lib.contraction('L', eri_vj, 'LK', auxcoeff, 'K').reshape(1, -1)
    del vhfopt
    lib.free_all_blocks()
    gc.collect()

    j2c = cupy.asarray(j2c)
    solve_triangular(j2c, eri_vj.T, lower=True, overwrite_b=True,)
    eri_vj = eri_vj.ravel()
    LL_svd = cupy.zeros(j2c.shape, order='F')

    avail_gpu_memory = lib.gpu_avail_bytes() / 8
    slice_len_i = min(max(nmo, nrdm_index), int(
        avail_gpu_memory // (nmo * (naux_cart + naux))))
    si_list = [slice(i[0], i[1]) for i in prange(0, nrdm_index, slice_len_i)]
    logger.info('----- The slice_len_i for get_vk: %s' % slice_len_i)

    buff_eri_d = cupy.empty((naux_cart, slice_len_i, nmo), dtype='f8')
    buff_eri_h = cupyx.empty_pinned(buff_eri_d.shape, dtype='f8')
    buff_tmp = cupy.empty((naux * slice_len_i * nmo), dtype='f8')
    vk_d = cupy.zeros((nmo, nmo), dtype='f8')
    buf_cpu_size = (buff_eri_h.size) * 8 / (1024 ** 3)
    buf_gpu_size = (buff_eri_d.size + buff_tmp.size
                    + j2c.size + LL_svd.size + vk_d.size) * 8 / (1024 ** 3)

    logger.info(f'CPU buffer size : {buf_cpu_size} GB')
    logger.info(f'GPU buffer size : {buf_gpu_size} GB')

    pool_r = multiprocessing.Pool(processes=lib.NumFileProcess)

    for si_ind, si in enumerate(si_list):
        logger.info(
            'eri_high_level_solver_incore_with_jk, get_vk, si:%d/%d' %
            (si_ind + 1, len(si_list)))
        ni = si.stop - si.start
        if si_ind == 0:
            sij_eri_vk_h = lib.empty_from_buf(
                buff_eri_h, (ni * nmo, naux_cart), 'f8')
            sij = slice(si.start * nmo, si.stop * nmo)
            r_list = eri_vk_f.getitem(
                numpy.s_[sij], pool=pool_r, buf=sij_eri_vk_h)

        for r in r_list:
            r.wait()

        sij_eri_vk_d = lib.empty_from_buf(
            buff_eri_d, (ni * nmo, naux_cart), 'f8')
        sij_eri_vk_d.set(sij_eri_vk_h)

        if si_ind + 1 < len(si_list):
            si_next = si_list[si_ind + 1]
            ni_next = si_next.stop - si_next.start
            sij_eri_vk_h = lib.empty_from_buf(
                buff_eri_h, (ni_next * nmo, naux_cart), 'f8')
            sij_next = slice(si_next.start * nmo, si_next.stop * nmo)
            r_list = eri_vk_f.getitem(
                numpy.s_[sij_next], pool=pool_r, buf=sij_eri_vk_h)

        sij_eri_vk_d = lib.gemm(
            sij_eri_vk_d,
            auxcoeff,
            buf=buff_tmp,
            transa='N',
            transb='N')
        solve_triangular(j2c, sij_eri_vk_d.T, lower=True, overwrite_b=True,)
        sij_eri_vk_d = sij_eri_vk_d.reshape(ni, nmo, naux)
        lib.contraction(
            'ipL',
            sij_eri_vk_d,
            'iqL',
            sij_eri_vk_d,
            'pq',
            vk_d,
            beta=1.0)

    pool_r.close()
    pool_r.join()
    vk = vk_d.get(blocking=True)
    sij_eri_vk_h = sij_eri_vk_d = vk_d = eri_vk_f = None

    del tmp_eri_file['eri_vk']
    lib.free_all_blocks()
    gc.collect()

    si_list = [slice(i[0], i[1]) for i in prange(0, nmo, slice_len_i)]
    logger.info(
        '----- The slice_len_i for get_vj and linalg.solve: %s' %
        slice_len_i)

    vj_d = cupy.zeros((nmo * nmo), dtype='f8')
    ijL = cupyx.zeros_pinned((nmo ** 2, naux), dtype='f8')
    buf_cpu_size = (ijL.size + buff_eri_h.size) * 8 / (1024 ** 3)
    buf_gpu_size = (buff_eri_d.size + j2c.size
                    + LL_svd.size + vj_d.size) * 8 / (1024 ** 3)

    logger.info(f'CPU buffer size : {buf_cpu_size} GB')
    logger.info(f'GPU buffer size : {buf_gpu_size} GB')

    pool_r = multiprocessing.Pool(processes=lib.NumFileProcess)

    for si_ind, si in enumerate(si_list):
        logger.info(
            'eri_high_level_solver_incore_with_jk, linalg.solve and get_vj, for ijL, si:%d/%d' %
            (si_ind + 1, len(si_list)))
        ni = si.stop - si.start
        if si_ind == 0:
            sij_unL_h = lib.empty_from_buf(
                buff_eri_h, (ni * nmo, naux_cart), 'f8')
            sij = slice(si.start * nmo, si.stop * nmo)
            r_list = ij_unL_f.getitem(
                numpy.s_[sij], pool=pool_r, buf=sij_unL_h)

        for r in r_list:
            r.wait()

        sij_unL_d = lib.empty_from_buf(buff_eri_d, (ni * nmo, naux_cart), 'f8')
        sij_unL_d.set(sij_unL_h)

        if si_ind + 1 < len(si_list):
            si_next = si_list[si_ind + 1]
            ni_next = si_next.stop - si_next.start
            sij_unL_h = lib.empty_from_buf(
                buff_eri_h, (ni_next * nmo, naux_cart), 'f8')
            sij_next = slice(si_next.start * nmo, si_next.stop * nmo)
            r_list = ij_unL_f.getitem(
                numpy.s_[sij_next], pool=pool_r, buf=sij_unL_h)

        sij_L_d = lib.gemm(
            sij_unL_d,
            auxcoeff,
            buf=buff_tmp,
            transa='N',
            transb='N')
        solve_triangular(j2c, sij_L_d.T, lower=True, overwrite_b=True,)
        sij = slice(si.start * nmo, si.stop * nmo)
        sij_L_d.get(out=ijL[sij])
        lib.contraction('iL', sij_L_d, 'L', eri_vj, 'i', vj_d[sij], beta=1.0)
        lib.gemm(sij_L_d, sij_L_d, LL_svd, transa='T', transb='N', beta=1.0)
        cupy.cuda.get_current_stream().synchronize()

    pool_r.close()
    pool_r.join()
    vj = vj_d.reshape(nmo, nmo).get(blocking=True)

    buff_eri_d = buff_eri_h = buff_tmp = None
    sij_unL_h = sij_unL_d = sij_unL_h = sij_L_d = eri_vj = j2c = vj_d = auxcoeff = None
    del tmp_eri_file['ijunL']
    tmp_eri_file.close()

    lib.free_all_blocks()
    gc.collect()

    logger.info('----- Start SVD to cut aux-basis')
    S, U_svd = cupy.linalg._eigenvalue._syevd(
        LL_svd, 'L', with_eigen_vector=True, overwrite_a=True)
    S = cupy.sqrt(abs(S))
    sort_ind = S.argsort()[::-1]
    S = S[sort_ind]
    U_svd = U_svd[:, sort_ind]
    newind = cupy.where(S > svd_tol)[0]
    assert newind.size > 0
    naux_cut = newind.size
    U_svd = cupy.ascontiguousarray(U_svd[:, newind])
    L_cd = lib.gemm(U_svd, U_svd, transa='T', transb='N')
    lib.linalg.cholesky(L_cd, overwrite=True)
    U_svd = lib.gemm(U_svd, L_cd, transa='N', transb='N')
    L_cd = S = LL_svd = None

    logger.info(f'----- SVD cut aux basis size from : {naux} to : {naux_cut}')

    avail_gpu_memory = lib.gpu_avail_bytes() / 8
    slice_len_ij = int(avail_gpu_memory / (naux_cut + naux))
    slice_len_ij = min(slice_len_ij, nmo ** 2)
    logger.info('----- The slice_len_ov for SVD cut: %s' % slice_len_ij)
    sij_list = [slice(i[0], i[1]) for i in prange(0, nmo ** 2, slice_len_ij)]

    cderi_cut = cupyx.empty_pinned((naux_cut, nmo ** 2), dtype='f8')

    buff_ijL = cupy.empty((naux * slice_len_ij), dtype='f8')
    buff_cderi_cut = cupy.empty((naux_cut * slice_len_ij), dtype='f8')
    buff_cderi_cut_h = cupyx.empty_pinned(buff_cderi_cut.shape, dtype='f8')

    for sij_ind, sij in enumerate(sij_list):
        logger.info(
            'eri_high_level_solver_incore_with_jk, SVD cut, for cderi_cut, sij:%d/%d' %
            (sij_ind + 1, len(sij_list)))
        nij = sij.stop - sij.start
        ijs_L = lib.empty_from_buf(buff_ijL, (nij, naux), 'f8')
        ijs_L.set(ijL[sij])

        cderi_cut_s = lib.gemm(
            U_svd,
            ijs_L,
            buf=buff_cderi_cut,
            transa='T',
            transb='T')
        cderi_cut_s_h = lib.empty_from_buf(
            buff_cderi_cut_h, cderi_cut_s.shape, 'f8')
        cderi_cut_s.get(out=cderi_cut_s_h, blocking=True)
        cderi_cut[:, sij] = cderi_cut_s_h

    U_svd = buff_ijL = buff_cderi_cut = buff_cderi_cut_h = ijL = ijs_L = cderi_cut_s = cderi_cut_s_h = None
    lib.free_all_blocks()
    gc.collect()

    return cderi_cut, vj, vk


def eri_ondisk_high_level_solver_incore_with_jk(
        mol, cderi_AO_file, mo_coeff, logger, rdm1_core_coeff, svd_tol=1e-4):
    '''
    Generate the cderi for CCSD by using the full system cderi on disk,
    and obtain j and k at the meantime.
    '''
    lib.free_all_blocks()
    gc.collect()

    load_process_num = lib.NumFileProcess

    mo_coeff = cupy.asarray(mo_coeff)
    rdm1_core_coeff = cupy.asarray(rdm1_core_coeff)
    rdm1_core = cupy.dot(rdm1_core_coeff, rdm1_core_coeff.T)

    if os.path.isdir(cderi_AO_file):
        j3c_type = 0
        cderi_AO_file_tmp = os.path.join(cderi_AO_file, '0-0')
        with h5py.File(cderi_AO_file_tmp, 'r') as f:
            naux = f['j3c'].shape[0]
    else:
        if getattr(mol, 'pbc_intor', None):
            j3c_type = 1
            with h5py.File(cderi_AO_file, 'r') as f:
                naux = f['j3c/0/0'].shape[0]
        else:
            j3c_type = 2
            with h5py.File(cderi_AO_file, 'r') as f:
                naux = f['j3c/0'].shape[0]

    nao = mo_coeff.shape[0]
    nmo = mo_coeff.shape[1]

    free_size = lib.gpu_avail_bytes() / 8

    nao_pair = int(nao * (nao - 1) / 2 + nao)

    naux_slice_len = min(naux, free_size / (nao ** 2
                                            + max(nao_pair, nao * nmo) + nmo ** 2))
    naux_slice_len = int(naux_slice_len)

    auxslice = [slice(i[0], i[1]) for i in prange(0, naux, naux_slice_len)]

    max_num_pair_slice_len = 0
    if j3c_type == 0:
        slice_ind_list = []
        for _, _, files in os.walk(cderi_AO_file):
            for filename in files:
                assert 'swap' not in filename
                assert '0' == filename[0]

                slice_ind_list.append(int(filename.split('-')[-1]))

                with h5py.File(os.path.join(cderi_AO_file, filename), 'r') as f:
                    max_num_pair_slice_len = max(
                        f['j3c'].shape[1], max_num_pair_slice_len)
        slice_ind_list.sort()

    elif j3c_type == 1:
        with h5py.File(cderi_AO_file, 'r') as f:
            assert len(list(f['j3c'].keys())) == 1
            j3c_0_slice = list(f['j3c/0'].keys())
            for s_j3c_0 in j3c_0_slice:
                max_num_pair_slice_len = max(
                    max_num_pair_slice_len, f[f'j3c/0/{s_j3c_0}'].shape[1])
        slice_ind_list = None

    elif j3c_type == 2:
        with h5py.File(cderi_AO_file, 'r') as f:
            j3c_0_slice = list(f['j3c'].keys())
            for s_j3c_0 in j3c_0_slice:
                max_num_pair_slice_len = max(
                    max_num_pair_slice_len, f[f'j3c/{s_j3c_0}'].shape[1])
        slice_ind_list = None
    else:
        assert False

    buffer_CPU_eri_load = cupyx.empty_pinned(
        (naux_slice_len * nao_pair), dtype='f8')
    buffer_CPU_eri_load2 = cupyx.empty_pinned(
        (naux_slice_len * nmo * nmo), dtype='f8')
    eri_load_tmp = cupyx.empty_pinned(
        (naux_slice_len, max_num_pair_slice_len), dtype='f8')
    buffer_gpu_cderi = cupy.empty((naux_slice_len, nao * nao), dtype='f8')
    buffer_gpu_tmp = cupy.empty(
        (naux_slice_len * max(nao_pair, nao * nmo)), dtype='f8')
    buffer_gpu_save = cupy.empty((naux_slice_len, nmo * nmo), dtype='f8')

    vj_d = cupy.zeros((nmo, nmo), dtype='f8')
    vk_d = cupy.zeros((nmo, nmo), dtype='f8')

    buffer_eri_vj_d = cupy.empty((naux_slice_len), dtype='f8')

    ijL = cupyx.zeros_pinned((nmo * nmo, naux), dtype='f8')

    read_pool = multiprocessing.Pool(processes=load_process_num)
    cderi_next = None
    for sL_ind, sL in enumerate(auxslice):
        logger.info(
            'eri_ondisk_high_level_solver_incore_with_jk, get int3c, cp_aux_id:%d/%d' %
            (sL_ind + 1, len(auxslice)))
        sL_len = sL.stop - sL.start
        if sL_ind == 0:
            cderi = lib.empty_from_buf(
                buffer_CPU_eri_load, (sL_len, nao_pair), 'f8')
            process_r_cderi = []
            norb_taken_tmp = max(int(sL_len / load_process_num), 1)
            slice_load = [slice(i, min(i + norb_taken_tmp, sL.start + sL_len))
                          for i in range(sL.start, (sL.start + sL_len), norb_taken_tmp)]
            slice_load_r = [slice(i, min(i + norb_taken_tmp, sL_len))
                            for i in range(0, sL_len, norb_taken_tmp)]

            for sload_ind in range(len(slice_load)):
                s_load = slice_load[sload_ind]
                s_load_r = slice_load_r[sload_ind]
                process = read_pool.apply_async(
                    read_async_PBC,
                    (
                        j3c_type, cderi_AO_file, s_load,
                        take_pointer_data_from_array(cderi[s_load_r]),
                        take_pointer_data_from_array(eri_load_tmp[s_load_r]),
                        slice_ind_list
                    )
                )
                process_r_cderi.append(process)

            for p in process_r_cderi:
                p.get()
        else:
            for p in process_r_cderi:
                p.get()

            cderi = cderi_next

        cderi_pack = lib.empty_from_buf(buffer_gpu_tmp, cderi.shape, 'f8')
        cderi_pack.set(cderi)
        cupy.cuda.get_current_stream().synchronize()

        if sL_ind + 1 < len(auxslice):
            sL_next = auxslice[sL_ind + 1]
            sL_len_next = min(
                naux - sL_next.start,
                sL_next.stop - sL_next.start)
            cderi_next = lib.empty_from_buf(
                buffer_CPU_eri_load, (sL_len_next, nao_pair), 'f8')

            process_r_cderi = []
            norb_taken_tmp = max(int(sL_len_next / load_process_num), 1)
            slice_load = [slice(i, min(i + norb_taken_tmp, sL_next.start + sL_len_next))
                          for i in range(sL_next.start, (sL_next.start + sL_len_next), norb_taken_tmp)]
            slice_load_r = [slice(i, min(i + norb_taken_tmp, sL_len_next))
                            for i in range(0, sL_len_next, norb_taken_tmp)]

            for sload_ind in range(len(slice_load)):
                s_load = slice_load[sload_ind]
                s_load_r = slice_load_r[sload_ind]
                process = read_pool.apply_async(
                    read_async_PBC, (j3c_type, cderi_AO_file, s_load, take_pointer_data_from_array(
                        cderi_next[s_load_r]), take_pointer_data_from_array(
                        eri_load_tmp[s_load_r]), slice_ind_list))
                process_r_cderi.append(process)

        cderi = lib.empty_from_buf(buffer_gpu_cderi, (sL_len, nao, nao), 'f8')
        cderi_tmp_h = lib.empty_from_buf(
            buffer_CPU_eri_load2, (nmo * nmo, sL_len))

        lib.unpack_tril(cderi_pack, out=cderi)

        eri_vj_d_tmp = lib.contraction(
            'Lpq',
            cderi,
            'pq',
            rdm1_core,
            'L',
            buf=buffer_eri_vj_d).ravel()

        tmp = lib.contraction(
            'Lpq',
            cderi,
            'pi',
            mo_coeff,
            'iqL',
            buf=buffer_gpu_tmp)
        ij_sL = lib.contraction(
            'iqL', tmp, 'qj', mo_coeff, 'ijL', buf=buffer_gpu_save).reshape(
            (nmo * nmo, sL_len))
        ij_sL.get(out=cderi_tmp_h, blocking=True)
        ij_sL = ij_sL.reshape((nmo, nmo, sL_len))
        lib.contraction('ijL', ij_sL, 'L', eri_vj_d_tmp, 'ij', vj_d, beta=1.0)
        eri_vk_d_tmp = lib.contraction(
            'iqL',
            tmp,
            'qk',
            rdm1_core_coeff,
            'ikL',
            buf=buffer_gpu_cderi)
        lib.contraction(
            'ikL',
            eri_vk_d_tmp,
            'jkL',
            eri_vk_d_tmp,
            'ij',
            vk_d,
            beta=1.0)
        ijL[:, sL] = cderi_tmp_h

    read_pool.close()
    read_pool.join()
    vj = vj_d.get(blocking=True)
    vk = vk_d.get(blocking=True)
    buffer_CPU_eri_load = eri_load_tmp = buffer_gpu_cderi \
        = buffer_gpu_tmp = buffer_gpu_save = buffer_eri_vj_d = None
    tmp = ij_sL = cderi = cderi_next = cderi_pack = \
        eri_vk_d_tmp = eri_vj_d_tmp = vj_d = vk_d = None
    lib.free_all_blocks()
    gc.collect()

    logger.info('----- Start SVD to cut aux-basis')
    LL_svd = cupy.zeros((naux, naux), dtype='f8', order='F')
    avail_gpu_memory = lib.gpu_avail_bytes() / 8
    slice_len_ij = int(avail_gpu_memory / naux)
    slice_len_ij = min(slice_len_ij, nmo ** 2)

    ijslice_list = [
        slice(
            i[0],
            i[1]) for i in prange(
            0,
            nmo ** 2,
            slice_len_ij)]

    buffer_ijL = cupy.empty((slice_len_ij, naux), dtype='f8')

    for sij in ijslice_list:
        sij_len = sij.stop - sij.start
        sij_L = lib.empty_from_buf(buffer_ijL, (sij_len, naux), 'f8')
        sij_L.set(ijL[sij])
        lib.gemm(sij_L, sij_L, transa='T', c=LL_svd, beta=1.0)

    buffer_ijL = sij_L = None

    lib.free_all_blocks()
    gc.collect()

    S, U_svd = cupy.linalg._eigenvalue._syevd(
        LL_svd, 'L', with_eigen_vector=True, overwrite_a=True)
    S = cupy.sqrt(abs(S))
    sort_ind = S.argsort()[::-1]
    S = S[sort_ind]
    U_svd = U_svd[:, sort_ind]
    newind = cupy.where(S > svd_tol)[0]
    assert newind.size > 0
    naux_cut = newind.size
    U_svd = cupy.ascontiguousarray(U_svd[:, newind])
    L_cd = lib.gemm(U_svd, U_svd, transa='T', transb='N')
    lib.linalg.cholesky(L_cd, overwrite=True)
    U_svd = lib.gemm(U_svd, L_cd, transa='N', transb='N')
    L_cd = S = LL_svd = None

    logger.info(f'----- SVD cut aux basis size from : {naux} to : {naux_cut}')

    avail_gpu_memory = lib.gpu_avail_bytes() / 8
    slice_len_ij = int(avail_gpu_memory / (naux_cut + naux))
    slice_len_ij = min(slice_len_ij, nmo ** 2)
    logger.info('----- The slice_len_ov for SVD cut: %s' % slice_len_ij)
    sij_list = [slice(i[0], i[1]) for i in prange(0, nmo ** 2, slice_len_ij)]

    cderi_cut = cupyx.empty_pinned((naux_cut, nmo * nmo), dtype='f8')

    buff_ijL = cupy.empty((naux * slice_len_ij), dtype='f8')
    buff_cderi_cut = cupy.empty((naux_cut * slice_len_ij), dtype='f8')
    buff_cderi_cut_h = cupyx.empty_pinned(buff_cderi_cut.shape, dtype='f8')

    for sij_ind, sij in enumerate(sij_list):
        logger.info(
            'eri_ondisk_high_level_solver_incore_with_jk, SVD cut, for cderi_cut, sij:%d/%d' %
            (sij_ind + 1, len(sij_list)))
        nij = sij.stop - sij.start
        ijs_L = lib.empty_from_buf(buff_ijL, (nij, naux), 'f8')
        ijs_L.set(ijL[sij])

        cderi_cut_s = lib.gemm(
            U_svd,
            ijs_L,
            buf=buff_cderi_cut,
            transa='T',
            transb='T')  # may cause problem
        cderi_cut_s_h = lib.empty_from_buf(
            buff_cderi_cut_h, cderi_cut_s.shape, 'f8')
        cderi_cut_s.get(out=cderi_cut_s_h, blocking=True)
        cderi_cut[:, sij] = cderi_cut_s_h

    cupy.cuda.get_current_stream().synchronize()
    U_svd = buff_ijL = buff_cderi_cut = buff_cderi_cut_h = ijL = ijs_L = cderi_cut_s = cderi_cut_s_h = None
    lib.free_all_blocks()
    gc.collect()

    return cderi_cut, vj, vk
