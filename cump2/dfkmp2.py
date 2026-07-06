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

import tempfile
import numpy
import cupy
from byteqc import lib
from byteqc.lib import Mg, MemoryTypeHost, contraction, unpack_tril, \
    arr2ptr, ptr2arr
from byteqc.cump2.dfmp2 import mp2_get_corr
from pyscf import df
from pyscf.lib import logger, prange, param
from multiprocessing import Pool
import os
import h5py


def kernel(mol, rhf, auxbasis=None, verbose=None,
           cleanfile=True, cderi_path=None, with_rdm1=False):
    if verbose is None:
        verbose = mol.verbose
    log = logger.new_logger(rhf, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    nocc = mol.nelectron // 2
    coeff_o = rhf.mo_coeff[:, :nocc].copy()
    coeff_v = rhf.mo_coeff[:, nocc:].copy()
    nvir = coeff_v.shape[1]

    if auxbasis is None:
        auxbasis = df.make_auxbasis(mol, mp2fit=True)
    auxmol = df.make_auxmol(mol, auxbasis)

    if cderi_path is None:
        cderi_path = rhf.with_df._cderi_to_save

    assert isinstance(cderi_path, str), 'cderi_path must be a string, but ' \
        'got  %s' % type(cderi_path)

    memory = lib.gpu_avail_bytes() // 8
    a = nvir ** 2
    b = auxmol.nao * nvir * 2
    c = -1 * memory
    ngpu = lib.Mg.ngpu
    oblk = int(min((-1 * b + numpy.sqrt((b ** 2 - 4 * a * c))) / (2 * a),
                   nocc / ngpu))
    if with_rdm1:
        oblk_rdm = int(min(
            (-1 * (b / 2) + numpy.sqrt(((b / 2) ** 2 - 2 * 4 * a * c))) / (2 * 2 * a),
            nocc / ngpu))
        oblk = min(oblk, oblk_rdm)

        a = nocc ** 2
        b = auxmol.nao * nocc * 2
        c = -1 * memory

        vblk = int(min(
            (-1 * (b / 2) + numpy.sqrt(((b / 2) ** 2 - 2 * 4 * a * c))) / (2 * 2 * a),
            nvir / ngpu))

        assert vblk > 0, 'No enough GPU memory (%f.2GB) to perform MP2 1-RDM '\
            'calculations' % memory
    else:
        vblk = nvir

    assert oblk > 0, 'No enough GPU memory (%f.2GB) to perform MP2 ' \
        'calculations' % memory
    log.info('MP2 with %.2fGB free memory, slice nocc(%d) to %d. nvir:%d' % (
        memory * 8 / 1e9, nocc, oblk, nvir))

    path, oslices = cderi_ovL_gamma_point_outcore(
        mol, auxmol, cderi_path, coeff_o, coeff_v, oblk, vblk, log=log)
    time1 = log.timer('cderi_ovL_outcore', *time0)

    lib.Mg.mapgpu(lambda: lib.free_all_blocks())
    mo_energy = rhf.mo_energy.copy()
    vslices = [slice(i[0], i[1]) for i in prange(0, nvir, vblk)]
    e_corr, rdm1 = mp2_get_corr(
        mol, path, oslices, nvir, nocc, auxmol.nao, mo_energy, log=log, with_rdm1=with_rdm1, vslices=vslices)
    time1 = log.timer('mp2_get_corr', *time1)
    log.timer('mp2', *time0)

    if with_rdm1:
        log.info(f'MP2 rdm1 is obtained.')

    log.info(f'MP2 total energy: {e_corr + rhf.e_tot}')
    log.info(f'MP2 correlation energy: {e_corr}')

    if cleanfile:
        file = lib.FileMp(path + '/eris.dat', 'r+')
        del file['cderi']
        file.close()

    if with_rdm1:
        return e_corr, e_corr + rhf.e_tot, rdm1
    else:
        return e_corr, e_corr + rhf.e_tot


def cderi_ovL_gamma_point_outcore(mol, auxmol, eri_path, coeff_o, coeff_v,
                                  oblk, vblk, log=None, path=None):
    '''Store the ERI in the desired format in disk for gamma point.'''
    if log is None:
        log = logger.new_logger(mol, mol.verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    nocc = coeff_o.shape[1]
    nvir = coeff_v.shape[1]
    nmin = min(nocc, nvir)
    nmo = coeff_o.shape[0]
    nmo_pair = int(nmo * (nmo - 1) / 2 + nmo)
    naux = auxmol.nao

    Mg.mapgpu(lambda: lib.free_all_blocks())
    memory = lib.gpu_avail_bytes(0.5) / 8

    nauxblk = min(int(memory / (nmo * nmo + max(nmo_pair, nmo * nmin))),
                  int(naux / lib.NumFileProcess) + 1)

    assert nauxblk > 0, 'No enough GPU memory (currently %f.2GB) to ' \
        'perform MP2 calculations' % memory

    coeff_o, coeff_v = Mg.broadcast(coeff_o, coeff_v)
    Mg.mapgpu(lambda: lib.free_all_blocks())

    Lslices = [slice(*i) for i in prange(0, naux, nauxblk)]

    max_nmo_pair_slice_len = 0

    assert os.path.exists(
        eri_path), 'The eri file %s is not exist.' % (eri_path)

    if os.path.isdir(eri_path):
        # suit for self-developed pyscf.pbc.scf, will be remove in the future
        slice_ind_list = []
        for _, _, files in os.walk(eri_path):
            for filename in files:
                assert 'swap' not in filename
                assert '0' == filename[0]

                slice_ind_list.append(int(filename.split('-')[-1]))

                with h5py.File(os.path.join(eri_path, filename), 'r') as f:
                    max_nmo_pair_slice_len = max(
                        f['j3c'].shape[1], max_nmo_pair_slice_len)

        slice_ind_list.sort()

    else:
        # suit for pyscf.pbc.scf gamma point HF eri
        with h5py.File(eri_path, 'r') as f:
            assert len(list(f['j3c'].keys())) == 1, \
                'Only support the gamma point cderi, while there are %d k ' \
                'points in the cderi file.' \
                % (len(list(f['j3c'].keys())))
            j3c_0_slice = list(f['j3c/0'].keys())
            for s_j3c_0 in j3c_0_slice:
                max_nmo_pair_slice_len = max(
                    max_nmo_pair_slice_len, f[f'j3c/0/{s_j3c_0}'].shape[1])

        slice_ind_list = None

    oslices = [slice(*i) for i in prange(0, nocc, oblk)]

    if path is None:
        path = tempfile.NamedTemporaryFile(dir=param.TMPDIR).name
    os.mkdir(path)
    file = lib.FileMp(path + '/eris.dat', 'w')

    cderi = file.create_dataset('cderi', (nocc, nvir, naux), 'f8',
                                blksizes=(oblk, vblk, nauxblk))
    file.close()

    ngpu = Mg.ngpu
    eri_buffer_h = [lib.empty((nauxblk, nmo_pair), 'f8', type=MemoryTypeHost)
                    for _ in range(ngpu)]
    eri_read_buffer_h = [lib.empty((nauxblk, max_nmo_pair_slice_len), 'f8',
                                   type=MemoryTypeHost) for _ in range(ngpu)]
    cderi_buffer_h = [lib.empty((nauxblk, nocc * nvir), 'f8',
                                type=MemoryTypeHost) for _ in range(ngpu)]

    eri_buffer_Lmm_d = Mg.mapgpu(lambda: cupy.empty(
        (nauxblk * nmo * nmo), 'f8'))
    eri_buffer_Lom_d = Mg.mapgpu(lambda: cupy.empty(
        (nauxblk * max(nmo_pair, nmo * nmin)), 'f8'))

    read_pools = [Pool(processes=lib.NumFileProcess) for _ in range(ngpu)]
    write_pools = [Pool(processes=lib.NumFileProcess) for _ in range(ngpu)]
    write_waits = [None] * ngpu
    time0 = log.timer("cderi_ovL_gamma_point_outcore prepare", *time0)

    def eri_gen_OVL(aux_id_list):
        gid = Mg.getgid()
        time1 = logger.process_clock(), logger.perf_counter()

        for sL_ind_t, sL_ind in enumerate(aux_id_list):
            sL = Lslices[sL_ind]
            sL_len = min(naux - sL.start, sL.stop - sL.start)
            if sL_ind_t == 0:
                eri_packed_h = lib.empty_from_buf(
                    eri_buffer_h[gid], (sL_len, nmo_pair), 'f8')
                process_read_eri = []
                sL_len_tmp = max(int(sL_len / lib.NumFileProcess), 1)
                slice_load = [
                    slice(i, min(i + sL_len_tmp, sL.start + sL_len))
                    for i in range(sL.start, (sL.start + sL_len), sL_len_tmp)]
                slice_load_inplace = [
                    slice(i, min(i + sL_len_tmp, sL_len))
                    for i in range(0, sL_len, sL_len_tmp)]
                for sL_read_ind in range(len(slice_load)):
                    s_read = slice_load[sL_read_ind]
                    s_read_inplace = slice_load_inplace[sL_read_ind]
                    process = read_pools[gid].apply_async(read_eri_async, (
                        eri_path, s_read,
                        arr2ptr(eri_packed_h[s_read_inplace]),
                        arr2ptr(eri_read_buffer_h[gid][s_read_inplace]),
                        slice_ind_list))
                    process_read_eri.append(process)

                for p in process_read_eri:
                    p.wait()

                process_read_eri = process = slice_load_inplace = None
                slice_load = sL_len_tmp = None
            else:
                for p in process_read_eri:
                    p.wait()
                process_read_eri = None

            eri_packed_d = lib.empty_from_buf(
                eri_buffer_Lom_d[gid], eri_packed_h.shape, 'f8')
            eri_packed_d.set(eri_packed_h)
            cupy.cuda.Device().synchronize()

            if sL_ind_t + 1 < len(aux_id_list):
                sL_next = Lslices[aux_id_list[sL_ind_t + 1]]
                sL_len_next = min(naux - sL_next.start,
                                  sL_next.stop - sL_next.start)
                eri_packed_h = lib.empty_from_buf(
                    eri_buffer_h[gid], (sL_len_next, nmo_pair), 'f8')

                process_read_eri = []
                sL_len_tmp = max(int(sL_len_next / lib.NumFileProcess), 1)
                slice_load = [
                    slice(i, min(i + sL_len_tmp, sL_next.start + sL_len_next))
                    for i in range(sL_next.start,
                                   (sL_next.start + sL_len_next), sL_len_tmp)]
                slice_load_inplace = [
                    slice(i, min(i + sL_len_tmp, sL_len_next))
                    for i in range(0, sL_len_next, sL_len_tmp)]
                for sL_read_ind in range(len(slice_load)):
                    s_read = slice_load[sL_read_ind]
                    s_read_inplace = slice_load_inplace[sL_read_ind]
                    process = read_pools[gid].apply_async(read_eri_async, (
                        eri_path, s_read,
                        arr2ptr(eri_packed_h[s_read_inplace]),
                        arr2ptr(eri_read_buffer_h[gid][s_read_inplace]),
                        slice_ind_list))
                    process_read_eri.append(process)

                slice_load_inplace = slice_load = sL_len_tmp = None

            eri_d = lib.empty_from_buf(
                eri_buffer_Lmm_d[gid], (sL_len, nmo, nmo), 'f8')
            eri_d[:] = 0
            unpack_tril(eri_packed_d, out=eri_d)

            if nocc < nvir:
                tmp = contraction('Lij', eri_d, 'io', coeff_o[gid], 'oLj',
                                  buf=eri_buffer_Lom_d[gid])
                cderi_d = contraction('oLj', tmp, 'jv', coeff_v[gid], 'ovL',
                                      buf=eri_buffer_Lmm_d[gid])
            else:
                tmp = contraction('Lij', eri_d, 'jv', coeff_v[gid], 'vLi',
                                  buf=eri_buffer_Lom_d[gid])
                cderi_d = contraction('vLi', tmp, 'io', coeff_o[gid], 'ovL',
                                      buf=eri_buffer_Lmm_d[gid])

            if write_waits[gid]:
                for w in write_waits[gid]:
                    w.wait()

            cderi_h = lib.empty_from_buf(
                cderi_buffer_h[gid], cderi_d.shape, 'f8')
            cderi_d.get(out=cderi_h, blocking=True)
            cderi_h = cderi_h.reshape(nocc, nvir, cderi_h.shape[2])
            tmp = cderi_d = None
            write_waits[gid] = cderi.setitem(
                numpy.s_[:, :, sL], cderi_h, pool=write_pools[gid])
            log.timer(
                'cderi_ovL_gamma_point_outcore aux_id:%d/%d on GPU%d' % (
                    sL_ind + 1, len(Lslices), Mg.gpus[gid]), *time1)

    tmp = list(range(len(Lslices)))
    sL_list = [tmp[gpu_id::ngpu] for gpu_id in range(ngpu)]

    Mg.map(eri_gen_OVL, sL_list)

    for wait in write_waits:
        if wait:
            for w in wait:
                w.wait()

    time0 = log.timer("cderi_ovL_gamma_point_outcore is done", *time0)

    for gid in range(ngpu):
        read_pools[gid].close()
        read_pools[gid].join()
        write_pools[gid].close()
        write_pools[gid].join()

    eri_buffer_h = eri_read_buffer_h = cderi_buffer_h \
        = eri_buffer_Lmm_d = eri_buffer_Lom_d = read_pools \
        = write_pools = write_waits = None

    Mg.mapgpu(lambda: lib.free_all_blocks())

    return path, oslices


def read_eri_async(filepath, sv, x_pointer, buf_tmp_pointer,
                   slice_ind_list=None):
    x = ptr2arr(x_pointer)
    buf_tmp = ptr2arr(buf_tmp_pointer)
    if os.path.isdir(filepath):
        assert slice_ind_list is not None
        start_j3c = 0
        for s_j3c_0 in slice_ind_list:
            s_j3c_0 = str(s_j3c_0)
            with h5py.File(os.path.join(filepath, f'0-{s_j3c_0}'), 'r') as f:
                s_j3c_0_len = f['j3c'].shape[1]
                x_tmp = lib.empty_from_buf(
                    buf_tmp, (x.shape[0], s_j3c_0_len), 'f8')
                f['j3c'].read_direct(x_tmp, source_sel=sv)
                s_j3c_w = slice(start_j3c, start_j3c + s_j3c_0_len)
                x[:, s_j3c_w] = x_tmp
                start_j3c += s_j3c_0_len

    else:
        with h5py.File(filepath, 'r') as f:
            j3c_0_slice = list(f['j3c/0'].keys())
            j3c_0_slice = numpy.asarray([int(i) for i in j3c_0_slice])
            j3c_0_slice = j3c_0_slice[j3c_0_slice.argsort()]
            start_j3c = 0
            for s_j3c_0 in j3c_0_slice:
                s_j3c_0 = str(s_j3c_0)
                s_j3c_0_len = f[f'j3c/0/{s_j3c_0}'].shape[1]
                x_tmp = lib.empty_from_buf(
                    buf_tmp, (x.shape[0], s_j3c_0_len), 'f8')
                f[f'j3c/0/{s_j3c_0}'].read_direct(x_tmp, source_sel=sv)
                s_j3c_w = slice(start_j3c, start_j3c + s_j3c_0_len)
                x[:, s_j3c_w] = x_tmp
                start_j3c += s_j3c_0_len
