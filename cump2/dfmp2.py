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
from byteqc.lib import Mg, MemoryTypeHost, gemm, contraction
from pyscf import df
from pyscf.lib import logger, prange, param
from byteqc.cuobc.lib.int3c import VHFOpt3c, get_int2c, get_int3c
from multiprocessing import Pool
import os


def kernel(mol, rhf, auxbasis=None, verbose=None, cleanfile=True, with_rdm1=False):
    if verbose is None:
        verbose = mol.verbose
    log = logger.new_logger(rhf, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    nocc = mol.nelectron // 2
    coeff_o = rhf.mo_coeff[:, :nocc]
    coeff_v = rhf.mo_coeff[:, nocc:]
    nvir = coeff_v.shape[1]

    if auxbasis is None:
        auxbasis = df.make_auxbasis(mol, mp2fit=True)
    auxmol = df.make_auxmol(mol, auxbasis)

    memory = lib.gpu_avail_bytes() // 8
    a = nvir ** 2
    b = auxmol.nao * nvir * 2
    c = -1 * memory
    ngpu = lib.Mg.ngpu

    oblk = int(min(
        (-1 * b + numpy.sqrt((b ** 2 - 4 * a * c))) / (2 * a),
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

    assert oblk > 0, 'No enough GPU memory (%f.2GB) to perform MP2 '\
        'calculations' % memory
    log.info('MP2 with %.2fGB free memory, slice nocc(%d) to %d. nvir:%d' % (
        memory * 8 / 1e9, nocc, oblk, nvir))

    path, oslices = cderi_ovL_outcore(
        mol, auxmol, coeff_o, coeff_v, oblk, vblk, log=log)
    time1 = log.timer('cderi_ovL_outcore', *time0)

    lib.Mg.mapgpu(lambda: lib.free_all_blocks())
    mo_energy = rhf.mo_energy.copy()
    vslices = [slice(i[0], i[1]) for i in prange(0, nvir, vblk)]
    e_corr, rdm1 = mp2_get_corr(
        mol, path, oslices, nvir, nocc, auxmol.nao, mo_energy, log=log, with_rdm1=with_rdm1, vslices=vslices)
    time1 = log.timer('mp2_get_corr', *time1)
    log.timer('mp2', *time0)

    log.info(f'MP2 total energy: {e_corr + rhf.e_tot}')
    log.info(f'MP2 correlation energy: {e_corr}')

    if with_rdm1:
        log.info(f'MP2 rdm1 is obtained.')

    if cleanfile:
        file = lib.FileMp(path + '/eris.dat', 'r+')
        del file['eri']
        del file['cderi']
        file.close()

    if with_rdm1:
        return e_corr, e_corr + rhf.e_tot, rdm1
    else:
        return e_corr, e_corr + rhf.e_tot


def div_t2(t2, A, B, C, D):
    '''Divide the t2 tensor by energies in-place.'''
    kernel = cupy.ElementwiseKernel(
        'raw T A, raw T B, raw T C, raw T D, int64 w, int64 z, int64 y',
        'T out',
        '''
        size_t p = i / y;
        size_t q = (i % y) / z;
        size_t r = (i % z) / w;
        size_t s = i % w;

        double denom = A[p] - B[q] + C[r] - D[s];
        if (denom != 0) {
            out /= denom;
        } else {
            out = 0;
        }
        ''',
        'custom_division_kernel')

    kernel(A, B, C, D, t2.shape[-1], t2.shape[-1] * t2.shape[-2],
           t2.shape[-3] * t2.shape[-2] * t2.shape[-1], t2)

    return t2


def cderi_ovL_outcore(mol, auxmol, coeff_o, coeff_v,
                      oblk, vblk, log=None, path=None, save_j2c=False):
    '''Store the ERI in the desired format in disk.'''
    if log is None:
        log = logger.new_logger(mol, mol.verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    nocc = coeff_o.shape[1]
    nvir = coeff_v.shape[1]
    nmin = min(nocc, nvir)

    nauxblk = 500
    Mg.mapgpu(lambda: lib.free_all_blocks())
    memory = lib.gpu_avail_bytes(0.5) / 8
    c = nvir * nocc * nauxblk - memory
    b = nauxblk * min(nocc, nvir)
    a = nauxblk
    naoblk = (-1 * b + numpy.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    naoblk = min(int(naoblk), mol.nao)
    assert naoblk > 0, 'No enough GPU memory (currently %f.2GB) to ' \
        'perform MP2 calculations' % memory

    vhfopt = VHFOpt3c(mol, auxmol, 'int2e')
    vhfopt.build(group_size=naoblk, aux_group_size=nauxblk)
    nauxid = len(vhfopt.aux_log_qs)
    j2c = get_int2c(vhfopt=vhfopt)

    coeff_o = vhfopt.coeff.dot(cupy.asarray(coeff_o))
    coeff_v = vhfopt.coeff.dot(cupy.asarray(coeff_v))
    coeff_o, coeff_v = Mg.broadcast(coeff_o, coeff_v)
    Mg.mapgpu(lambda: lib.free_all_blocks())

    kslices = []
    kextents = []
    for cp_aux_id in range(nauxid):
        k0, k1 = vhfopt.auxmol.ao_loc[vhfopt.aux_l_ctr_offsets
                                      [cp_aux_id: cp_aux_id + 2]]
        kslices.append(slice(k0, k1))
        kextents.append(k1 - k0)

    nauxblk = max(kextents)
    naux_cart, naux = vhfopt.auxcoeff.shape

    oslices = [slice(*i) for i in prange(0, nocc, oblk)]

    if path is None:
        path = tempfile.NamedTemporaryFile(dir=param.TMPDIR).name
    os.mkdir(path)
    file = lib.FileMp(path + '/eris.dat', 'w')
    eri = file.create_dataset(
        'eri', (naux_cart, nocc, nvir), 'f8', blksizes=(kextents, oblk))
    cderi = file.create_dataset('cderi', (nocc, nvir, naux), 'f8',
                                blksizes=(oblk, vblk))
    if save_j2c:
        file.create_dataset('j2c', (naux, naux), 'f8', data=j2c.get())
    file.close()

    int3cs = Mg.mapgpu(lambda: cupy.empty((nauxblk, naoblk, naoblk), 'f8'))
    eris_d = Mg.mapgpu(lambda: cupy.empty((nauxblk, nocc, nvir), 'f8'))
    bufs = Mg.mapgpu(lambda: cupy.empty((nauxblk, naoblk, nmin, ), 'f8'))
    ngpu = Mg.ngpu
    eris_h = [lib.empty((nauxblk, nocc, nvir), 'f8', type=MemoryTypeHost)
              for _ in range(ngpu)]
    pools = [Pool(processes=lib.NumFileProcess) for _ in range(ngpu)]
    waits = [None] * ngpu
    time0 = log.timer("cderi_ovL_outcore prepare", *time0)

    def eri_gen_OVL(cp_aux_id):
        gid = Mg.getgid()
        time1 = logger.process_clock(), logger.perf_counter()
        eri_d = lib.empty_from_buf(eris_d[gid],
                                   (kextents[cp_aux_id], nocc, nvir), 'f8')
        eri_d[:] = 0
        eri_h = lib.empty_from_buf(eris_h[gid], eri_d.shape, 'f8')

        for cp_ij_id in range(len(vhfopt.log_qs)):
            si, sj, sk, int3c = get_int3c(
                cp_ij_id, cp_aux_id, vhfopt, buf=int3cs[gid])

            if nvir > nocc:
                tmp = contraction('ijL', int3c, 'io', coeff_o[gid][si],
                                  'ojL', buf=bufs[gid])
                contraction('ojL', tmp, 'jv', coeff_v[gid][sj], 'Lov',
                            eri_d, beta=1.0)
            else:
                tmp = contraction('ijL', int3c, 'jv', coeff_v[gid][sj],
                                  'ivL', buf=bufs[gid])
                contraction('ivL', tmp, 'io', coeff_o[gid][si], 'Lov',
                            eri_d, beta=1.0)

            if si != sj:
                if nvir > nocc:
                    tmp = contraction('ijL', int3c, 'jo', coeff_o[gid][sj],
                                      'oiL', buf=bufs[gid])
                    contraction('oiL', tmp, 'iv', coeff_v[gid][si], 'Lov',
                                eri_d, beta=1.0)
                else:
                    tmp = contraction('ijL', int3c, 'iv', coeff_v[gid][si],
                                      'jvL', buf=bufs[gid])
                    contraction('jvL', tmp, 'jo', coeff_o[gid][sj], 'Lov',
                                eri_d, beta=1.0)
        tmp = None
        if waits[gid]:
            for w in waits[gid]:
                w.wait()

        eri_d.get(out=eri_h, blocking=True)
        waits[gid] = eri.setitem(numpy.s_[sk, :], eri_h, pool=pools[gid])
        log.timer(
            'cderi_ovL_outcore cp_aux_id:%d/%d on GPU%d' % (
                cp_aux_id + 1, nauxid, Mg.gpus[gid]), *time1)

    Mg.map(eri_gen_OVL, range(nauxid))
    for wait in waits:
        if wait:
            for w in wait:
                w.wait()
    time0 = log.timer("cderi_ovL_outcore step1", *time0)

    auxcoeff = cupy.asarray(vhfopt.auxcoeff.T, order='C')
    auxcoeff, j2c = Mg.broadcast(auxcoeff, j2c)
    for p in pools:
        p.terminate()
        p.join()
    vhfopt = int3cs = eris_d = bufs = eris_h = pools = waits = None
    Mg.mapgpu(lambda: lib.free_all_blocks())

    cderis_d = Mg.mapgpu(lambda: cupy.empty((naux, oblk, nvir), 'f8'))
    eris_d = Mg.mapgpu(lambda: cupy.empty((naux_cart, oblk, nvir), 'f8'))
    cderis_h = [lib.empty((naux, oblk, nvir), 'f8', type=MemoryTypeHost)
                for _ in range(ngpu)]
    eris_h = [lib.empty((naux_cart, oblk, nvir), 'f8', type=MemoryTypeHost)
              for _ in range(ngpu)]
    waits = [None] * ngpu
    pools_r = [Pool(processes=lib.NumFileProcess) for _ in range(ngpu)]
    pools_w = [Pool(processes=lib.NumFileProcess) for _ in range(ngpu)]

    def get_cderi(so):
        gid = Mg.getgid()
        time1 = logger.process_clock(), logger.perf_counter()

        so_len = so.stop - so.start
        eri_h = lib.empty_from_buf(
            eris_h[gid], (naux_cart, so_len, nvir), 'f8')

        eri.getitem(numpy.s_[:, so], pool=pools_r[gid],
                    buf=eri_h).wait()

        eri_d = lib.empty_from_buf(eris_d[gid], eri_h.shape)
        eri_d.set(eri_h)
        cderi_d = contraction('Lov', eri_d, 'KL', auxcoeff[gid], 'ovK',
                              buf=cderis_d[gid]).reshape(-1, naux)
        lib.solve_triangular(j2c[gid], cderi_d.T, overwrite_b=True,
                             lower=True)

        if waits[gid]:
            for w in waits[gid]:
                w.wait()
        cderi_h = lib.empty_from_buf(cderis_h[gid], cderi_d.shape)
        cderi_d.get(out=cderi_h, blocking=True)
        cderi_h = cderi_h.reshape(so_len, nvir, naux)

        waits[gid] = cderi.setitem(numpy.s_[so], cderi_h, pool=pools_w[gid])
        log.timer('cderi_ovL_outcore nao:[%d:%d]/%d on GPU%s' %
                  (so.start, so.stop, nocc, Mg.gpus[gid]), *time1)

    Mg.map(get_cderi, oslices)
    for wait in waits:
        if wait:
            for w in wait:
                w.wait()
    for p in pools_r:
        p.terminate()
        p.join()
    for p in pools_w:
        p.terminate()
        p.join()
    cderis_d = eris_d = cderis_h = eris_h = waits = pools_r = pools_w = None
    time0 = log.timer("cderi_ovL_outcore step2", *time0)
    return path, oslices


def mp2_get_corr(mol, path, oslices, nvir, nocc, naux, e_mo, log=None, with_rdm1=False, vslices=None):
    if log is None:
        log = logger.new_logger(mol, mol.verbose)

    oblk = oslices[0].stop - oslices[0].start

    if with_rdm1:
        assert vslices, 'with_rdm1 and vslices must be provided together! ' \
            'with_rdm1 : %s, vslices : %s' % (with_rdm1, vslices)

    file = lib.FileMp(path + '/eris.dat', 'r')
    cderi = file['cderi']
    file.close()

    e_mos = Mg.broadcast(e_mo)

    ias_d = Mg.mapgpu(lambda: cupy.empty((oblk, nvir, naux), 'f8'))
    jbs_d = Mg.mapgpu(lambda: cupy.empty((oblk, nvir, naux), 'f8'))

    if with_rdm1:
        tau_d = Mg.mapgpu(lambda: cupy.empty((oblk ** 2 * nvir ** 2), 'f8'))

    if with_rdm1:
        rdm1_vir_d = Mg.mapgpu(lambda: cupy.zeros((nvir, nvir), 'f8'))

    t2s = Mg.mapgpu(lambda: cupy.empty((oblk ** 2, nvir ** 2), 'f8'))
    ngpu = Mg.ngpu
    ias_h = [lib.empty((oblk, nvir, naux), 'f8', type=MemoryTypeHost)
             for _ in range(ngpu)]
    jbs_h = [lib.empty((oblk, nvir, naux), 'f8', type=MemoryTypeHost)
             for _ in range(ngpu)]
    pools = [Pool(processes=lib.NumFileProcess) for _ in range(ngpu)]

    def MP2_kernel(o_ind):
        gid = Mg.getgid()
        time0 = logger.process_clock(), logger.perf_counter()
        so = oslices[o_ind]

        e_mo_o = e_mos[gid][:nocc]
        e_mo_v = e_mos[gid][nocc:]

        so_len = so.stop - so.start
        ia_h = cderi.getitem(numpy.s_[so], pool=pools[gid], buf=ias_h[gid])
        # `reshape` will triger `wait()`
        ia_h.wait()
        ia_d = lib.empty_from_buf(ias_d[gid], ia_h.shape, 'f8')
        ia_d.set(ia_h)

        if o_ind + 1 < len(oslices):
            so2 = oslices[o_ind + 1]
            jb_h = cderi.getitem(numpy.s_[so2], pool=pools[gid],
                                 buf=jbs_h[gid])

        t2 = gemm(ia_d.reshape((-1, naux)), ia_d.reshape((-1, naux)),
                  transb='T', buf=t2s[gid]).reshape(
            (so_len, nvir, so_len, nvir))
        div_t2(t2, e_mo_o[so], e_mo_v, e_mo_o[so], e_mo_v)
        tmp = contraction('iajb', t2, 'iaL', ia_d, 'jbL', buf=jbs_d[gid],
                          alpha=2.0)
        contraction('iajb', t2, 'ibL', ia_d, 'jaL', tmp, beta=1.0,
                    alpha=-1.0)
        e_corr = tmp.ravel().dot(ia_d.ravel()).item()

        if with_rdm1:
            tau = lib.empty_from_buf(tau_d[gid], t2.shape, 'f8')
            cupy.copyto(tau, t2)
            tau *= 2
            tau -= t2.transpose(0, 3, 2, 1)
            contraction('iajc', t2, 'ibjc', tau, 'ab', rdm1_vir_d[gid],
                        alpha=2.0, beta=1.0)

        oslices2 = oslices[o_ind + 1:]
        for o_ind2, so2 in enumerate(oslices2):
            so_len2 = so2.stop - so2.start
            # `reshape` will triger `wait()`
            jb_h.wait()
            jb_d = lib.empty_from_buf(jbs_d[gid], jb_h.shape, 'f8')
            jb_d.set(jb_h)
            cupy.cuda.Device().synchronize()

            if o_ind2 + 1 < len(oslices2):
                so3 = oslices2[o_ind2 + 1]
                jb_h = cderi.getitem(numpy.s_[so3], pool=pools[gid],
                                     buf=jbs_h[gid])

            t2 = gemm(ia_d.reshape((-1, naux)), jb_d.reshape((-1, naux)),
                      transb='T', buf=t2s[gid]).reshape(
                (so_len, nvir, so_len2, nvir))
            div_t2(t2, e_mo_o[so], e_mo_v, e_mo_o[so2], e_mo_v)
            tmp = contraction('iajb', t2, 'jbL', jb_d, 'iaL', buf=ias_d[gid],
                              alpha=2.0)
            contraction('iajb', t2, 'jaL', jb_d, 'ibL', tmp, beta=1.0,
                        alpha=-1.0)

            ia_d2 = lib.empty_from_buf(jbs_d[gid], ia_h.shape, 'f8')
            ia_d2.set(ia_h)
            e_corr += tmp.ravel().dot(ia_d2.ravel()).item() * 2

            cupy.copyto(ia_d, ia_d2)

            if with_rdm1:
                tau = lib.empty_from_buf(tau_d[gid], t2.shape, 'f8')
                cupy.copyto(tau, t2)
                tau *= 2
                tau -= t2.transpose(0, 3, 2, 1)

                contraction('iajc', t2, 'ibjc', tau, 'ab', rdm1_vir_d[gid],
                            alpha=2.0, beta=1.0)
                contraction('icja', t2, 'icjb', tau, 'ab', rdm1_vir_d[gid],
                            alpha=2.0, beta=1.0)

                tau = None

        log.timer('mp2_kernel nao:[%d:%d]/%d on GPU%s' %
                  (so.start, so.stop, nocc, Mg.gpus[gid]), *time0)

        return e_corr

    e_corr_list = Mg.map(MP2_kernel, range(len(oslices)))
    ias_d = jbs_d = t2s = ias_h = jbs_h = pools = tau_d = e_mos = None

    e_corr = numpy.sum(e_corr_list).item()

    if with_rdm1:
        rdm1_vir_list = Mg.map(cupy.asnumpy, rdm1_vir_d)
        rdm1_vir = numpy.sum(rdm1_vir_list, axis=0)
        rdm1_vir_list = None

        rdm1_occ = mp2_get_occ_1rdm(path, vslices, nvir, nocc, naux, e_mo, log)

        nao = nocc + nvir
        rdm1 = lib.empty((nao, nao), 'f8', type=MemoryTypeHost)
        rdm1[:] = 0
        rdm1[:nocc, :nocc] = rdm1_occ + numpy.eye(nocc) * 2
        rdm1[nocc:, nocc:] = rdm1_vir
    else:
        rdm1 = None

    return e_corr, rdm1


def mp2_get_occ_1rdm(path, vslices, nvir, nocc, naux, e_mo, log):

    time0 = logger.process_clock(), logger.perf_counter()
    vblk = vslices[0].stop - vslices[0].start

    file = lib.FileMp(path + '/eris.dat', 'r')
    cderi = file['cderi']
    file.close()

    e_mos = Mg.broadcast(e_mo)

    ias_d = Mg.mapgpu(lambda: cupy.empty((nocc, vblk, naux), 'f8'))
    jbs_d = Mg.mapgpu(lambda: cupy.empty((nocc, vblk, naux), 'f8'))

    tau_d = Mg.mapgpu(lambda: cupy.empty((vblk ** 2 * nocc ** 2), 'f8'))
    rdm1_occ_d = Mg.mapgpu(lambda: cupy.zeros((nocc, nocc), 'f8'))

    t2s = Mg.mapgpu(lambda: cupy.empty((nocc ** 2, vblk ** 2), 'f8'))
    ngpu = Mg.ngpu
    ias_h = [lib.empty((vblk, nocc, naux), 'f8', type=MemoryTypeHost)
             for _ in range(ngpu)]
    jbs_h = [lib.empty((vblk, nocc, naux), 'f8', type=MemoryTypeHost)
             for _ in range(ngpu)]
    pools = [Pool(processes=lib.NumFileProcess) for _ in range(ngpu)]
    log.timer('mp2_get_occ_1rdm prepare', *time0)

    def MP2_occ_1rdm_kernel(v_ind):
        gid = Mg.getgid()
        time0 = logger.process_clock(), logger.perf_counter()
        sv = vslices[v_ind]

        e_mo_o = e_mos[gid][:nocc]
        e_mo_v = e_mos[gid][nocc:]

        sv_len = sv.stop - sv.start
        ia_h = cderi.getitem(numpy.s_[:, sv], pool=pools[gid], buf=ias_h[gid])
        ia_h.wait()
        ia_d = lib.empty_from_buf(ias_d[gid], ia_h.shape, 'f8')
        ia_d.set(ia_h)

        if v_ind + 1 < len(vslices):
            sv2 = vslices[v_ind + 1]
            jb_h = cderi.getitem(numpy.s_[:, sv2], pool=pools[gid],
                                 buf=jbs_h[gid])

        t2 = gemm(ia_d.reshape((-1, naux)), ia_d.reshape((-1, naux)),
                  transb='T', buf=t2s[gid]).reshape(
            (nocc, sv_len, nocc, sv_len))
        div_t2(t2, e_mo_o, e_mo_v[sv], e_mo_o, e_mo_v[sv])

        tau = lib.empty_from_buf(tau_d[gid], t2.shape, 'f8')
        cupy.copyto(tau, t2)
        tau *= 2
        tau -= t2.transpose(2, 1, 0, 3)
        contraction('iakb', t2, 'jakb', tau, 'ij', rdm1_occ_d[gid],
                    alpha=-2.0, beta=1.0)

        vslices2 = vslices[v_ind + 1:]
        for v_ind2, sv2 in enumerate(vslices2):
            sv_len2 = sv2.stop - sv2.start
            jb_h.wait()
            jb_d = lib.empty_from_buf(jbs_d[gid], jb_h.shape, 'f8')
            jb_d.set(jb_h)
            cupy.cuda.Device().synchronize()

            if v_ind2 + 1 < len(vslices2):
                sv3 = vslices2[v_ind2 + 1]
                jb_h = cderi.getitem(numpy.s_[:, sv3], pool=pools[gid],
                                     buf=jbs_h[gid])

            t2 = gemm(ia_d.reshape((-1, naux)), jb_d.reshape((-1, naux)),
                      transb='T', buf=t2s[gid]).reshape(
                (nocc, sv_len, nocc, sv_len2))
            div_t2(t2, e_mo_o, e_mo_v[sv], e_mo_o, e_mo_v[sv2])

            tau = lib.empty_from_buf(tau_d[gid], t2.shape, 'f8')
            cupy.copyto(tau, t2)
            tau *= 2
            tau -= t2.transpose(2, 1, 0, 3)

            contraction('iakb', t2, 'jakb', tau, 'ij', rdm1_occ_d[gid],
                        alpha=-2.0, beta=1.0)
            contraction('kaib', t2, 'kajb', tau, 'ij', rdm1_occ_d[gid],
                        alpha=-2.0, beta=1.0)

        log.timer('mp2_get_occ_1rdm nao:[%d:%d]/%d on GPU%s' %
                  (sv.start, sv.stop, nvir, Mg.gpus[gid]), *time0)

    Mg.map(MP2_occ_1rdm_kernel, range(len(vslices)))
    rdm1_occ_list = Mg.map(cupy.asnumpy, rdm1_occ_d)
    rdm1_occ = numpy.sum(rdm1_occ_list, axis=0)
    ias_d = jbs_d = t2s = ias_h = jbs_h = pools = tau_d = e_mos = rdm1_occ_list = None

    return rdm1_occ
