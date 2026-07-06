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

import numpy
import cupy
try:
    from gpu4pyscf import scf as gpuscf
except ImportError:
    print("The dfmp2_grad is a advance feature of ByteQC.\n",
          "To use it, gpu4pyscf is required.\n",
          "Please install gpu4pyscf by using the command\n",
          "------------------------- \n",
          "pip install --no-deps \n",
          "   gpu4pyscf-cuda12x==1.3.1 \n",
          "   pyscf==2.8.0 \n",
          "   pyscf-dispersion==1.3.0 \n",
          "   geometric==1.1 \n",
          "   gpu4pyscf-libxc-cuda12x==0.6 \n",
          "   networkx==3.4.2 \n",
          "------------------------- \n",)
    raise ImportError
from byteqc import lib
from byteqc.lib import Mg, MemoryTypeHost, gemm, contraction
from pyscf import df
from pyscf.lib import logger, prange
from byteqc.cump2.dfmp2 import cderi_ovL_outcore, div_t2, mp2_get_occ_1rdm
from multiprocessing import Pool
from byteqc.cuobc.lib.int3c import VHFOpt3c, get_int3c
from gpu4pyscf.grad.rhf import _jk_energy_per_atom
from gpu4pyscf.df import int3c2e
from functools import reduce
from itertools import product


get_de_int3c_nao_gs = 128
get_de_int3c_naux_gs = 256


def kernel(mol, rhf, auxbasis=None, verbose=None, cleanfile=True):
    if verbose is None:
        verbose = mol.verbose
    log = logger.new_logger(rhf, verbose)
    time0 = logger.process_clock(), logger.perf_counter()
    if hasattr(rhf, 'with_df'):
        rhf = rhf.undo_df()
    nocc = mol.nelectron // 2
    coeff_o = rhf.mo_coeff[:, :nocc]
    coeff_v = rhf.mo_coeff[:, nocc:]
    nvir = coeff_v.shape[1]
    nao, nmo = rhf.mo_coeff.shape

    if auxbasis is None:
        auxbasis = df.make_auxbasis(mol, mp2fit=True)
    auxmol = df.make_auxmol(mol, auxbasis)
    naux = auxmol.nao
    memory = lib.gpu_avail_bytes(ratio=0.8) // 8
    ngpu = lib.Mg.ngpu

    # esitimate the blk of occ, vir, aux
    a = nvir ** 2 * 2
    b = naux * nvir * 3
    c = -1 * (memory - nvir ** 2)

    oblk1 = int(min(
        (-1 * b + numpy.sqrt((b ** 2 - 4 * a * c))) / (2 * a),
        nocc / ngpu))

    c = -1 * (memory - nvir ** 2 - naux ** 2)
    a = nvir ** 2

    oblk2 = int(min(
        (-1 * b + numpy.sqrt((b ** 2 - 4 * a * c))) / (2 * a),
        nocc / ngpu))

    oblk3 = int(
        min(
            (memory - nao ** 2 - naux ** 2 - 3 * naux
             - 4 * 3 * get_de_int3c_naux_gs * (get_de_int3c_nao_gs ** 2))
            / (naux * nao * 4),
            nocc / ngpu
        )
    )

    oblk = min(oblk1, oblk2, oblk3)

    a = nocc ** 2 * 2
    b = naux * nocc * 2
    c = -1 * (memory - nocc ** 2)

    vblk = int(min(
        (-1 * b + numpy.sqrt((b ** 2 - 4 * a * c))) / (2 * a),
        nvir / ngpu))

    auxblk1 = int((memory - nao ** 2 * 5 - naux ** 2)
                  / (nao ** 2 + nao * max(nocc, nvir) + get_de_int3c_nao_gs ** 2))

    auxblk2 = int((memory - nao ** 2 - 3 * nao - naux ** 2
                  - (4 * 3 + 1) * get_de_int3c_naux_gs * (get_de_int3c_nao_gs ** 2))
                  / (nocc * nvir + get_de_int3c_nao_gs * nocc + get_de_int3c_nao_gs ** 2))

    auxblk = min(int(naux / ngpu), auxblk1, auxblk2)

    log.info('MP2 with %.2fGB free memory, slice nocc(%d) to %d, slice nvir(%d) to %d slice naux(%d) to %d' % (
        memory * 8 / 1e9, nocc, oblk, nvir, vblk, naux, auxblk))
    assert oblk > 0 and vblk > 0 and auxblk > 0, 'No enough GPU memory (%f.2GB) to perform MP2 '\
        'calculations' % (memory * 8 / (1024 ** 3))

    path, oslices = cderi_ovL_outcore(
        mol, auxmol, coeff_o, coeff_v, oblk, vblk, log=log, save_j2c=True)
    log.timer('cderi_ovL_outcore', *time0)
    lib.Mg.mapgpu(lambda: lib.free_all_blocks())
    mo_energy = cupy.asarray(rhf.mo_energy)
    vslices = [slice(i[0], i[1]) for i in prange(0, nvir, vblk)]
    auxslices = [slice(i[0], i[1]) for i in prange(0, naux, auxblk)]
    e_corr, doo, dvv, gamma_2c = _grad_intermediates(
        mol, path, vslices, oslices, auxslices, nvir, nocc, auxmol.nao, mo_energy, log=log)
    lib.Mg.mapgpu(lambda: lib.free_all_blocks())
    de = get_de_int3c(mol,
                      auxmol,
                      path,
                      auxslices,
                      oslices,
                      coeff_o,
                      coeff_v,
                      log=log)
    lib.Mg.mapgpu(lambda: lib.free_all_blocks())
    I_mat = get_I_mat(mol, auxmol, path, auxslices, coeff_o, coeff_v, log=log)

    lib.Mg.mapgpu(lambda: lib.free_all_blocks())
    ao_ovlp = cupy.asarray(rhf.get_ovlp())
    aomo_coeff = cupy.asarray(rhf.mo_coeff)
    I_mat = reduce(cupy.dot, (aomo_coeff.T, I_mat, ao_ovlp, aomo_coeff)) * -1
    ao_ovlp = None
    dm1mo = cupy.zeros((nmo, nmo))
    dm1mo[:nocc, :nocc] = cupy.asarray(doo)
    dm1mo[nocc:, nocc:] = cupy.asarray(dvv)
    doo = dvv = None

    dm1 = reduce(cupy.dot, (aomo_coeff, dm1mo, aomo_coeff.T))
    j, k = gpuscf.jk.get_jk(mol, dm1)
    vhf = 2 * j - k
    j = k = None
    Xvo = reduce(
        cupy.dot,
        (cupy.asarray(coeff_v.T),
         vhf,
         cupy.asarray(coeff_o)))
    vhf = None
    Xvo += cupy.asarray(I_mat[:nocc, nocc:].T - I_mat[nocc:, :nocc])
    dm1mo += _response_dm1(rhf, cupy.asarray(aomo_coeff), Xvo)
    Xvo = None
    I_mat[nocc:, :nocc] = I_mat[:nocc, nocc:].T
    im1 = reduce(cupy.dot, (aomo_coeff, I_mat, aomo_coeff.T))
    I_mat = None

    if hasattr(rhf, 'to_cpu'):
        gpu_rhf = rhf
        grad_rhf = rhf.to_cpu().nuc_grad_method()
    else:
        gpu_rhf = rhf.to_gpu()
        grad_rhf = rhf.nuc_grad_method()
    s1 = grad_rhf.get_ovlp(mol)
    hcore_deriv = grad_rhf.hcore_generator(mol)

    zeta = mo_energy.reshape((-1, 1)) + mo_energy.reshape((1, -1))
    zeta *= 0.5
    zeta[nocc:, :nocc] = mo_energy[:nocc]
    zeta[:nocc, nocc:] = mo_energy[:nocc].reshape(-1, 1)
    zeta = reduce(cupy.dot, (aomo_coeff, zeta * dm1mo, aomo_coeff.T))
    gpu_grad_rhf = grad_rhf.to_gpu()
    dm1 = reduce(cupy.dot, (aomo_coeff, dm1mo, aomo_coeff.T))
    p1_tmp = cupy.dot(cupy.asarray(coeff_o), cupy.asarray(coeff_o.T))
    vhf_s1occ = reduce(cupy.dot, (p1_tmp,
                                  gpu_rhf.get_veff(mol, dm1 + dm1.T),
                                  p1_tmp))
    dm1mo = p1_tmp = None
    hf_dm1 = cupy.asarray(rhf.make_rdm1())
    dm1p = hf_dm1 + dm1 * 2
    dm1 += hf_dm1
    zeta += gpu_grad_rhf.make_rdm1e(mo_energy,
                                    aomo_coeff,
                                    cupy.asarray(rhf.mo_occ))
    aomo_coeff = mo_energy = None
    int2c2e_ip1 = auxmol.intor('int2c2e_ip1')

    dm1_mix = hf_dm1 + dm1p
    de += _jk_energy_per_atom(mol, dm1_mix)
    de -= _jk_energy_per_atom(mol, dm1p)
    de -= _jk_energy_per_atom(mol, hf_dm1)
    dm1_mix = dm1p = hf_dm1 = None
    offsetdic = mol.offset_nr_by_atom()
    aux_offsetdic = auxmol.offset_nr_by_atom()
    for k, ia in enumerate(list(range(mol.natm))):
        _, _, p0, p1 = offsetdic[ia]
        sa = slice(p0, p1)
        _, _, ap0, ap1 = aux_offsetdic[ia]
        asa = slice(ap0, ap1)

        de[k] += cupy.einsum('xij,ij->x', s1[:, sa], im1[sa])
        de[k] += cupy.einsum('xji,ij->x', s1[:, sa], im1[:, sa])

        h1ao = cupy.asarray(hcore_deriv(ia))
        de[k] += cupy.einsum('xij,ji->x', h1ao, dm1)

        de[k] -= cupy.einsum('xij,ij->x', s1[:, sa], zeta[sa])
        de[k] -= cupy.einsum('xji,ij->x', s1[:, sa], zeta[:, sa])

        de[k] -= cupy.einsum('xij,ij->x', s1[:, sa], vhf_s1occ[sa]) * 2

        de[k] += cupy.einsum('RS, xRS -> x',
                             gamma_2c[asa], int2c2e_ip1[:, asa])
        de[k] += cupy.einsum('SR, xRS -> x',
                             gamma_2c[:, asa], int2c2e_ip1[:, asa])
    s1 = im1 = h1ao = hcore_deriv = zeta = None
    vhf_s1occ = gamma_2c = int2c2e_ip1 = dm1 = None
    lib.Mg.mapgpu(lambda: lib.free_all_blocks())

    de += cupy.asarray(grad_rhf.grad_nuc())
    lib.Mg.mapgpu(lambda: lib.free_all_blocks())

    log.info(f'MP2 total energy: {e_corr + rhf.e_tot}')
    log.info(f'MP2 correlation energy: {e_corr}')
    log.info('--------------- RI-MP2 gradients ---------------')
    log.info('            x                y                z')
    for k, ia in enumerate(list(range(mol.natm))):
        log.info('%4d %2s  %15.10f  %15.10f  %15.10f' %
                 (ia, mol.atom_symbol(ia), de[k, 0], de[k, 1], de[k, 2]))

    if cleanfile:
        file = lib.FileMp(path + '/eris.dat', 'r+')
        del file['eri']
        del file['cderi']
        del file['j2c']
        del file['gamma_3c']
        file.close()

    return e_corr, e_corr + rhf.e_tot, de


def _grad_intermediates(mol, path, vslices, oslices, auxslices,
                        nvir, nocc, naux, e_mo, log=None):
    if log is None:
        log = logger.new_logger(mol, mol.verbose)
    time0 = logger.process_clock(), logger.perf_counter()
    oblk = oslices[0].stop - oslices[0].start
    file = lib.FileMp(path + '/eris.dat', 'r+')
    gamma_3c_f = file.create_dataset('gamma_3c', (naux, nocc, nvir),
                                     'f8', blksizes=(naux, oblk, nvir))
    cderi = file['cderi']
    j2c_h = lib.empty((naux, naux), 'f8', type=MemoryTypeHost)
    j2c_h[:] = file['j2c']
    file.close()
    e_mos = Mg.broadcast(e_mo)

    ias_d = Mg.mapgpu(lambda: cupy.empty((oblk, nvir, naux), 'f8'))
    jbs_d = Mg.mapgpu(lambda: cupy.empty((oblk, nvir, naux), 'f8'))
    gamma_3c_d = Mg.mapgpu(lambda: cupy.empty((naux, oblk, nvir), 'f8'))
    tau_d = Mg.mapgpu(
        lambda: cupy.empty(
            (max(
                oblk ** 2
                * nvir ** 2,
                naux ** 2)),
            'f8'))
    rdm1_vir_d = Mg.mapgpu(lambda: cupy.zeros((nvir, nvir), 'f8'))
    t2s = Mg.mapgpu(
        lambda: cupy.empty(
            (max(
                oblk ** 2
                * nvir ** 2,
                naux ** 2)),
            'f8'))

    ngpu = Mg.ngpu
    ias_h = [lib.empty((oblk, nvir, naux), 'f8', type=MemoryTypeHost)
             for _ in range(ngpu)]
    jbs_h = [lib.empty((oblk, nvir, naux), 'f8', type=MemoryTypeHost)
             for _ in range(ngpu)]
    gamma_3c_h = [lib.empty((naux, oblk, nvir), 'f8', type=MemoryTypeHost)
                  for _ in range(ngpu)]
    gamma_2c_h = [lib.empty((naux, naux), 'f8', type=MemoryTypeHost)
                  for _ in range(ngpu)]
    for g in gamma_2c_h:
        g[:] = 0
    pools = [Pool(processes=int(0.8 * lib.NumFileProcess))
             for _ in range(ngpu)]
    pools_w = [Pool(processes=int(0.2 * lib.NumFileProcess))
               for _ in range(ngpu)]
    waits = [None for _ in range(ngpu)]
    e_corr_list = [0 for _ in range(ngpu)]
    time0 = log.timer("_grad_intermediates prepare", *time0)

    def kernel(o_ind):
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

        if o_ind > 0:
            so2 = oslices[0]
            jb_h = cderi.getitem(numpy.s_[so2], pool=pools[gid],
                                 buf=jbs_h[gid])
        elif o_ind + 1 < len(oslices):
            so2 = oslices[1]
            jb_h = cderi.getitem(numpy.s_[so2], pool=pools[gid],
                                 buf=jbs_h[gid])

        t2 = gemm(ia_d.reshape((-1, naux)), ia_d.reshape((-1, naux)),
                  transb='T', buf=t2s[gid]).reshape(
            (so_len, nvir, so_len, nvir))
        tau = lib.empty_from_buf(tau_d[gid], t2.shape, 'f8')
        cupy.copyto(tau, t2)
        tau *= 2
        tau -= t2.transpose(0, 3, 2, 1)
        div_t2(tau, e_mo_o[so], e_mo_v, e_mo_o[so], e_mo_v)

        e_corr = contraction('iajb', t2, 'iajb', tau, '')
        div_t2(t2, e_mo_o[so], e_mo_v, e_mo_o[so], e_mo_v)
        contraction('iajc', t2, 'ibjc', tau, 'ab', rdm1_vir_d[gid],
                    alpha=2.0, beta=1.0)

        gamma_3c = contraction('iajb', tau, 'jbQ', ia_d, 'Qia', buf=gamma_3c_d[gid],
                               alpha=2.0)

        for o_ind2, so2 in enumerate(oslices):

            if o_ind2 == o_ind:
                continue

            so_len2 = so2.stop - so2.start
            # `reshape` will triger `wait()`
            jb_h.wait()
            jb_d = lib.empty_from_buf(jbs_d[gid], jb_h.shape, 'f8')
            jb_d.set(jb_h)
            cupy.cuda.Device().synchronize()

            if o_ind2 + 1 < len(oslices):
                if o_ind2 + 1 == o_ind:
                    if o_ind2 + 2 < len(oslices):
                        so3 = oslices[o_ind2 + 2]
                        jb_h = cderi.getitem(numpy.s_[so3], pool=pools[gid],
                                             buf=jbs_h[gid])
                else:
                    so3 = oslices[o_ind2 + 1]
                    jb_h = cderi.getitem(numpy.s_[so3], pool=pools[gid],
                                         buf=jbs_h[gid])

            t2 = gemm(ia_d.reshape((-1, naux)), jb_d.reshape((-1, naux)),
                      transb='T', buf=t2s[gid]).reshape(
                (so_len, nvir, so_len2, nvir))
            tau = lib.empty_from_buf(tau_d[gid], t2.shape, 'f8')
            cupy.copyto(tau, t2)
            tau *= 2
            tau -= t2.transpose(0, 3, 2, 1)
            div_t2(tau, e_mo_o[so], e_mo_v, e_mo_o[so2], e_mo_v)

            contraction('iajb', tau, 'jbQ', jb_d, 'Qia',
                        gamma_3c, beta=1.0, alpha=2.0)

            if o_ind2 > o_ind:
                contraction(
                    'iajb',
                    t2,
                    'iajb',
                    tau,
                    '',
                    e_corr,
                    beta=1.0,
                    alpha=2.0)
                div_t2(t2, e_mo_o[so], e_mo_v, e_mo_o[so2], e_mo_v)
                contraction('iajc', t2, 'ibjc', tau, 'ab', rdm1_vir_d[gid],
                            alpha=2.0, beta=1.0)
                contraction('icja', t2, 'icjb', tau, 'ab', rdm1_vir_d[gid],
                            alpha=2.0, beta=1.0)

            tau = None
        j2c = lib.empty_from_buf(tau_d[gid], j2c_h.shape, 'f8')
        j2c.set(j2c_h)
        gamma_3c = gamma_3c.reshape(naux, -1)
        lib.solve_triangular(j2c.T, gamma_3c, overwrite_b=True, lower=False)
        if waits[gid] is not None:
            for w in waits[gid]:
                w.wait()
        gamma_3c = gamma_3c.reshape(naux, -1, nvir)
        gamma_3c_tmp_h = lib.empty_from_buf(gamma_3c_h[gid],
                                            gamma_3c.shape, 'f8')
        gamma_3c.get(out=gamma_3c_tmp_h, blocking=True)
        waits[gid] = gamma_3c_f.setitem(numpy.s_[:, so],
                                        gamma_3c_tmp_h, pool=pools_w[gid])
        gamma_2c = lib.empty_from_buf(t2s[gid], j2c_h.shape, 'f8')
        contraction('iaR', ia_d, 'Sia', gamma_3c, 'RS', gamma_2c)
        lib.solve_triangular(j2c.T, gamma_2c, overwrite_b=True, lower=False)
        gamma_2c_pass = j2c
        gamma_2c_pass.set(gamma_2c_h[gid])
        gamma_2c += gamma_2c_pass
        gamma_2c.get(out=gamma_2c_h[gid], blocking=True)

        log.timer('_grad_intermediates kernel nao:[%d:%d]/%d on GPU%s' %
                  (so.start, so.stop, nocc, Mg.gpus[gid]), *time0)

        e_corr_list[gid] += e_corr.item()

    Mg.map(kernel, range(len(oslices)))

    for wait_list in waits:
        for w in wait_list:
            w.wait()
    for i in range(ngpu):
        pools[i].close()
        pools[i].join()
        pools_w[i].close()
        pools_w[i].join()

    ias_d = jbs_d = t2s = ias_h = jbs_h = pools = tau_d = e_mos = None
    gamma_3c_d = gamma_3c_h = j2c_h = None

    e_corr = numpy.sum(e_corr_list).item()
    tmp = gamma_2c_h[0]
    for i in range(1, ngpu):
        tmp += gamma_2c_h[i]
    gamma_2c_h = tmp
    tmp = None

    rdm1_vir_list = Mg.map(cupy.asnumpy, rdm1_vir_d)
    rdm1_vir = numpy.sum(rdm1_vir_list, axis=0)
    rdm1_vir_list = None

    lib.Mg.mapgpu(lambda: lib.free_all_blocks())

    rdm1_occ = mp2_get_occ_1rdm(path, vslices, nvir,
                                nocc, naux, e_mo, log)
    time0 = log.timer("_grad_intermediates done!", *time0)
    return e_corr, rdm1_occ, rdm1_vir, gamma_2c_h


def get_I_mat(mol, auxmol, path, auxslices,
              coeff_o, coeff_v, log=None):
    if log is None:
        log = logger.new_logger(mol, mol.verbose)
    time0 = logger.process_clock(), logger.perf_counter()
    nocc = coeff_o.shape[1]
    nvir = coeff_v.shape[1]
    nao = mol.nao
    gamma_auxblk = auxslices[0].stop - auxslices[0].start

    vhfopt = VHFOpt3c(mol, auxmol, 'int2e')
    vhfopt.build(group_size=get_de_int3c_nao_gs,
                 aux_group_size=gamma_auxblk)
    nauxid = len(vhfopt.aux_log_qs)
    org_coeff_o = Mg.mapgpu(lambda: cupy.asarray(coeff_o))
    org_coeff_v = Mg.mapgpu(lambda: cupy.asarray(coeff_v))
    coeff_o = vhfopt.coeff.dot(cupy.asarray(coeff_o))
    coeff_v = vhfopt.coeff.dot(cupy.asarray(coeff_v))
    coeff_o = Mg.mapgpu(lambda: cupy.asarray(coeff_o))
    coeff_v = Mg.mapgpu(lambda: cupy.asarray(coeff_v))
    aocoeff = Mg.mapgpu(lambda: cupy.asarray(vhfopt.coeff))
    auxcoeff = cupy.asarray(vhfopt.auxcoeff.T, order='C')
    auxcoeff = Mg.mapgpu(lambda: cupy.asarray(auxcoeff))
    Mg.mapgpu(lambda: lib.free_all_blocks())

    kslices = []
    kextents = []
    for cp_aux_id in range(nauxid):
        k0, k1 = vhfopt.auxmol.ao_loc[vhfopt.aux_l_ctr_offsets
                                      [cp_aux_id: cp_aux_id + 2]]
        kslices.append(slice(k0, k1))
        kextents.append(k1 - k0)

    nauxblk = max(kextents)

    _, naux = vhfopt.auxcoeff.shape

    file = lib.FileMp(path + '/eris.dat', 'r')
    gamma_3c_f = file['gamma_3c']
    file.close()
    ngpu = Mg.ngpu
    time0 = log.timer("get_I_mat prepare", *time0)

    int3cs = Mg.mapgpu(lambda: cupy.empty((nauxblk,
                                           get_de_int3c_nao_gs, get_de_int3c_nao_gs), 'f8'))
    bufs = Mg.mapgpu(lambda: cupy.empty(
        (gamma_auxblk, max(nocc, nvir), nao), 'f8'))
    bufs_Lop = Mg.mapgpu(lambda: cupy.empty((gamma_auxblk, nocc, nao), 'f8'))
    bufs_Lvp = Mg.mapgpu(lambda: cupy.empty((gamma_auxblk, nvir, nao), 'f8'))
    I_mat_d = Mg.mapgpu(lambda: cupy.zeros((nao, nao), 'f8'))
    I_mat_tmp_d = Mg.mapgpu(lambda: cupy.empty((max(nocc, nvir), nao), 'f8'))
    I_mat_h = [lib.empty((nao, nao), 'f8', type=MemoryTypeHost)
               for _ in range(ngpu)]
    gamma_3c_h = [lib.empty((gamma_auxblk, nocc, nvir),
                            'f8', type=MemoryTypeHost) for _ in range(ngpu)]
    pools = [Pool(processes=lib.NumFileProcess) for _ in range(ngpu)]

    def kernel(G_Q_index):
        gid = Mg.getgid()
        time1 = logger.process_clock(), logger.perf_counter()
        sQ = auxslices[G_Q_index]

        g3c_h = lib.empty_from_buf(gamma_3c_h[gid],
                                   (sQ.stop - sQ.start, nocc, nvir), 'f8')
        waits = gamma_3c_f.getitem(numpy.s_[sQ], pool=pools[gid], buf=g3c_h)

        for cp_aux_id in range(len(vhfopt.aux_log_qs)):
            sk2 = kslices[cp_aux_id]
            if numpy.isclose(abs(auxcoeff[gid][sQ, sk2]).sum(), 0):
                continue

            int3c_Lop = lib.empty_from_buf(bufs_Lop[gid],
                                           (kextents[cp_aux_id], nocc, nao), 'f8')
            int3c_Lvp = lib.empty_from_buf(bufs_Lvp[gid],
                                           (kextents[cp_aux_id], nvir, nao), 'f8')
            int3c_Lop[:] = 0
            int3c_Lvp[:] = 0
            for cp_ij_id in range(len(vhfopt.log_qs)):
                si, sj, sk, int3c = get_int3c(
                    cp_ij_id, cp_aux_id, vhfopt, buf=int3cs[gid])
                tmp = contraction('ijL', int3c, 'jp', aocoeff[gid][sj],
                                  'ipL', buf=bufs[gid])
                contraction('ipL', tmp, 'io', coeff_o[gid][si],
                            'Lop', int3c_Lop, beta=1.0)
                contraction('ipL', tmp, 'iv', coeff_v[gid][si],
                            'Lvp', int3c_Lvp, beta=1.0)

                if si != sj:
                    tmp = contraction('ijL', int3c,
                                      'ip', aocoeff[gid][si], 'jpL', buf=bufs[gid])
                    contraction('jpL', tmp, 'jo', coeff_o[gid][sj],
                                'Lop', int3c_Lop, beta=1.0)
                    contraction('jpL', tmp, 'jv', coeff_v[gid][sj],
                                'Lvp', int3c_Lvp, beta=1.0)

            tmp = contraction('Lop', int3c_Lop, 'KL', auxcoeff[gid][sQ, sk],
                              'Kop', buf=bufs[gid])
            int3c_Lop = lib.empty_from_buf(bufs_Lop[gid], tmp.shape, 'f8')
            cupy.copyto(int3c_Lop, tmp)

            tmp = contraction('Lvp', int3c_Lvp, 'KL', auxcoeff[gid][sQ, sk],
                              'Kvp', buf=bufs[gid])
            int3c_Lvp = lib.empty_from_buf(bufs_Lvp[gid], tmp.shape, 'f8')
            cupy.copyto(int3c_Lvp, tmp)

            if waits is not None:
                for w in waits:
                    w.wait()
                waits = None

            g3c_d = lib.empty_from_buf(bufs[gid], g3c_h.shape, 'f8')
            g3c_d.set(g3c_h)

            tmp = contraction('Lia', g3c_d, 'Liv', int3c_Lop,
                              'av', buf=I_mat_tmp_d[gid])
            contraction('av', tmp, 'ua', org_coeff_v[gid],
                        'vu', I_mat_d[gid], beta=1.0)
            tmp = contraction('Lia', g3c_d, 'Lav', int3c_Lvp,
                              'iv', buf=I_mat_tmp_d[gid])
            contraction('iv', tmp, 'ui', org_coeff_o[gid],
                        'vu', I_mat_d[gid], beta=1.0)

        I_mat_d[gid].get(out=I_mat_h[gid], blocking=True)
        log.timer('get_I_mat nao:[%d:%d]/%d on GPU%s' %
                  (sQ.start, sQ.stop, naux, Mg.gpus[gid]), *time1)

    Mg.map(kernel, range(len(auxslices)))
    for pool in pools:
        pool.close()
        pool.join()

    I_mat = cupy.asarray(I_mat_h).sum(axis=0)
    I_mat_d = I_mat_tmp_d = int3cs = bufs = bufs_Lop = bufs_Lvp = I_mat_h = None
    auxcoeff = coeff_o = coeff_v = org_coeff_o = org_coeff_v = None
    time0 = log.timer("get_I_mat done!", *time0)
    return I_mat


def _response_dm1(mf, mo_coeff, Xvo, log=None):
    if log is None:
        log = logger.new_logger(mf.mol, mf.mol.verbose)
    time0 = logger.process_clock(), logger.perf_counter()
    nvir, nocc = Xvo.shape
    nmo = nocc + nvir
    mo_energy = cupy.asarray(mf.mo_energy)
    mo_occ = cupy.asarray(mf.mo_occ)
    time0 = log.timer("gpu4pyscf.scf.cphf.solve perpare", *time0)

    def fvind(x):
        x = x.reshape(Xvo.shape)
        dm = reduce(cupy.dot, (mo_coeff[:, nocc:], x, mo_coeff[:, :nocc].T))
        j, k = gpuscf.jk.get_jk(mf.mol, dm + dm.T)
        v = j - 0.5 * k
        v = reduce(cupy.dot, (mo_coeff[:, nocc:].T, v, mo_coeff[:, :nocc]))
        return v * 2
    dvo = gpuscf.cphf.solve(fvind, mo_energy, mo_occ, Xvo, max_cycle=30)[0]
    dm1 = cupy.zeros((nmo, nmo))
    dm1[nocc:, :nocc] = dvo
    dm1[:nocc, nocc:] = dvo.T
    time0 = log.timer("gpu4pyscf.scf.cphf.solve done!", *time0)
    return dm1


def get_de_int3c(mol, auxmol, path, auxslices,
                 oslices, coeff_o, coeff_v, log=None):
    if log is None:
        log = logger.new_logger(mol, mol.verbose)
    time0 = logger.process_clock(), logger.perf_counter()
    nocc = coeff_o.shape[1]
    nvir = coeff_v.shape[1]
    nao = mol.nao
    naux = auxmol.nao
    gamma_auxblk = auxslices[0].stop - auxslices[0].start
    gamma_oblk = oslices[0].stop - oslices[0].start

    intopt = int3c2e.VHFOpt(mol, auxmol, 'int2e')
    intopt.build(
        group_size=get_de_int3c_nao_gs,
        group_size_aux=get_de_int3c_naux_gs)

    file = lib.FileMp(path + '/eris.dat', 'r')
    gamma_3c_f = file['gamma_3c']
    file.close()

    ao_loc = intopt.ao_loc
    aux_ao_loc = intopt.aux_ao_loc

    kextents = []
    kslices = []
    for aux_id in range(len(intopt.aux_log_qs)):
        k0, k1 = aux_ao_loc[aux_id], aux_ao_loc[aux_id + 1]
        kslices.append(slice(k0, k1))
        kextents.append(k1 - k0)

    iextents = []
    islices = []
    for cp_ij_ind in range(len(intopt.log_qs)):
        cpi = intopt.cp_idx[cp_ij_ind]
        i0, i1 = ao_loc[cpi], ao_loc[cpi + 1]
        islices.append(slice(i0, i1))
        iextents.append(i1 - i0)

    ngpu = Mg.ngpu
    aux_sort = intopt.sort_orbitals(cupy.eye(naux), aux_axis=[1])
    lib.Mg.mapgpu(lambda: lib.free_all_blocks())
    aux_sort = Mg.broadcast(aux_sort)
    gamma_3c_d = Mg.mapgpu(
        lambda: cupy.empty(
            (gamma_auxblk, nocc, nvir), 'f8'))
    inter_tmp_d = Mg.mapgpu(lambda: cupy.empty(
        (gamma_auxblk, get_de_int3c_nao_gs, nocc), 'f8'))
    G_tmp_d = Mg.mapgpu(
        lambda: cupy.empty(
            (gamma_auxblk,
             get_de_int3c_nao_gs,
             get_de_int3c_nao_gs),
            'f8'))
    K_tmp_d = Mg.mapgpu(
        lambda: cupy.empty(
            (get_de_int3c_naux_gs,
             get_de_int3c_nao_gs,
             get_de_int3c_nao_gs),
            'f8'))
    de_ip1 = Mg.mapgpu(lambda: cupy.zeros((nao, 3), 'f8'))
    de_ip1_h = [lib.empty((nao, 3), 'f8', type=MemoryTypeHost)
                for _ in range(ngpu)]
    gamma_3c_h = [
        lib.empty(
            (gamma_auxblk,
             nocc,
             nvir),
            'f8',
            type=MemoryTypeHost) for _ in range(ngpu)]

    intopt = Mg.mapgpu(lambda: intopt)
    coeff_o_sort = Mg.mapgpu(
        lambda: cupy.asarray(
            intopt[0].sort_orbitals(
                coeff_o, axis=[0])))
    coeff_v_sort = Mg.mapgpu(
        lambda: cupy.asarray(
            intopt[0].sort_orbitals(
                coeff_v, axis=[0])))
    pools = [Pool(processes=lib.NumFileProcess) for _ in range(ngpu)]
    time0 = log.timer("get_de_int3c prepare", *time0)

    def kernel_ip1(G_Q_index):
        gid = Mg.getgid()
        time0 = logger.process_clock(), logger.perf_counter()
        sQ = auxslices[G_Q_index]
        g3c_h = lib.empty_from_buf(
            gamma_3c_h[gid], (sQ.stop - sQ.start, nocc, nvir), 'f8')
        waits = gamma_3c_f.getitem(numpy.s_[sQ], pool=pools[gid], buf=g3c_h)
        for w in waits:
            w.wait()
        g3c_d = lib.empty_from_buf(gamma_3c_d[gid], g3c_h.shape, 'f8')
        g3c_d.set(g3c_h)

        for aux_id in range(len(intopt[gid].aux_log_qs)):
            sk = kslices[aux_id]
            if numpy.isclose(abs(aux_sort[gid][sQ, sk]).sum(), 0):
                continue
            task_list = zip([aux_id] * len(intopt[gid].log_qs),
                            list(range(len(intopt[gid].log_qs))))
            for p0, p1, q0, q1, k0, k1, int3c_blk in int3c2e.loop_int3c2e_general(
                    intopt[gid], task_list=task_list, ip_type='ip1'):
                assert sk.start == k0 and sk.stop == k1
                sp = slice(p0, p1)
                sq = slice(q0, q1)
                inter_tmp = contraction(
                    'Lia',
                    g3c_d,
                    'qa',
                    coeff_v_sort[gid][sq],
                    'Liq',
                    buf=inter_tmp_d[gid])
                G_tmp = contraction(
                    'Liq',
                    inter_tmp,
                    'pi',
                    coeff_o_sort[gid][sp],
                    'Lpq',
                    buf=G_tmp_d[gid])
                inter_tmp = contraction(
                    'Lia',
                    g3c_d,
                    'pa',
                    coeff_v_sort[gid][sp],
                    'Lpi',
                    buf=inter_tmp_d[gid])
                contraction(
                    'Lpi',
                    inter_tmp,
                    'qi',
                    coeff_o_sort[gid][sq],
                    'Lpq',
                    G_tmp,
                    beta=1.0)
                K_tmp = contraction(
                    'Lpq', G_tmp, 'LK', aux_sort[gid][sQ, sk], 'Kpq', buf=K_tmp_d[gid])
                contraction(
                    'Kpq',
                    K_tmp,
                    'xKqp',
                    int3c_blk,
                    'px',
                    de_ip1[gid][sp],
                    beta=1.0,
                    alpha=-2.0)

            int3c_blk = None

        de_ip1[gid].get(out=de_ip1_h[gid], blocking=True)
        log.timer('get_de_int3c kernel_ip1 naux:[%d:%d]/%d on GPU%s' %
                  (sQ.start, sQ.stop, naux, Mg.gpus[gid]), *time0)

    Mg.map(kernel_ip1, range(len(auxslices)))

    gamma_3c_h = gamma_3c_d = int3c_d = None
    lib.Mg.mapgpu(lambda: lib.free_all_blocks())
    for pool in pools:
        pool.close()
        pool.join()

    de_ip1 = cupy.asarray(de_ip1_h).sum(axis=0)
    de_ip1 = intopt[0].unsort_orbitals(de_ip1, axis=[0])
    offsetdic = mol.offset_nr_by_atom()
    de = cupy.zeros((mol.natm, 3), 'f8')
    for k, ia in enumerate(list(range(mol.natm))):
        _, _, p0, p1 = offsetdic[ia]
        sa = slice(p0, p1)
        de[k] += de_ip1[sa].sum(axis=0).ravel()
    de_ip1 = de_ip1_h = None

    gamma_3c_h = [lib.empty((naux, gamma_oblk, nvir), 'f8',
                            type=MemoryTypeHost) for _ in range(ngpu)]
    gamma_3c_d = Mg.mapgpu(lambda: cupy.empty((naux, gamma_oblk, nao), 'f8'))
    int3c_d = Mg.mapgpu(lambda: cupy.empty((3, naux, gamma_oblk, nao), 'f8'))
    de_ip2 = Mg.mapgpu(lambda: cupy.zeros((naux, 3), 'f8'))
    de_ip2_h = [lib.empty((naux, 3), 'f8', type=MemoryTypeHost)
                for _ in range(ngpu)]
    pools = [Pool(processes=lib.NumFileProcess) for _ in range(ngpu)]
    j_log_qs_fix_i = {i: [] for i in list(set(list(intopt[0].cp_idx.ravel())))}
    for cp_ij_ind in range(len(intopt[0].log_qs)):
        cpi = intopt[0].cp_idx[cp_ij_ind]
        j_log_qs_fix_i[cpi].append(cp_ij_ind)

    def kernel_ip2(G_o_index):
        gid = Mg.getgid()
        time0 = logger.process_clock(), logger.perf_counter()

        so = oslices[G_o_index]

        g3c_h = lib.empty_from_buf(gamma_3c_h[gid],
                                   (naux, so.stop - so.start, nvir), 'f8')
        waits = gamma_3c_f.getitem(numpy.s_[:, so],
                                   pool=pools[gid], buf=g3c_h)

        for w in waits:
            w.wait()

        g3c_d = lib.empty_from_buf(gamma_3c_d[gid], g3c_h.shape, 'f8')
        g3c_d.set(g3c_h)
        inter_tmp = contraction('Qia', g3c_d, 'QK', aux_sort[gid],
                                'Kia', buf=int3c_d[gid])
        G = contraction('Kia', inter_tmp, 'qa', coeff_v_sort[gid],
                        'Kiq', buf=gamma_3c_d[gid])

        waits = None

        for i_ind in j_log_qs_fix_i.keys():
            si = islices[j_log_qs_fix_i[i_ind][0]]
            if numpy.isclose(abs(coeff_o_sort[gid][si, so]).sum(), 0):
                continue
            task_list = product(list(range(len(intopt[gid].aux_log_qs))),
                                j_log_qs_fix_i[i_ind])
            int3c = lib.empty_from_buf(int3c_d[gid],
                                       (3, naux, so.stop - so.start, nao), 'f8')
            int3c[:] = 0
            for i0, i1, j0, j1, k0, k1, int3c_blk in \
                    int3c2e.loop_int3c2e_general(intopt[gid], task_list=task_list, ip_type='ip2'):

                assert si.start == i0 and si.stop == i1
                sk = slice(k0, k1)
                sj = slice(j0, j1)
                contraction('xKqp', int3c_blk,
                            'pi', coeff_o_sort[gid][si, so],
                            'xKiq', int3c[:, sk, :, sj], beta=1.0)

            int3c_blk = None
            contraction('Kiq', G, 'xKiq', int3c, 'Kx', de_ip2[gid],
                        beta=1.0, alpha=-2.0)

        de_ip2[gid].get(out=de_ip2_h[gid], blocking=True)
        log.timer('get_de_int3c kernel_ip2  nocc:[%d:%d]/%d on GPU%s' %
                  (so.start, so.stop, nocc, Mg.gpus[gid]), *time0)

    Mg.map(kernel_ip2, range(len(oslices)))

    for pool in pools:
        pool.close()
        pool.join()
    gamma_3c_h = gamma_3c_d = int3c_d = None
    lib.Mg.mapgpu(lambda: lib.free_all_blocks())

    de_ip2 = cupy.asarray(de_ip2_h).sum(axis=0)
    de_ip2 = intopt[0].unsort_orbitals(de_ip2, aux_axis=[0])
    aux_offsetdic = auxmol.offset_nr_by_atom()
    for k, ia in enumerate(list(range(mol.natm))):
        _, _, ap0, ap1 = aux_offsetdic[ia]
        asa = slice(ap0, ap1)
        de[k] += de_ip2[asa].sum(axis=0).ravel()
    intopt = de_ip2 = de_ip2_h = None
    time0 = log.timer("get_de_int3c done!", *time0)

    return de
