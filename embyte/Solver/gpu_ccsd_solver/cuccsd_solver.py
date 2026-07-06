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

import h5py
from pyscf.lib import prange
import gc
from byteqc.embyte.ERI import eri_trans
from functools import reduce
import cupyx
from byteqc import lib
from byteqc.embyte.Tools.tool_lib import fix_orbital_sign
from multiprocessing import Pool
from byteqc.cucc import ccsd_lambda, ccsd_t
from byteqc.cucc import culib
from byteqc.cucc.ccsd import EriWrapper
from byteqc.cucc.dfccsd import _ChemistsERIs, nr_e2
from byteqc import cucc
import pyscf
from pyscf import scf, gto
import numpy
import os
import cupy
cupy.cuda.set_pinned_memory_allocator(None)


def _make_df_eris(mycc, projector, mo_coeff=None):
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nfrag = eris.nfrag = projector.shape[1]
    nvir = nmo - nocc
    with_df = mycc.with_df
    naux = eris.naux = with_df.get_naoaux()
    status = mycc.pool.memory_status(nocc, nvir, naux,
                                     mem_ratio=mycc.mem_ratio)

    Loo = eris.Loo = mycc.pool.new('Loo', (naux, nocc, nocc), 'f8')
    Lov = eris.Lov = mycc.pool.new('Lov', (naux, nocc, nvir), 'f8')
    Lvv = eris.Lvv = mycc.pool.new('Lvv', (naux, nvir, nvir), 'f8')
    Lfv = eris.Lfv = mycc.pool.new('Lfv', (naux, nfrag, nvir), 'f8')
    d_mo_coeff = mycc.pool.asarray(eris.mo_coeff)
    ijslice = (0, nmo, 0, nmo)
    p1 = 0
    Lpq = None
    for _, eri1 in enumerate(with_df.loop()):
        eri1 = cupy.asarray(eri1)
        Lpq = nr_e2(eri1, d_mo_coeff.T, ijslice,
                    aosym='s2', mosym='s1', out=Lpq)
        p0, p1 = p1, p1 + Lpq.shape[0]
        Lpq = Lpq.reshape(p1 - p0, nmo, nmo)
        Loo[p0:p1] = Lpq[:, :nocc, :nocc]
        Lov[p0:p1] = Lpq[:, :nocc, nocc:]
        Lvv[p0:p1] = Lpq[:, nocc:, nocc:]
        culib.contraction('Lib', Lov[p0:p1], 'if',
                          projector, 'Lfb', Lfv[p0:p1])
        eri1 = None
    Lpq = d_mo_coeff = None
    lib.free_all_blocks()

    _Loo = Loo.reshape(naux, nocc**2)
    _Lov = Lov.reshape(naux, nocc * nvir)
    _Lvv = Lvv.reshape(naux, nvir**2)

    if status['oooo'] != 0:
        eris.oooo = EriWrapper(Loo, Loo)
    else:
        eris.oooo = EriWrapper(mycc.pool.new(
            'oooo', (nocc, nocc, nocc, nocc), 'f8'))
        culib.gemm('T', 'N', _Loo, _Loo, eris.oooo.reshape(nocc**2, -1))

    if status['ovoo'] != 0:
        eris.ovoo = EriWrapper(Lov, Loo)
    else:
        eris.ovoo = EriWrapper(mycc.pool.new(
            'ovoo', (nocc, nvir, nocc, nocc), 'f8'))
        culib.gemm('T', 'N', _Lov, _Loo, eris.ovoo.reshape(nocc * nvir, -1))

    if status['ovov'] != 0:
        eris.ovov = EriWrapper(Lov, Lov)
    else:
        eris.ovov = EriWrapper(mycc.pool.new(
            'ovov', (nocc, nvir, nocc, nvir), 'f8'))
        culib.gemm('T', 'N', _Lov, _Lov, eris.ovov.reshape(nocc * nvir, -1))

    if status['oovv'] != 0:
        eris.oovv = EriWrapper(Loo, Lvv)
    else:
        eris.oovv = EriWrapper(mycc.pool.new(
            'oovv', (nocc, nocc, nvir, nvir), 'f8'))
        culib.gemm('T', 'N', _Loo, _Lvv, eris.oovv.reshape(nocc**2, -1))

    if status['ovvv'] != 0:
        eris.ovvv = EriWrapper(Lov, Lvv)
    else:
        eris.ovvv = EriWrapper(mycc.pool.new(
            'ovvv', (nocc, nvir, nvir, nvir), 'f8'))
        culib.gemm('T', 'N', _Lov, _Lvv, eris.ovvv.reshape(nocc * nvir, -1))
    _Loo = _Lov = None
    lib.free_all_blocks()
    return eris


class GPU_CCSDSolver():
    '''
    GPU accelerated CCSD/CCSD(T) solver.
    '''

    def __init__(self):
        self.cc_fragment = None
        self.__name__ = 'GPU_CCSDSolver'

    def make_param(
        self,
        nelec_high,
        cluster_list,
        LOBNO,
        low_level_info,
        Logger,
        vhfopt3c,
        rdm1_core_coeff,
        nfrag,
        eri_file=None
    ):

        self.Logger = Logger
        self.eri_file = eri_file
        self.vhfopt3c = vhfopt3c
        self.rdm1_core_coeff = rdm1_core_coeff
        self.cluster_list = cluster_list
        self.nelec_high = nelec_high

        Logger.info('==== CCSD (cucc) solver has been used here ====')

        nocc = round(nelec_high // 2)

        self.LOEO, self.orb_energy, self.EOMO, self.LOMO, self.AOMO = self.get_coeff(
            LOBNO, cluster_list, low_level_info)
        self.projector = self.EOMO[:, :nocc].T[:, :nfrag].copy()
        self.nocc = nocc
        self.nvir = (self.AOMO.shape[1] - nocc)
        self.nmo = self.AOMO.shape[1]
        self.AOMO = self.AOMO

    def get_eri(self, low_level_info, eri_path, save_or_load):
        '''
        Load/Generate a CDERI for cluster.
        '''

        AOLO = cupy.asarray(low_level_info.AOLO)
        rdm1_core_coeff = reduce(
            cupy.dot, (AOLO, cupy.asarray(
                self.rdm1_core_coeff)))
        rdm1_core = reduce(cupy.dot, (rdm1_core_coeff, rdm1_core_coeff.T))

        if save_or_load:
            if self.eri_file is None:

                if rdm1_core.any():

                    self.eri_general, j, k, = eri_trans.eri_high_level_solver_incore_with_jk(
                        low_level_info.mol_full,
                        low_level_info.auxmol,
                        self.AOMO,
                        low_level_info.j2c,
                        self.Logger,
                        rdm1_core_coeff,
                        vhfopt=self.vhfopt3c,
                        svd_tol=1e-4)

                    veff = cupy.asarray(j - 0.5 * k)
                    self.cluster_oei = reduce(
                        cupy.dot, (self.LOMO.T, cupy.asarray(
                            low_level_info.oei_LO), self.LOMO))
                    self.cluster_oei += veff
                    self.cluster_oei = self.cluster_oei.get(blocking=True)

                else:
                    self.eri_general = eri_trans.eri_high_level_solver_incore(
                        low_level_info.mol_full,
                        low_level_info.auxmol,
                        self.AOMO,
                        self.AOMO,
                        low_level_info.j2c,
                        self.Logger,
                        solver_type='CCSD',
                        vhfopt=self.vhfopt3c,
                        svd_tol=1e-4)

                    self.cluster_oei = cupy.asarray(low_level_info.oei_LO)
                    self.cluster_oei = reduce(
                        cupy.dot, (self.LOMO.T, self.cluster_oei, self.LOMO)).get(
                        blocking=True)

            else:
                if rdm1_core.any():

                    self.eri_general, j, k = eri_trans.eri_ondisk_high_level_solver_incore_with_jk(
                        low_level_info.mol_full,
                        self.eri_file,
                        self.AOMO,
                        self.Logger,
                        rdm1_core_coeff,
                        svd_tol=1e-4)

                    veff = cupy.asarray(j - 0.5 * k)
                    self.cluster_oei = reduce(
                        cupy.dot, (self.LOMO.T, cupy.asarray(
                            low_level_info.oei_LO), self.LOMO))
                    self.cluster_oei += veff
                    self.cluster_oei = self.cluster_oei.get(blocking=True)

                else:

                    self.eri_general = eri_trans.eri_ondisk_high_level_solver_incore(
                        low_level_info.mol_full,
                        self.eri_file,
                        self.AOMO,
                        self.AOMO,
                        self.Logger,
                        solver_type='CCSD',
                        svd_tol=1e-4)

                    self.cluster_oei = cupy.asarray(low_level_info.oei_LO)
                    self.cluster_oei = reduce(
                        cupy.dot, (self.LOMO.T, self.cluster_oei, self.LOMO)).get(
                        blocking=True)

            # pool_rw = Pool(processes=min(lib.NumFileProcess, self.eri_general.shape[0]))
            file = lib.FileMp(
                os.path.join(
                    eri_path,
                    'cluster_cderis.dat'),
                'w')
            blk = max(int(self.eri_general.shape[0] / lib.NumFileProcess), 1)
            cderi = file.create_dataset(
                'cderi', self.eri_general.shape, 'f8', blksizes=(blk))
            # wait_list = cderi.setitem(
            #     numpy.s_[:], self.eri_general, pool=pool_rw)
            wait_list = cderi.setitem(
                numpy.s_[:], self.eri_general)
            for w in wait_list:
                w.wait()
            # pool_rw.close()
            # pool_rw.join()
            file.close()
            numpy.save(
                os.path.join(
                    eri_path,
                    'cluster_oei.npy'),
                self.cluster_oei)

        else:
            self.Logger.info('Load cderi from disk path : %s' %
                             os.path.join(eri_path, 'cluster_cderis.dat'))
            file = lib.FileMp(
                os.path.join(
                    eri_path,
                    'cluster_cderis.dat'),
                'r')
            cderi = file['cderi']
            self.eri_general = cupyx.zeros_pinned(cderi.shape, dtype='f8')
            pool_rw = Pool(processes=lib.NumFileProcess)
            wait_list = cderi.getitem(
                numpy.s_[:], pool=pool_rw, buf=self.eri_general)
            for w in wait_list:
                w.wait()
            pool_rw.close()
            pool_rw.join()
            file.close()
            self.cluster_oei = numpy.load(
                os.path.join(eri_path, 'cluster_oei.npy'))

        self.eri_general = self.eri_general.reshape(
            self.eri_general.shape[0], self.nmo, self.nmo)
        veff = None

    def get_cluster_coeff(self):

        return self.EOMO, self.LOMO

    def kernel(self):

        mol_frag = gto.Mole()
        mol_frag.build(verbose=0)
        mol_frag.atom.append(('C', (0, 0, 0)))
        mol_frag.nelectron = self.nelec_high
        mol_frag.incore_anyway = True

        mf_frag = scf.RHF(mol_frag).density_fit()
        mf_frag.get_hcore = lambda *args: self.cluster_oei
        mf_frag.get_ovlp = lambda *args: numpy.eye(self.nmo)
        mf_frag.mo_coeff = numpy.eye(self.nmo)
        mf_frag.mo_occ = numpy.asarray([2 for i in range(
            self.nocc)] + [0 for i in range(len(self.cluster_list) - self.nocc)])

        self.eri_general = pyscf.lib.pack_tril(self.eri_general)
        mf_frag.with_df._cderi = self.eri_general

        self.cc_fragment = cucc.CCSD(mf_frag)
        self.eris = _make_df_eris(
            self.cc_fragment,
            self.projector,
            mf_frag.mo_coeff)
        self.eris.e_hf = 0

        self.eri_general = self.cc_fragment.with_df._cderi = None

        self.cc_fragment.verbose = 4
        self.cc_fragment.conv_tol = 1e-8
        self.cc_fragment.conv_tol_normt = 1e-6
        self.cc_fragment.stdout = self.Logger
        self.cc_fragment.diis_space = 6
        self.cc_fragment.level_shift = 0.2
        self.cc_fragment.max_cycle = 200
        self.cc_fragment.kernel(eris=self.eris)
        self.pool = self.cc_fragment.pool

        assert self.cc_fragment.converged

    def get_coeff(self, LOBNO, cluster_list, low_level_info):
        LOEO = LOBNO[:, cluster_list]
        Fock_clu = cupy.einsum(
            'ip, jq, ij -> pq',
            cupy.asarray(LOEO),
            cupy.asarray(LOEO),
            cupy.asarray(low_level_info.fock_LO))
        orb_energy, EOMO = cupy.linalg.eigh(Fock_clu)
        EOMO = fix_orbital_sign(EOMO)
        LOMO = cupy.dot(LOEO, EOMO)
        AOMO = cupy.dot(low_level_info.AOLO, LOMO)

        return LOEO, orb_energy, EOMO, LOMO, AOMO

    def get_truncation_T(self, if_RDM=True):
        if if_RDM:
            conv, self.l1, self.l2 = ccsd_lambda.kernel(
                self.cc_fragment, eris=self.eris, tol=self.cc_fragment.conv_tol_normt)
            assert conv

        self.eris.Loo = self.eris.Loo.asnumpy()
        self.eris.Lov = self.eris.Lov.asnumpy()
        self.eris.Lvv = self.eris.Lvv.asnumpy()

        self.t1 = self.cc_fragment.t1.ascupy()
        self.t2 = self.cc_fragment.t2.ascupy()
        self.cc_fragment.t1 = self.cc_fragment.t2 = None

        projector = cupy.asarray(self.projector)
        nL, no, nv = self.eris.Lov.shape
        nf = self.eris.nfrag
        self.t1c = self.cc_fragment.pool.new('t1c', (nf, nv), 'f8')
        culib.contraction('ia', self.t1, 'if', projector, 'fa', self.t1c)
        self.t2c = self.cc_fragment.pool.new('t2c', (nf, no, nv, nv), 'f8')
        culib.contraction('ijab', self.t2, 'if', projector, 'fjab', self.t2c)
        self.t1c = self.t1c.asnumpy()
        self.t2c = self.t2c.asnumpy()
        self.t2 = self.t2.asnumpy()
        if if_RDM:
            self.t1x = self.cc_fragment.pool.new('t1x', (no, nv), 'f8')
            culib.contraction('fa', self.t1c, 'if', projector, 'ia', self.t1x)

            self.l1 = self.l1.ascupy()
            self.l2 = self.l2.ascupy()
            l1c = self.cc_fragment.pool.empty((nf, nv), 'f8')
            culib.contraction('ia', self.l1, 'if', projector, 'fa', l1c)
            self.l1x = self.cc_fragment.pool.new('l1x', (no, nv), 'f8')
            culib.contraction('fa', l1c, 'if', projector, 'ia', self.l1x)
            l1c = None
            self.l2c = self.cc_fragment.pool.new('l2c', (nf, no, nv, nv), 'f8')
            culib.contraction(
                'ijab',
                self.l2,
                'if',
                projector,
                'fjab',
                self.l2c)
            self.l2c = self.l2c.asnumpy()
            self.l1 = self.l1.asnumpy()
            self.l2 = self.l2.asnumpy()

        self.mo_energy = self.eris.mo_energy
        self.fock = self.eris.fock

        lib.free_all_blocks()
        gc.collect()

    def get_T_correction(self, eri_path, t1t2_path):
        '''
        In-situ perturbative (T) correaction for CCSD.
        '''
        if not hasattr(self, 'eris'):
            self.Logger.info(
                'Load cderi from disk path : %s' %
                os.path.join(eri_path, 'cluster_cderis.dat'))

            file = lib.FileMp(
                os.path.join(
                    eri_path,
                    'cluster_cderis.dat'),
                'r')
            cderi = file['cderi']
            self.eri_general = cupyx.zeros_pinned(cderi.shape, dtype='f8')
            pool_rw = Pool(processes=lib.NumFileProcess)
            wait_list = cderi.getitem(
                numpy.s_[:], pool=pool_rw, buf=self.eri_general)
            for w in wait_list:
                w.wait()
            pool_rw.close()
            pool_rw.join()
            file.close()
            self.cluster_oei = numpy.load(
                os.path.join(eri_path, 'cluster_oei.npy'))
            self.eri_general = self.eri_general.reshape(
                (-1, self.nmo, self.nmo))

            mol_frag = gto.Mole()
            mol_frag.build(verbose=0)
            mol_frag.atom.append(('C', (0, 0, 0)))
            mol_frag.nelectron = self.nelec_high
            mol_frag.incore_anyway = True

            mf_frag = scf.RHF(mol_frag).density_fit()
            mf_frag.get_hcore = lambda *args: self.cluster_oei
            mf_frag.get_ovlp = lambda *args: numpy.eye(self.nmo)
            mf_frag.mo_coeff = numpy.eye(self.nmo)
            mf_frag.mo_occ = numpy.asarray([2 for i in range(
                self.nocc)] + [0 for i in range(len(self.cluster_list) - self.nocc)])

            self.eri_general = pyscf.lib.pack_tril(self.eri_general)
            mf_frag.with_df._cderi = self.eri_general

            self.cc_fragment = cucc.CCSD(mf_frag)
            self.cc_fragment.stdout = self.Logger
            self.eris = _make_df_eris(
                self.cc_fragment, self.projector, mf_frag.mo_coeff)
            self.eri_general = self.cluster_oei = None
            self.pool = self.cc_fragment.pool
            lib.free_all_blocks()
            gc.collect()

        if not hasattr(self, 't2'):
            t2_h = cupyx.empty_pinned(
                (self.nocc, self.nocc, self.nvir, self.nvir), dtype='f8')
            file = lib.FileMp(os.path.join(t1t2_path, '_t2'), 'r')
            t2_f = file['t2']
            pool_rw = Pool(processes=lib.NumFileProcess)
            wait_list = t2_f.getitem(numpy.s_[:], pool=pool_rw, buf=t2_h)
            for w in wait_list:
                w.wait()
            pool_rw.close()
            pool_rw.join()
            file.close()
            self.t2 = self.pool.new(
                't2', (self.nocc, self.nocc, self.nvir, self.nvir), dtype='f8')
            self.t2.set(t2_h)
            cupy.cuda.get_current_stream().synchronize()
            t2_h = None

        if not hasattr(self, 't1'):
            self.t1 = self.pool.new('t1', (self.nocc, self.nvir), dtype='f8')
            t1_h = cupyx.empty_pinned((self.nocc, self.nvir), dtype='f8')
            with h5py.File(os.path.join(t1t2_path, 't1'), 'r') as f:
                t1_h[:] = f['t1'][:]
            self.t1.set(t1_h)
            cupy.cuda.get_current_stream().synchronize()
            t1_h = None

        self.cc_fragment.verbose = 7
        projector = cupy.asarray(self.projector)
        projector = cupy.dot(projector, projector.T)
        self.et = ccsd_t.kernel(
            self.cc_fragment,
            self.eris,
            self.t1,
            self.t2,
            projector)

        lib.free_all_blocks()
        gc.collect()
        return self.et

    def get_frag_correlation_energy(self, if_RDM=True):
        '''
        Get the partition wavefunction correlation energy and
        in-cluster 2-RDM correlation energy contribution if if_RDM is True.
        '''

        no, nv = self.t1.shape
        self.l2 = None

        lib.free_all_blocks()
        gc.collect()

        e_wf = self.get_frag_amplitude_energy()
        self.eris.Lfv = self.t2c = self.t1c = None

        if not if_RDM:
            return e_wf, None

        self.l2x = self.pool.new('l2x', (no, no, nv, nv), 'f8', pin=1)
        self.l2c = self.l2c.ascupy()
        with self.l2x as arr:
            culib.contraction(
                'fjab',
                self.l2c,
                'if',
                self.projector,
                'ijab',
                arr,
                alpha=0.5)
            culib.contraction(
                'fjab',
                self.l2c,
                'if',
                self.projector,
                'jiba',
                arr,
                beta=1.0,
                alpha=0.5)

        self.l2c = None

        lib.free_all_blocks()

        tau = self.pool.empty(self.t2.shape, 'f8')
        tau.set(self.t2)
        culib.contraction(
            'ia',
            self.t1,
            'jb',
            self.t1,
            'ijab',
            tau,
            beta=1.0,
            alpha=1.0)
        Lov = self.eris.Lov.ascupy()
        e_tmp = self.pool.empty(Lov.shape, 'f8')
        culib.contraction('Lia', Lov, 'ijab', tau, 'Ljb', e_tmp)
        culib.contraction(
            'Lib',
            Lov,
            'ijab',
            tau,
            'Lja',
            e_tmp,
            beta=1.0,
            alpha=-0.5)

        e_ovov_delta = -1 * cupy.dot(Lov.ravel(), e_tmp.ravel().T)
        e_ovov_delta += e_wf * 0.5
        e_ovov_delta *= 4

        Lov = e_tmp = None
        lib.free_all_blocks()
        gc.collect()

        self.goooo = self.pool.new('goooo', (no, no, no, no), 'f8', pin=1)
        with self.goooo as arr:
            culib.contraction(
                'ijab',
                tau,
                'klab',
                self.l2x,
                'ijkl',
                arr,
                alpha=0.5)
        tau = None
        lib.free_all_blocks()
        gc.collect()

        self.make_aux_tensor()
        self.Logger.info('----- make_aux_tensor Done!')

        lib.free_all_blocks()
        gc.collect()
        e_oooo = self.get_cumulant_energy_oooo()
        self.Logger.info('----- get_cumulant_energy_oooo Done!')

        lib.free_all_blocks()
        gc.collect()
        e_ooov = self.get_cumulant_energy_ooov()
        self.Logger.info('----- get_cumulant_energy_ooov Done!')

        lib.free_all_blocks()
        gc.collect()
        e_ovov = self.get_cumulant_energy_ovov()
        self.Logger.info('----- get_cumulant_energy_ovov Done!')

        lib.free_all_blocks()
        gc.collect()
        e_ovvo, e_oovv = self.get_cumulant_energy_ovvo_oovv()
        self.Logger.info('----- get_cumulant_energy_ovvo_oovv Done!')

        lib.free_all_blocks()
        gc.collect()
        e_vvvv_ovvv = self.get_cumulant_energy_vvvv_ovvv()
        self.Logger.info('----- get_cumulant_energy_vvvv_ovvv Done!')

        e_2rdm = e_oooo + e_ooov + e_ovov + e_ovvo + \
            e_oovv + e_vvvv_ovvv + e_ovov_delta.tolist()
        e_2rdm /= 2
        return e_wf, e_2rdm

    def get_frag_amplitude_energy(self):

        no, nv = self.t1.shape
        nf = self.eris.nfrag
        nL = self.eris.naux

        e_corr = 0
        self.t2c = self.t2c.ascupy()
        self.t1c = self.t1c.ascupy()
        self.t1 = self.t1.ascupy()
        e_tmp = self.pool.empty((nL, nf, nv), 'f8')
        e_tmp[:] = 0
        culib.contraction(
            'fa',
            self.t1c,
            'jb',
            self.t1,
            'fjab',
            self.t2c,
            beta=1.0)
        Lov = self.eris.Lov.ascupy()
        culib.contraction(
            'fjab',
            self.t2c,
            'Ljb',
            Lov,
            'Lfa',
            e_tmp,
            beta=1.0,
            alpha=2.0)
        culib.contraction(
            'fjab',
            self.t2c,
            'Lja',
            Lov,
            'Lfb',
            e_tmp,
            beta=1.0,
            alpha=-1.0)

        Lfv = self.eris.Lfv.ascupy()

        e_corr += cupy.dot(e_tmp.ravel(), Lfv.ravel().T)
        Lov = self.t2c = Lfv = None
        lib.free_all_blocks()

        return e_corr.item()

    def make_aux_tensor(self):

        no, nv = self.t1.shape

        self.pvOOv = self.pool.new('pvOOv', (nv, no, no, nv), 'f8', pin=1)
        self.pvOOv[:] = 0
        with self.pvOOv as arr_pvOOv:
            free_size = self.pool.free_memory / 8
            slice_len = min(no, int(free_size / (2 * no * nv * nv)))
            # slice_len = 2
            slice_o = [slice(i[0], i[1]) for i in prange(0, no, slice_len)]
            for so in slice_o:
                with self.l2x[so] as arr_l2:
                    with self.t2[so] as arr_t2:
                        culib.contraction(
                            'kiac', arr_l2, 'kjbc', arr_t2, 'aijb', arr_pvOOv, beta=1.0)

            self.moo = self.pool.new('moo', (no, no), 'f8', pin=0)
            self.mvv = self.pool.new('mvv', (nv, nv), 'f8', pin=0)
            culib.contraction(
                'aljb',
                arr_pvOOv,
                'ab',
                cupy.eye(nv),
                'jl',
                self.moo,
                alpha=2.0)
            culib.contraction(
                'bijd',
                arr_pvOOv,
                'ij',
                cupy.eye(no),
                'db',
                self.mvv,
                alpha=2.0)

        with self.t2 as arr:
            self.mia = self.pool.new('mia', (no, nv), 'f8', pin=0)
            culib.contraction(
                'kc',
                self.l1x,
                'ikac',
                arr,
                'ia',
                self.mia,
                alpha=2.0)
            culib.contraction(
                'kc',
                self.l1x,
                'ikca',
                arr,
                'ia',
                self.mia,
                beta=1.0,
                alpha=-1.0)

        self.mab = self.pool.new('mab', (nv, nv), 'f8', pin=0)
        culib.contraction('kc', self.l1x, 'kb', self.t1, 'cb', self.mab)

        lib.free_all_blocks()
        gc.collect()

        self.pvoOV = self.pool.new('pvOOv', (nv, no, no, nv), 'f8', pin=1)
        self.pvoOV[:] = 0
        with self.pvoOV as arr_pvoOV:
            free_size = self.pool.free_memory / 8
            slice_len = min(no, int(free_size / (2 * no * nv * nv)))
            # slice_len = 2
            slice_o = [slice(i[0], i[1]) for i in prange(0, no, slice_len)]
            for so in slice_o:
                with self.l2x[so] as arr_l2:
                    with self.t2[so] as arr_t2:
                        culib.contraction(
                            'kiac',
                            arr_l2,
                            'kjcb',
                            arr_t2,
                            'aijb',
                            arr_pvoOV,
                            beta=1.0,
                            alpha=-1.0)
                        culib.contraction(
                            'kica',
                            arr_l2,
                            'kjcb',
                            arr_t2,
                            'aijb',
                            arr_pvoOV,
                            beta=1.0,
                            alpha=2.0)
                        culib.contraction(
                            'kica',
                            arr_l2,
                            'kjbc',
                            arr_t2,
                            'aijb',
                            arr_pvoOV,
                            beta=1.0,
                            alpha=-1.0)

            culib.contraction(
                'aljb',
                arr_pvoOV,
                'ab',
                cupy.eye(nv),
                'jl',
                self.moo,
                beta=1.0)
            culib.contraction(
                'bijd',
                arr_pvoOV,
                'ij',
                cupy.eye(no),
                'db',
                self.mvv,
                beta=1.0)

        self.mij = self.pool.new('mij', (no, no), 'f8', pin=0)
        self.mij[:] = self.moo
        self.mij *= 0.5
        culib.contraction(
            'kc',
            self.l1x,
            'jc',
            self.t1,
            'jk',
            self.mij,
            beta=1.0)
        lib.free_all_blocks()
        gc.collect()

        return

    def get_cumulant_energy_oooo(self):
        e_oooo = 0

        with self.goooo as arr_goooo:
            with self.eris.Loo as arr_Loo:
                buffer_Loo = self.pool.empty(arr_Loo.shape, dtype='f8')
                gooL_tmp = culib.contraction(
                    'ijkl', arr_goooo, 'Lik', arr_Loo, 'Ljl', buf=buffer_Loo, alpha=2.0)
                culib.contraction(
                    'ijkl',
                    arr_goooo,
                    'Ljk',
                    arr_Loo,
                    'Lil',
                    gooL_tmp,
                    beta=1.0,
                    alpha=-1.0)
                e_oooo += cupy.dot(gooL_tmp.ravel(), arr_Loo.ravel().T)
                e_oooo *= 4

        buffer_Loo = gooL_tmp = None
        lib.free_all_blocks()
        gc.collect()

        return e_oooo.item()

    def get_cumulant_energy_ooov(self):
        no, nv = self.t1.shape
        nL = self.eris.naux
        e_ooov = 0

        gooov = self.pool.empty((no, no, no, nv), dtype='f8')
        gooov[:] = 0

        with self.goooo as arr_goooo:
            culib.contraction(
                'la',
                self.t1,
                'jkil',
                arr_goooo,
                'jkia',
                gooov,
                beta=1.0,
                alpha=2.0)

        lib.free_all_blocks()

        free_size = self.pool.free_memory / 8
        slice_len = min(nv, int(free_size / (no * no * nv)))
        # slice_len = 2
        slice_v = [slice(i[0], i[1]) for i in prange(0, nv, slice_len)]
        buffer_voov = self.pool.empty((slice_len, no, no, nv), dtype='f8')
        for sv in slice_v:
            t1_tmp = self.t1[:, sv].copy()
            arr_pvOOv = self.pvOOv[sv].ascupy(buf=buffer_voov)
            culib.contraction(
                'kc',
                t1_tmp,
                'cija',
                arr_pvOOv,
                'jkia',
                gooov,
                beta=1.0)
            arr_pvoOV = self.pvoOV[sv].ascupy(buf=buffer_voov)
            culib.contraction(
                'jc',
                t1_tmp,
                'cika',
                arr_pvoOV,
                'jkia',
                gooov,
                beta=1.0,
                alpha=-1.0)

        arr_pvOOv = arr_pvoOV = t1_tmp = buffer_voov = None
        lib.free_all_blocks()

        free_size = self.pool.free_memory / 8
        slice_len = min(no, int(free_size / (no * nv * nv)))
        # slice_len = 2
        slice_o = [slice(i[0], i[1]) for i in prange(0, no, slice_len)]
        buffer_oovv = self.pool.empty((slice_len, no, nv, nv), dtype='f8')
        for so in slice_o:
            arr_t2 = self.t2[so].ascupy(buf=buffer_oovv)
            culib.contraction(
                'ia',
                self.t1[so],
                'jb',
                self.t1,
                'ijab',
                arr_t2,
                beta=1.0)
            culib.contraction(
                'ib',
                self.l1x,
                'jkba',
                arr_t2,
                'jkia',
                gooov[so],
                beta=1.0,
                alpha=-1.0)
            arr_l2 = self.l2x[so].ascupy(buf=buffer_oovv)
            culib.contraction(
                'jkba',
                arr_l2,
                'ib',
                self.t1,
                'jkia',
                gooov[so],
                beta=1.0,
                alpha=-1.0)

        buffer_oovv = arr_t2 = arr_l2 = None

        culib.contraction(
            'ji',
            self.moo,
            'ka',
            self.t1,
            'jkia',
            gooov,
            beta=1.0,
            alpha=-0.5)
        buffer_Loo = self.pool.empty((nL, no, no), dtype='f8')
        with self.eris.Lov as Lov:
            gooL_tmp = culib.contraction(
                'ijka', gooov, 'Lja', Lov, 'Lik', buf=buffer_Loo, alpha=2.0)
            culib.contraction(
                'ijka',
                gooov,
                'Lia',
                Lov,
                'Ljk',
                gooL_tmp,
                beta=1.0,
                alpha=-1.0)

        with self.eris.Loo as Loo:
            e_ooov += cupy.dot(Loo.ravel(), gooL_tmp.ravel().T)

        gooL_tmp = buffer_Loo = gooov = None

        return e_ooov.item() * 4

    def get_cumulant_energy_ovov(self):
        no, nv = self.t1.shape
        nL = self.eris.naux
        e_ovov = 0

        goovv = self.pool.new('goovv', (no, no, nv, nv), dtype='f8', pin=1)
        goovv[:] = 0
        with goovv as arr:
            culib.contraction(
                'ia',
                self.mia,
                'jb',
                self.t1,
                'ijab',
                arr,
                beta=1.0)

        buffer_oovv = self.pool.empty((no, no, nv, nv), 'f8')
        tau = self.t2.ascupy(buf=buffer_oovv)

        culib.contraction('ia', self.t1, 'jb', self.t1, 'ijab', tau, beta=1.0)
        tau *= 0.5

        with self.goooo as arr_goooo:
            free_size = self.pool.free_memory / 8
            slice_len = min(no, int(free_size / (no * nv * nv)))
            # slice_len = 2
            slice_o = [slice(i[0], i[1]) for i in prange(0, no, slice_len)]
            for so in slice_o:
                with goovv[so] as arr_goovv:
                    culib.contraction(
                        'ijkl',
                        arr_goooo[so],
                        'klab',
                        tau,
                        'ijab',
                        arr_goovv,
                        beta=1.0,
                        alpha=2.0)
                    culib.contraction(
                        'jk',
                        self.mij,
                        'ikab',
                        tau[so],
                        'ijab',
                        arr_goovv,
                        beta=1.0,
                        alpha=-2.0)
                    culib.contraction(
                        'bd',
                        self.mvv,
                        'ijad',
                        tau[so],
                        'ijab',
                        arr_goovv,
                        beta=1.0,
                        alpha=-1.0)
                    arr_goovv += tau[so]

        self.goooo = None

        free_size = self.pool.free_memory / 8
        slice_len = min(no, int(free_size / (no * nv * nv * 2)))
        # slice_len = 2
        slice_o = [slice(i[0], i[1]) for i in prange(0, no, slice_len)]

        buffer_tau = self.pool.empty((slice_len, no, nv, nv), dtype='f8')

        for so in slice_o:
            with goovv[so] as arr_goovv:
                tau = self.t2[so].ascupy(buf=buffer_tau)
                arr_pvOOv = self.pvOOv.ascupy(buf=buffer_oovv)
                culib.contraction(
                    'dljb',
                    arr_pvOOv,
                    'ilad',
                    tau,
                    'ijab',
                    arr_goovv,
                    beta=1.0,
                    alpha=0.5)
                culib.contraction(
                    'cb',
                    self.mab,
                    'ijac',
                    tau,
                    'ijab',
                    arr_goovv,
                    beta=1.0,
                    alpha=-1.0)
                tau *= 0.5
                culib.contraction(
                    'ia',
                    self.t1[so],
                    'jb',
                    self.t1,
                    'ijab',
                    tau,
                    beta=1.0)
                culib.contraction(
                    'dlja',
                    arr_pvOOv,
                    'ildb',
                    tau,
                    'ijab',
                    arr_goovv,
                    beta=1.0)

        for so in slice_o:
            with goovv[so] as arr_goovv:
                tau = self.t2[so].ascupy(buf=buffer_tau)
                arr_pvoOV = self.pvoOV.ascupy(buf=buffer_oovv)
                culib.contraction(
                    'dljb',
                    arr_pvoOV,
                    'ilad',
                    tau,
                    'ijab',
                    arr_goovv,
                    beta=1.0)
                tau *= 0.5
                culib.contraction(
                    'ia',
                    self.t1[so],
                    'jb',
                    self.t1,
                    'ijab',
                    tau,
                    beta=1.0)
                culib.contraction(
                    'dljb',
                    arr_pvoOV,
                    'ilda',
                    tau,
                    'ijab',
                    arr_goovv,
                    beta=1.0,
                    alpha=-1.0)

        tau = buffer_tau = None
        lib.free_all_blocks()

        lib.axpy(self.l2x, goovv, 0.5)

        goovv = goovv.ascupy(buf=buffer_oovv)
        g_tmp = self.pool.empty((nL, no, nv), dtype='f8')
        g_tmp[:] = 0
        with self.eris.Lov as Lov:
            culib.contraction(
                'ijab',
                goovv,
                'Lia',
                Lov,
                'Ljb',
                g_tmp,
                beta=1.0,
                alpha=2.0)
            culib.contraction(
                'ijab',
                goovv,
                'Lja',
                Lov,
                'Lib',
                g_tmp,
                beta=1.0,
                alpha=-1.0)
            e_ovov += cupy.dot(g_tmp.ravel(), Lov.ravel().T)

        goovv = g_tmp = buffer_oovv = None
        lib.free_all_blocks()

        return e_ovov.item() * 4

    def get_cumulant_energy_ovvo_oovv(self):

        no, nv = self.t1.shape
        nL = self.eris.naux
        e_ovvo = 0
        e_oovv = 0

        gvOOv = self.pool.new('gvOOv', (nv, no, no, nv), dtype='f8', pin=0)
        gvOOv[:] = 0

        free_size = self.pool.free_memory / 8
        slice_len = min(no, int(free_size / (no * nv * nv * 2)))
        # slice_len = 2
        slice_o = [slice(i[0], i[1]) for i in prange(0, no, slice_len)]

        buffer_tmp = self.pool.empty((slice_len, no, nv, nv), dtype='f8')
        buffer_tmp2 = self.pool.empty((slice_len, no, nv, nv), dtype='f8')
        for so in slice_o:
            tmp = culib.contraction(
                'jc',
                self.t1,
                'kb',
                self.t1[so],
                'jckb',
                buf=buffer_tmp)
            l2tmp = self.l2x[so].ascupy(buf=buffer_tmp2)
            culib.contraction(
                'kiac',
                l2tmp,
                'jckb',
                tmp,
                'aijb',
                gvOOv,
                beta=1.0)
        buffer_tmp = buffer_tmp2 = tmp = l2tmp = None
        lib.free_all_blocks()

        free_size = self.pool.free_memory / 8
        slice_len = min(nv, int(free_size / (no * no * nv)))
        # slice_len = 2
        slice_v = [slice(i[0], i[1]) for i in prange(0, nv, slice_len)]
        buffer_tmp = self.pool.empty((slice_len, no, no, nv), dtype='f8')
        for sv in slice_v:
            tmp = self.pvOOv[sv].ascupy(buf=buffer_tmp)
            gvOOv[sv] += tmp

        tmp = buffer_tmp = None
        lib.free_all_blocks()

        buffer_g_tmp = self.pool.empty((nL, no, max(nv, no)), dtype='f8')
        with self.eris.Lov as Lov:
            g_tmp = culib.contraction(
                'aijb', gvOOv, 'Lia', Lov, 'Ljb', buf=buffer_g_tmp)
            e_ovvo += cupy.dot(g_tmp.ravel(), Lov.ravel().T).item()
        with self.eris.Lvv as Lvv:
            g_tmp = culib.contraction(
                'aijb', gvOOv, 'Lba', Lvv, 'Lij', buf=buffer_g_tmp)
        gvOOv = None
        lib.free_all_blocks()
        with self.eris.Loo as Loo:
            e_oovv -= cupy.dot(g_tmp.ravel(), Loo.ravel().T) * 2

        buffer_g_tmp = g_tmp = None
        lib.free_all_blocks()

        gvoOV = self.pool.new('gvoOV', (nv, no, no, nv), dtype='f8')
        culib.contraction('ia', self.l1x, 'jb', self.t1, 'aijb', gvoOV)

        free_size = self.pool.free_memory / 8
        slice_len = min(no, int(free_size / (no * nv * nv * 2)))
        # slice_len = 2
        slice_o = [slice(i[0], i[1]) for i in prange(0, no, slice_len)]

        buffer_tmp = self.pool.empty((slice_len, no, nv, nv), dtype='f8')
        buffer_tmp2 = self.pool.empty((slice_len, no, nv, nv), dtype='f8')

        for so in slice_o:
            tmp = culib.contraction(
                'jc',
                self.t1,
                'kb',
                self.t1[so],
                'jckb',
                buf=buffer_tmp)
            l2tmp = self.l2x[so].ascupy(buf=buffer_tmp2)
            culib.contraction(
                'kica',
                l2tmp,
                'jckb',
                tmp,
                'aijb',
                gvoOV,
                beta=1.0,
                alpha=-1.0)

        buffer_tmp = buffer_tmp2 = tmp = l2tmp = None

        free_size = self.pool.free_memory / 8
        slice_len = min(nv, int(free_size / (no * no * nv)))
        # slice_len = 2
        slice_v = [slice(i[0], i[1]) for i in prange(0, nv, slice_len)]
        buffer_tmp = self.pool.empty((slice_len, no, no, nv), dtype='f8')
        for sv in slice_v:
            tmp = self.pvoOV[sv].ascupy(buf=buffer_tmp)
            gvoOV[sv] += tmp

        tmp = buffer_tmp = None
        lib.free_all_blocks()

        buffer_g_tmp = self.pool.empty((nL, no, max(nv, no)), dtype='f8')
        with self.eris.Lov as Lov:
            g_tmp = culib.contraction(
                'aijb', gvoOV, 'Lia', Lov, 'Ljb', buf=buffer_g_tmp)
            e_ovvo += 2 * cupy.dot(g_tmp.ravel(), Lov.ravel().T)
        with self.eris.Lvv as Lvv:
            g_tmp = culib.contraction(
                'aijb', gvoOV, 'Lba', Lvv, 'Lij', buf=buffer_g_tmp)
        with self.eris.Loo as Loo:
            e_oovv -= cupy.dot(g_tmp.ravel(), Loo.ravel().T)

        gvoOV = buffer_g_tmp = g_tmp = None

        return e_ovvo.item() * 4, e_oovv.item() * 4

    def get_cumulant_energy_vvvv_ovvv(self):

        no, nv = self.t1.shape
        nL = self.eris.naux
        e_vvvv_ovvv = 0
        lib.free_all_blocks()
        free_size = self.pool.free_memory / 8
        free_size_tmp = self.pool.free_memory / 8 - \
            max(self.t2.size, self.eris.Lvv.size, self.eris.Lov.size)
        slice_len1 = free_size_tmp / \
            (nL * nv + nv ** 2 * max(no, nv) + nv * no ** 2)
        slice_len2 = free_size / \
            (nL * no + no ** 2 * max(nv, no) + 2 * no * nv ** 2)
        slice_len = min(int(slice_len1), int(slice_len2))
        slice_len = min(slice_len, nv)
        # slice_len = 2
        assert slice_len > 0
        slice_v = [slice(i[0], i[1]) for i in prange(0, nv, slice_len)]
        buffer_cpu = self.pool.empty(max(
            no * no * slice_len * nv, nL * nv * slice_len), dtype='f8', type=lib.MemoryTypeHost)
        buffer_g_tmp = self.pool.empty((nL * slice_len * nv), dtype='f8')
        buffer_vvsv = self.pool.empty(
            (slice_len * nv ** 2 * max(no, nv)), dtype='f8')
        buffer_oovv_size = max(
            self.t2.size,
            self.eris.Lvv.size,
            self.eris.Lov.size,
            slice_len * nv ** 2 * no,
        )
        buffer_oovv = self.pool.empty((buffer_oovv_size, ), dtype='f8')
        buffer_oosv = self.pool.empty((slice_len * nv * no ** 2), dtype='f8')

        for sv in slice_v:

            sv_len = sv.stop - sv.start
            l2tmp_h = lib.empty_from_buf(
                buffer_cpu, (no, no, sv_len, nv), dtype='f8')
            cupy.cuda.get_current_stream().synchronize()
            numpy.copyto(l2tmp_h, self.l2x[:, :, sv])
            l1tmp = self.l1x[:, sv].copy()
            t1tmp = self.t1[:, sv].copy()

            g_tmp = lib.empty_from_buf(
                buffer_g_tmp, (nL, sv_len, nv), dtype='f8')
            g_tmp[:] = 0

            t2 = self.t2.ascupy(buf=buffer_oovv)

            l2tmp = lib.empty_from_buf(buffer_oosv, l2tmp_h.shape, dtype='f8')
            l2tmp.set(l2tmp_h)

            gvvvv = culib.contraction(
                'ijab', l2tmp, 'ijcd', t2, 'abcd', buf=buffer_vvsv)
            jabc = culib.contraction(
                'ijab', l2tmp, 'ic', self.t1, 'jabc', buf=buffer_oovv)

            culib.contraction(
                'jabc',
                jabc,
                'jd',
                self.t1,
                'abcd',
                gvvvv,
                beta=1.0)

            Lvv = self.eris.Lvv.ascupy(buf=buffer_oovv)

            culib.contraction(
                'abcd',
                gvvvv,
                'Lbd',
                Lvv,
                'Lac',
                g_tmp,
                beta=1.0)
            culib.contraction(
                'abcd',
                gvvvv,
                'Lcb',
                Lvv,
                'Lad',
                g_tmp,
                beta=1.0,
                alpha=-0.5)

            gvovv_tmp = culib.contraction(
                'adbc',
                gvvvv,
                'id',
                self.t1,
                'aibc',
                alpha=-1.0,
                buf=buffer_oovv)
            gvovv = lib.empty_from_buf(
                buffer_vvsv, gvovv_tmp.shape, dtype='f8')
            cupy.copyto(gvovv, gvovv_tmp)
            pvoOV = self.pvoOV[sv].ascupy(buf=buffer_oovv)

            culib.contraction(
                'akic',
                pvoOV,
                'kb',
                self.t1,
                'aibc',
                gvovv,
                beta=1.0)
            pvOOv = self.pvOOv[sv].ascupy(buf=buffer_oovv)

            culib.contraction(
                'akib',
                pvOOv,
                'kc',
                self.t1,
                'aibc',
                gvovv,
                beta=1.0,
                alpha=-1.0)
            t2 = self.t2.ascupy(buf=buffer_oovv)

            culib.contraction('ja', l1tmp, 'jibc', t2, 'aibc', gvovv, beta=1.0)
            l2 = self.l2x.ascupy(buf=buffer_oovv)

            culib.contraction('ja', t1tmp, 'jibc', l2, 'aibc', gvovv, beta=1.0)
            lt1tmp = culib.contraction('ja', l1tmp, 'jb', self.t1, 'ab')

            culib.contraction(
                'ab',
                lt1tmp,
                'ic',
                self.t1,
                'aibc',
                gvovv,
                beta=1.0)
            mvv_tmp = self.mvv[:, sv].copy() * 0.5

            culib.contraction(
                'ba',
                mvv_tmp,
                'ic',
                self.t1,
                'aibc',
                gvovv,
                beta=1.0)
            Lov = self.eris.Lov.ascupy(buf=buffer_oovv)
            culib.contraction(
                'aibc',
                gvovv,
                'Lic',
                Lov,
                'Lab',
                g_tmp,
                beta=1.0,
                alpha=2.0)
            culib.contraction(
                'aibc',
                gvovv,
                'Lib',
                Lov,
                'Lac',
                g_tmp,
                beta=1.0,
                alpha=-1.0)

            cupy.cuda.get_current_stream().synchronize()
            Lvv_tmp_h = lib.empty_from_buf(
                buffer_cpu, (nL, sv_len, nv), dtype='f8')
            numpy.copyto(Lvv_tmp_h, self.eris.Lvv[:, sv])

            Lvv_tmp = lib.empty_from_buf(
                buffer_oovv, (nL, sv_len, nv), dtype='f8')
            Lvv_tmp.set(Lvv_tmp_h)
            e_vvvv_ovvv += cupy.dot(g_tmp.ravel(), Lvv_tmp.ravel().T)

        return e_vvvv_ovvv.item() * 4
