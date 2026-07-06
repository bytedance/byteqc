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

import gc
from byteqc import lib
from byteqc.embyte.Tools.tool_lib import fix_orbital_sign
from byteqc.embyte.Tools.logger import Logger, Process_Record_cluster
import os
import h5py
from functools import reduce
from byteqc.embyte.Framework.SIE_helper import Get_bath, Impurity_1rdm
import time
from byteqc.embyte.Framework.BNO_building import SIE_BNO_builder
from byteqc.embyte.Tools.symm_operation import SymmetryRotation, SymmetryReflection, SymmetryInversion
import numpy
import shutil
import cupy
cupy.cuda.set_pinned_memory_allocator(None)


class high_level_processing:

    '''
    Perpare the cluster information for high-level calculation.
    The cluster information include:
        1. get bath orbitals for fragment.
        2. get BNO for fragment+bath and finally form cluster.
        3. solve cluster at MP2/CCSD/CCSD(T) level.
        4. in-cluster calculated 2-RDM correlation energy contribution if if_RDM is True.
        5. save t2/l2 for global 1-RDM calcualtion if if_RDM is True.
    '''

    def __init__(self, orb_list, low_level_info,
                 electronic_structure_solver, threshold, logfile, test=False):

        self.low_level_info_add = low_level_info
        self.orb_list = orb_list
        self.threshold = threshold
        self.electronic_structure_solver = electronic_structure_solver
        self.logfile = logfile
        self.RDM = False
        self.fragments = None
        self.cheat_th = None
        self.vhfopt = None
        self.in_situ_T = True
        self.eri = None
        self.equivalent_list = None

    def kernel(self, cluster_index_i):
        '''
        Do the high-level calculation for the fragment with the index of cluster_index_i.
        '''
        import pickle
        f = open(self.low_level_info_add, 'rb')
        self.low_level_info = pickle.loads(f.read())
        f.close()

        lib.free_all_blocks()

        cheat_th = None
        if self.cheat_th is not None and cluster_index_i in self.cheat_th:
            cheat_th = self.cheat_th[cluster_index_i]

        self.logfile_i = self.logfile + '/Cluster_%s' % cluster_index_i
        if not os.path.exists(self.logfile_i):
            os.mkdir(self.logfile_i)

        self.LG = Logger(self.logfile_i + '/cluster_log.log')
        self.PR = Process_Record_cluster(self.logfile_i)

        self.LG.logger.info(
            '=== Bath orbitals ranking for cluster %d begin' %
            cluster_index_i)

        self.cluster_index_i = cluster_index_i

        self.equi_frag = [tmp_ind for tmp_ind, value in enumerate(
            self.equivalent_list) if value == self.equivalent_list[cluster_index_i]]

        if self.RDM:
            self.symm_op = []
            for equ_ind in self.equi_frag:
                self.symm_op += self.fragments[equ_ind]['equivalent_operator']

                if not os.path.exists(self.logfile + '/Cluster_%s' % equ_ind):
                    os.mkdir(self.logfile + '/Cluster_%s' % equ_ind)
            self.symm_op_t = []
            for op_str_tmp in self.symm_op:
                if op_str_tmp not in self.symm_op_t:
                    self.symm_op_t.append(op_str_tmp)
            self.symm_op = self.symm_op_t

            self.symm_op_class = []
            for op in self.symm_op:
                if op == 'main':
                    self.symm_op_class.append(op)
                elif 'Roatation' in op:
                    Rotation_op = SymmetryRotation(
                        self.low_level_info.mol_full, op[1])
                    self.symm_op_class.append(Rotation_op)
                elif 'Reflection' in op:
                    Reflection_op = SymmetryReflection(
                        self.low_level_info.mol_full, op[1])
                    self.symm_op_class.append(Reflection_op)
                elif 'Inversion' in op:
                    Inversion_op = SymmetryInversion(
                        self.low_level_info.mol_full)
                    self.symm_op_class.append(Inversion_op)

        if not self.PR.recorder['stage']['0']:

            frag_orb_index = self.orb_list[cluster_index_i]

            frag_bath_size = []
            frag_bath_size.append(len(frag_orb_index))

            self.LG.logger.info('= Get bath and embedding orbitals cccupation')
            LOEO, EO_occupation, frag_bath_size, frag_bath_nelectron = Get_bath(
                self.low_level_info.mol_full, frag_bath_size, frag_orb_index,
                self.low_level_info.onerdm_low)
            self.LG.logger.info(
                '= Get bath and embedding orbitals cccupation Done!')

            core_orb_ind = numpy.argwhere(numpy.isclose(EO_occupation, 2))
            vir_orb_ind = numpy.argwhere(numpy.isclose(
                EO_occupation[round(numpy.sum(frag_bath_size)):], 0)) + round(numpy.sum(frag_bath_size))
            core_orb_ind = core_orb_ind.ravel()
            vir_orb_ind = vir_orb_ind.ravel()
            EO_occupation[core_orb_ind] = 2
            EO_occupation[vir_orb_ind] = 0

            norb_fb = frag_bath_size[0] + frag_bath_size[1]

            init_cluster_fock = cupy.asarray(reduce(
                cupy.dot, (LOEO[:, :norb_fb].T, self.low_level_info.fock_LO, LOEO[:, :norb_fb])))
            init_cluster_mo_energy, init_cluster_mo_coeff = cupy.linalg.eigh(
                init_cluster_fock)
            init_cluster_mo_coeff = fix_orbital_sign(init_cluster_mo_coeff)
            init_cluster_nele = round(
                self.low_level_info.mol_full.nelectron - numpy.sum(EO_occupation))

            del init_cluster_fock

            self.LG.logger.info(
                '=== Start to build BNO')

            LOEO = cupy.asarray(LOEO)
            LO_init_cluster_MO = cupy.dot(
                LOEO[:, :norb_fb], init_cluster_mo_coeff)
            self.cluster_mo_coeff = init_cluster_mo_coeff.copy()
            LO_init_cluster_MO = cupy.hstack(
                (LO_init_cluster_MO, LOEO[:, norb_fb:]))

            EO_occupation = numpy.asarray(EO_occupation)

            EO_occupation[: round(init_cluster_nele // 2)] = 2

            cluster_list = list(range(norb_fb))
            LOBNO = LOEO.copy()
            ele_diff = []

            if cheat_th is not None:
                threshold_cluster = cheat_th
            else:
                threshold_cluster = self.threshold

            LOBNO, ele_diff = SIE_BNO_builder(self.low_level_info, frag_bath_size, LOEO.get(
            ), EO_occupation, vhfopt=self.vhfopt, logger=self.LG.logger, eri=self.eri)
            LOEO = None
            LOBNO = cupy.asarray(LOBNO)

            del LO_init_cluster_MO, threshold_cluster

            new_index = (-numpy.array(ele_diff)).argsort()
            LOBNO[:, norb_fb:] = LOBNO[:, norb_fb:][:, new_index]

            ele_diff = numpy.array(ele_diff)[new_index]
            EO_occupation = list(EO_occupation[:norb_fb]) + list(
                numpy.array(EO_occupation[norb_fb:])[new_index])

            del init_cluster_mo_coeff

            self.PR.save_obj(LOBNO, 'LOBNO')
            self.PR.save_obj(ele_diff, 'ele_diff')
            self.PR.save_obj(EO_occupation, 'EO_occupation')
            self.PR.save_obj(cluster_list, 'cluster_list')
            self.PR.save_obj(self.cluster_mo_coeff, 'mf_fb_mo_coeff')
            self.PR.save_obj(frag_bath_size, 'frag_bath_size')
            self.PR.recorder['norb_fb'] = str(norb_fb)
            self.PR.recorder['stage']['0'] = True
            th_len = len(
                self.threshold) if isinstance(
                self.threshold,
                list) else 1
            self.PR.recorder['high_level_solved'] = [False] * th_len
            self.PR.recorder['eri_path'] = [None] * th_len
            self.PR.recorder['orb_used'] = [0] * th_len
            self.PR.recorder['CI_correlation_energy'] = [0] * th_len
            self.PR.recorder['solver_finish'] = [False] * th_len
            if 'CCSD' in self.electronic_structure_solver.__name__ and self.in_situ_T:
                self.PR.recorder['T_correction'] = [0] * th_len
            if self.RDM:
                self.PR.recorder['cumulant_energy'] = [0] * th_len
            self.PR.save()
            self.LG.logger.info(
                '--- Save Check point: Selection for extension bath finish')

        else:
            frag_bath_size = self.PR.load_class(
                self.PR.recorder['frag_bath_size'])
            LOBNO = cupy.asarray(self.PR.load_class(self.PR.recorder['LOBNO']))
            ele_diff = self.PR.load_class(self.PR.recorder['ele_diff'])
            EO_occupation = self.PR.load_class(
                self.PR.recorder['EO_occupation'])
            cluster_list = self.PR.load_class(self.PR.recorder['cluster_list'])
            norb_fb = int(self.PR.recorder['norb_fb'])
            self.cluster_mo_coeff = self.PR.load_class(
                self.PR.recorder['mf_fb_mo_coeff'])
            self.LG.logger.info(
                '=== Load Check point: Selection for extension bath')

        if cheat_th is not None:
            if isinstance(cheat_th, list) or isinstance(self.threshold, list):
                assert len(self.threshold) == len(cheat_th), \
                    f'cluster special threshold list len {len(cheat_th)} mismatchs with the defualt {len(self.threshold)}'
            self.LG.logger.info(
                '--- The special threshold %s has been set for cluster %s to instead of original thread %s' %
                (cheat_th, cluster_index_i, self.threshold))
            threshold_cluster = cheat_th
        else:
            threshold_cluster = self.threshold

        cluster_list_test = []
        EO_env_occupation = numpy.asarray(EO_occupation)[norb_fb:].copy()
        core_index = numpy.where(numpy.isclose(EO_env_occupation, 2))[0]
        vir_index = numpy.where(numpy.isclose(EO_env_occupation, 0))[0]
        for th in threshold_cluster:
            core_th, vir_th = th
            core_index_take = (core_index[numpy.where(numpy.asarray(ele_diff)[core_index]
                                                      > core_th)[0]].copy() + norb_fb).tolist()
            vir_index_take = (vir_index[numpy.where(numpy.asarray(ele_diff)[vir_index]
                                                    > vir_th)[0]].copy() + norb_fb).tolist()
            cluster_list_temp = cluster_list + core_index_take + vir_index_take
            cluster_list_test.append(cluster_list_temp)

        self.LG.logger.info(
            '=== Bath orbital ranking end at %s' %
            time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime()))

        self.LG.logger.info(
            '=== Run High Level Solver for cluster %d' %
            (cluster_index_i))
        frag_corr_list = []
        used_orb_list = []

        for th_index in range(len(threshold_cluster)):

            th = threshold_cluster[th_index]
            th_str = 'occ_1e%.1f_vir_1e%.1f' % (numpy.log10(th[0]), numpy.log10(th[1]))
            cluster_list = cluster_list_test[th_index]
            self.PR.recorder['orb_used'][th_index] = len(cluster_list)
            self.PR.save()

            if self.PR.recorder['high_level_solved'][th_index]:
                continue

            if th_index != 0:
                if threshold_cluster[th_index] == threshold_cluster[th_index - 1]:
                    self.PR.recorder['cumulant_energy'][th_index] = self.PR.recorder['cumulant_energy'][th_index - 1]
                    self.PR.recorder['high_level_solved'][th_index] = True
                    self.PR.recorder['eri_path'][th_index] = self.PR.recorder['eri_path'][th_index - 1]
                    self.PR.save()
                    continue

            self.LG.logger.info(
                'The cluster %d orbitals: %d / %d in the threshold %s' %
                (cluster_index_i, len(cluster_list), LOBNO.shape[0], th_str))

            nelec_high, rdm1_core, rdm1_core_coeff = Impurity_1rdm(
                cluster_list, LOBNO, EO_occupation, self.low_level_info.mol_full.nelectron)
            if 'MP2' in self.electronic_structure_solver.__name__:
                rdm1_core_coeff = None

            if numpy.isclose(nelec_high, 0):
                self.LG.logger.info(
                    '--- There is no electron in this cluster, skip this cluster!')
                return 0, 0

            high_level_solver_frag = self.electronic_structure_solver()

            high_level_solver_frag.make_param(nelec_high,
                                              cluster_list,
                                              LOBNO,
                                              self.low_level_info,
                                              self.LG.logger,
                                              self.vhfopt,
                                              rdm1_core_coeff,
                                              frag_bath_size[0],
                                              eri_file=self.eri)

            if self.PR.recorder['eri_path'][th_index] is None:
                eri_path = os.path.join(
                    self.PR.filepath, 'eri_%s' %
                    th_str)
                if os.path.exists(eri_path):
                    shutil.rmtree(eri_path)
                os.makedirs(eri_path)
                save_or_load = True
            else:
                eri_path = self.PR.recorder['eri_path'][th_index]
                save_or_load = False

            high_level_solver_frag.get_eri(
                self.low_level_info, eri_path, save_or_load)

            if save_or_load:
                self.PR.recorder['eri_path'][th_index] = eri_path
                self.PR.save()

            mf_fragment_mo_coeff, LOMO = high_level_solver_frag.get_cluster_coeff()

            if not self.PR.recorder['solver_finish'][th_index]:
                high_level_solver_frag.kernel()
                high_level_solver_frag.get_truncation_T(if_RDM=self.RDM)

            if 'CCSD' in self.electronic_structure_solver.__name__ and self.in_situ_T and not self.PR.recorder[
                    'solver_finish'][th_index]:
                if not os.path.exists(
                        self.PR.filepath + '/th_%s/' % th_str):
                    os.makedirs(self.PR.filepath + '/th_%s/' % th_str)
                t2_T_path = self.PR.filepath + '/th_%s/' % th_str + '_t2'
                # pool_rw = multiprocessing.Pool(processes=min(lib.NumFileProcess, high_level_solver_frag.t2.shape[0]))
                file = lib.FileMp(t2_T_path, 'w')
                blk = max(
                    int(high_level_solver_frag.t2.shape[0] / lib.NumFileProcess), 1)
                t2_f = file.create_dataset(
                    't2', high_level_solver_frag.t2.shape, 'f8', blksizes=(blk))
                # wait_list = t2_f.setitem(
                #     numpy.s_[:], high_level_solver_frag.t2, pool=pool_rw)
                wait_list = t2_f.setitem(
                    numpy.s_[:], high_level_solver_frag.t2)
                for w in wait_list:
                    w.wait()
                # pool_rw.close()
                # pool_rw.join()
                file.close()

            if self.RDM and not self.PR.recorder['solver_finish'][th_index]:

                if not os.path.exists(
                        self.PR.filepath + '/th_%s/' % th_str):
                    os.makedirs(self.PR.filepath + '/th_%s/' % th_str)
                nf, _, _, _ = high_level_solver_frag.t2c.shape
                # pool_w = multiprocessing.Pool(processes=min(lib.NumFileProcess, nf))
                if 'MP2' in self.electronic_structure_solver.__name__:
                    t2_path = self.PR.filepath + '/th_%s/' % th_str + 't2'
                    file = lib.FileMp(t2_path, 'w')
                    nf, no, nv, _ = high_level_solver_frag.t2c.shape
                    t2_c_f = file.create_dataset(
                        't2', (nf * no, nv, nv), 'f8', blksizes=(no, ))
                    # wait_list = t2_c_f.setitem(
                    #     numpy.s_[:], high_level_solver_frag.t2c.reshape(
                    #         t2_c_f.shape), pool=pool_w)
                    wait_list = t2_c_f.setitem(
                        numpy.s_[:], high_level_solver_frag.t2c.reshape(
                            t2_c_f.shape))
                    for w in wait_list:
                        w.wait()
                    file.close()
                    nf = no = nv = None

                elif 'CCSD' in self.electronic_structure_solver.__name__:
                    t2_path = self.PR.filepath + '/th_%s/' % th_str + 't2'
                    t1_path = self.PR.filepath + '/th_%s/' % th_str + 't1'
                    l2_path = self.PR.filepath + '/th_%s/' % th_str + 'l2'
                    l1_path = self.PR.filepath + '/th_%s/' % th_str + 'l1'

                    def save_amp(amp_path, amp_name, amp, logger):
                        with h5py.File(amp_path, 'w') as f:
                            f.create_dataset(
                                amp_name, dtype='float64', shape=amp.shape)
                            f[amp_name].write_direct(amp)
                        logger.info(f'Finish saving {amp_name}')

                    try:
                        t1_h = high_level_solver_frag.t1.asnumpy()
                        l1_h = high_level_solver_frag.l1.asnumpy()
                    except BaseException:
                        t1_h = high_level_solver_frag.t1
                        l1_h = high_level_solver_frag.l1

                    save_amp(t1_path, 't1', t1_h, self.LG.logger)
                    save_amp(l1_path, 'l1', l1_h, self.LG.logger)

                    file = lib.FileMp(t2_path, 'w')
                    nf, no, nv, _ = high_level_solver_frag.t2c.shape
                    t2_c_f = file.create_dataset(
                        't2', (nf * no, nv, nv), 'f8', blksizes=(no, ))
                    # wait_list = t2_c_f.setitem(
                    #     numpy.s_[:], high_level_solver_frag.t2c.reshape(
                    #         t2_c_f.shape), pool=pool_w)
                    wait_list = t2_c_f.setitem(
                        numpy.s_[:], high_level_solver_frag.t2c.reshape(
                            t2_c_f.shape))
                    for w in wait_list:
                        w.wait()
                    file.close()

                    file = lib.FileMp(l2_path, 'w')
                    l2_c_f = file.create_dataset(
                        'l2', (nf * no, nv, nv), 'f8', blksizes=(no, ))
                    # wait_list = l2_c_f.setitem(
                    #     numpy.s_[:], high_level_solver_frag.l2c.reshape(
                    #         l2_c_f.shape), pool=pool_w)
                    wait_list = l2_c_f.setitem(
                        numpy.s_[:], high_level_solver_frag.l2c.reshape(
                            l2_c_f.shape))
                    for w in wait_list:
                        w.wait()
                    file.close()

                else:
                    raise KeyboardInterrupt('Not support for now')

                # pool_w.close()
                # pool_w.join()

                for equi_frag_ind in self.equi_frag:
                    frag_equi_op = self.fragments[equi_frag_ind]['equivalent_operator']
                    AOLO = self.low_level_info.AOLO.get().copy()
                    LOMO = self.low_level_info.LOMO.copy()
                    LO_BNO_clu = LOBNO[:, cluster_list].get()
                    AOMO = numpy.dot(AOLO, LOMO)
                    AO_BNO_clu = numpy.dot(AOLO, LO_BNO_clu)

                    for op_str in frag_equi_op:
                        if op_str == 'main':
                            continue
                        op_class = self.symm_op_class[self.symm_op.index(
                            op_str)]
                        AO_BNO_clu = op_class(AO_BNO_clu)

                    nocc_full = round(
                        self.low_level_info.mol_full.nelectron // 2)
                    AO_MO_occ = AOMO[:, : nocc_full]
                    AO_MO_vir = AOMO[:, nocc_full:]
                    AO_CLU = cupy.dot(
                        cupy.asarray(AO_BNO_clu),
                        cupy.asarray(mf_fragment_mo_coeff)).get()
                    nocc_cluster = round(nelec_high // 2)
                    AO_CLU_occ = AO_CLU[:, : nocc_cluster]
                    AO_CLU_vir = AO_CLU[:, nocc_cluster:]
                    AO_FRAG = AO_BNO_clu[:, :frag_bath_size[0]]

                    S = cupy.asarray(self.low_level_info.ao_ovlp)

                    MO_occ_CLU_occ = reduce(
                        cupy.dot,
                        (cupy.asarray(
                            AO_MO_occ.T),
                            cupy.asarray(S),
                            cupy.asarray(AO_CLU_occ))).get()
                    MO_vir_CLU_vir = reduce(
                        cupy.dot,
                        (cupy.asarray(
                            AO_MO_vir.T),
                            cupy.asarray(S),
                            cupy.asarray(AO_CLU_vir))).get()
                    CLU_occ_FRAG = reduce(
                        cupy.dot,
                        (cupy.asarray(
                            AO_CLU_occ.T),
                            cupy.asarray(S),
                            cupy.asarray(AO_FRAG))).get()
                    MO_occ_FRAG = reduce(
                        cupy.dot,
                        (cupy.asarray(
                            AO_MO_occ.T),
                            cupy.asarray(S),
                            cupy.asarray(AO_FRAG))).get()

                    LO_CLU_occ = reduce(
                        cupy.dot,
                        (cupy.asarray(
                            AOLO.T),
                            cupy.asarray(S),
                            cupy.asarray(AO_CLU_occ)))
                    LO_CLU_vir = reduce(
                        cupy.dot,
                        (cupy.asarray(
                            AOLO.T),
                            cupy.asarray(S),
                            cupy.asarray(AO_CLU_vir)))

                    try:
                        os.makedirs(
                            self.logfile
                            + '/Cluster_%s' %
                            equi_frag_ind
                            + '/th_%s' %
                            th_str)
                    except BaseException:
                        pass

                    numpy.save(
                        self.logfile
                        + '/Cluster_%s' %
                        equi_frag_ind
                        + '/th_%s' %
                        th_str
                        + '/LO_CLU_occ.npy',
                        LO_CLU_occ)
                    numpy.save(
                        self.logfile
                        + '/Cluster_%s' %
                        equi_frag_ind
                        + '/th_%s' %
                        th_str
                        + '/LO_CLU_vir.npy',
                        LO_CLU_vir)
                    numpy.save(
                        self.logfile
                        + '/Cluster_%s' %
                        equi_frag_ind
                        + '/th_%s' %
                        th_str
                        + '/MO_occ_CLU_occ.npy',
                        MO_occ_CLU_occ)
                    numpy.save(
                        self.logfile
                        + '/Cluster_%s' %
                        equi_frag_ind
                        + '/th_%s' %
                        th_str
                        + '/MO_vir_CLU_vir.npy',
                        MO_vir_CLU_vir)
                    numpy.save(
                        self.logfile
                        + '/Cluster_%s' %
                        equi_frag_ind
                        + '/th_%s' %
                        th_str
                        + '/CLU_occ_FRAG.npy',
                        CLU_occ_FRAG)
                    numpy.save(
                        self.logfile
                        + '/Cluster_%s' %
                        equi_frag_ind
                        + '/th_%s' %
                        th_str
                        + '/MO_occ_FRAG.npy',
                        MO_occ_FRAG)

                    del LO_CLU_occ, LO_CLU_vir, MO_occ_CLU_occ, MO_vir_CLU_vir
                    del CLU_occ_FRAG, MO_occ_FRAG, S, AO_FRAG
                    del AO_CLU_vir, AO_CLU_occ, AO_CLU, AO_MO_vir, AO_MO_occ
                    del AO_BNO_clu, AOMO, LO_BNO_clu, LOMO, AOLO, frag_equi_op

            if not self.PR.recorder['solver_finish'][th_index]:
                frag_corr, cumulant_energy = high_level_solver_frag.get_frag_correlation_energy(
                    if_RDM=self.RDM)

            lib.free_all_blocks()
            gc.collect()

            if self.RDM and not self.PR.recorder['solver_finish'][th_index]:
                self.PR.recorder['cumulant_energy'][th_index] = cumulant_energy
                self.PR.save()
            elif self.RDM and self.PR.recorder['solver_finish'][th_index]:
                cumulant_energy = self.PR.recorder['cumulant_energy'][th_index]

            if not self.PR.recorder['solver_finish'][th_index]:
                self.PR.recorder['CI_correlation_energy'][th_index] = frag_corr
                self.PR.save()
            else:
                frag_corr = self.PR.recorder['CI_correlation_energy'][th_index]

            self.LG.logger.info('--------------')
            if self.RDM:
                self.LG.logger.info(
                    'The cumulant energy at threshold %s from this cluster: %s' %
                    (th_str, cumulant_energy))
            self.LG.logger.info(
                'The correlation energy at threshold %s from this cluster: %s' %
                (th_str, frag_corr))
            self.LG.logger.info('--------------')

            frag_corr_list.append(frag_corr)
            used_orb_list.append(len(cluster_list))
            self.PR.recorder['solver_finish'][th_index] = True
            self.PR.save()

            if 'CCSD' in self.electronic_structure_solver.__name__ and self.in_situ_T:
                t1t2_path = os.path.join(
                    self.PR.filepath, 'th_%s' %
                    th_str)
                et = high_level_solver_frag.get_T_correction(
                    eri_path, t1t2_path)
                self.PR.recorder['T_correction'][th_index] = et
                self.PR.save()
                self.LG.logger.info('--------------')
                self.LG.logger.info(
                    'The (T) correction energy at threshold %s from this cluster: %s' %
                    (th_str, et))
                self.LG.logger.info('--------------')

            high_level_solver_frag = None
            self.PR.recorder['high_level_solved'][th_index] = True
            self.PR.save()
            shutil.rmtree(self.PR.recorder['eri_path'][th_index])
            if 'CCSD' in self.electronic_structure_solver.__name__ and self.in_situ_T:
                file = lib.FileMp(os.path.join(t1t2_path, '_t2'), 'a')
                del file['t2']
                file.close()

        self.PR.recorder['stage'][1] = True
        self.PR.save()

        return numpy.asarray(frag_corr_list), numpy.asarray(used_orb_list)
