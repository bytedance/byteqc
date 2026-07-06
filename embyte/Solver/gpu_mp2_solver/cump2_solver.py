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

import os
import gc
from .... import lib
from pyscf.lib import prange
from pyscf import gto
from byteqc.embyte.Tools.tool_lib import fix_orbital_sign
from byteqc.embyte.ERI import eri_trans
from multiprocessing import Pool
import numpy
import cupyx
import cupy
cupy.cuda.set_pinned_memory_allocator(None)


def div_t2(t2, A, B, C, D):
    kernel = cupy.ElementwiseKernel(
        'raw T A, raw T B, raw T C, raw T D, int64 w, int64 z, int64 y',
        'T out',
        '''
        size_t p = i / y;
        size_t q = (i % y) / z;
        size_t r = (i % z) / w;
        size_t s = i % w;

        double denom = A[p] + B[q] - C[r] - D[s];
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


class GPU_MP2Solver():
    '''
    GPU accelerated MP2 solver.
    '''

    def __init__(self):
        self.__name__ = 'GPU_MP2Solver'

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

        Logger.info('==== MP2 solver has been used here ====')

        nocc = round(nelec_high // 2)

        LOEO, self.orb_energy, self.EOMO, self.LOMO, self.AOMO = self.get_coeff(
            LOBNO, cluster_list, low_level_info)
        self.projector = self.EOMO[:, :nocc].T[:, :nfrag].copy()
        mol_frag = gto.Mole()
        mol_frag.build(verbose=0)
        mol_frag.atom.append(('C', (0, 0, 0)))
        mol_frag.nelectron = nelec_high
        mol_frag.incore_anyway = True

        self.nocc = nocc
        self.nvir = (self.AOMO.shape[1] - nocc)
        self.AOMO = self.AOMO

        lib.free_all_blocks()
        gc.collect()

    def get_eri(self, low_level_info, eri_path, save_or_load):
        '''
        Load/Generate a CDERI for cluster.
        '''
        if save_or_load:
            if self.eri_file is None:
                self.eri_general = eri_trans.eri_high_level_solver_incore(
                    low_level_info.mol_full,
                    low_level_info.auxmol,
                    self.AOMO[:, :self.nocc],
                    self.AOMO[:, self.nocc:],
                    low_level_info.j2c,
                    self.Logger,
                    solver_type='MP2',
                    vhfopt=self.vhfopt3c,
                    svd_tol=1e-4)
            else:
                self.eri_general = eri_trans.eri_ondisk_high_level_solver_incore(
                    low_level_info.mol_full,
                    self.eri_file,
                    self.AOMO[:, :self.nocc],
                    self.AOMO[:, self.nocc:],
                    self.Logger,
                    solver_type='MP2',
                    svd_tol=1e-4)
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

        else:
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

    def get_cluster_coeff(self):

        return self.EOMO, self.LOMO

    def get_coeff(self, LOBNO, cluster_list, low_level_info):
        LOEO = cupy.asarray(LOBNO[:, cluster_list])
        Fock_clu = cupy.einsum(
            'ip, jq, ij -> pq',
            cupy.asarray(LOEO),
            cupy.asarray(LOEO),
            cupy.asarray(
                low_level_info.fock_LO))
        orb_energy, EOMO = cupy.linalg.eigh(Fock_clu)
        EOMO = fix_orbital_sign(EOMO)
        LOMO = cupy.dot(LOEO, EOMO)
        AOMO = cupy.dot(low_level_info.AOLO, LOMO)

        return LOEO, orb_energy, EOMO, LOMO, AOMO

    def kernel(self):
        # Here kernel only for sync the code sturcture for CC
        return

    def get_truncation_T(self, if_RDM=True):

        self.projector = cupy.asarray(self.projector)

        nocc, nfrag = self.projector.shape
        nvir = self.nvir
        if if_RDM:
            self.t2c = cupyx.zeros_pinned((nfrag, nocc, nvir, nvir))
        ovL = self.eri_general.reshape(nocc, nvir, -1)
        naux = ovL.shape[-1]

        occ_energy = self.orb_energy[: nocc]
        vir_energy = self.orb_energy[nocc:]

        avail_mem = lib.gpu_avail_bytes() / 8 - nfrag * nvir * naux * 2

        a = nvir ** 2
        b = naux * nvir * 2 + nfrag * nvir ** 2
        c = -1 * avail_mem

        occslice_len = int(
            (-1 * b + numpy.sqrt((b ** 2 - 4 * a * c))) / (2 * a))
        occslice_len = min(occslice_len, nocc)
        assert occslice_len > 0, 'Gpu memery is not enough, please reduce the cluster by inceaseing the BNO threshold.'

        occslice_list = [slice(i[0], i[1])
                         for i in prange(0, nocc, occslice_len)]
        fvL = cupy.zeros((nfrag, nvir, naux), 'f8')
        fvL_tmp = cupy.zeros((nfrag, nvir, naux), 'f8')
        buffer_ia_d = cupy.empty((occslice_len, nvir, naux), 'f8')
        buffer_jb_d = cupy.empty((occslice_len, nvir, naux), 'f8')
        buffer_t2_d = cupy.empty(
            (occslice_len, occslice_len, nvir, nvir), 'f8')
        buffer_t2_c_d = cupy.empty((occslice_len, nfrag, nvir, nvir), 'f8')
        buffer_t2_c_h = cupyx.empty_pinned(buffer_t2_c_d.shape, 'f8')

        for so1_ind, so1 in enumerate(occslice_list):
            so1_len = so1.stop - so1.start
            is_aL = lib.empty_from_buf(
                buffer_ia_d, (so1_len, nvir, naux), 'f8')
            is_aL.set(ovL[so1])
            lib.contraction(
                'iaL',
                is_aL,
                'if',
                self.projector[so1],
                'faL',
                fvL,
                beta=1.0)
            t2 = lib.contraction(
                'iaL',
                is_aL,
                'jbL',
                is_aL,
                'ijab',
                buf=buffer_t2_d)

            div_t2(
                t2,
                occ_energy[so1],
                occ_energy[so1],
                vir_energy,
                vir_energy)
            t2_c = lib.contraction(
                'ijab',
                t2,
                'if',
                self.projector[so1],
                'fjab',
                buf=buffer_t2_c_d)
            if if_RDM:
                tmp_t2_c_h = lib.empty_from_buf(
                    buffer_t2_c_h, t2_c.shape, 'f8')
                t2_c.get(out=tmp_t2_c_h, blocking=True)

            lib.contraction(
                'jbL',
                is_aL,
                'fjab',
                t2_c,
                'faL',
                fvL_tmp,
                alpha=2.0,
                beta=1.0)
            lib.contraction(
                'jaL',
                is_aL,
                'fjab',
                t2_c,
                'fbL',
                fvL_tmp,
                alpha=-1.0,
                beta=1.0)

            if if_RDM:
                self.t2c[:, so1] += tmp_t2_c_h
            cupy.cuda.get_current_stream().synchronize()

            occslice_list2 = occslice_list[so1_ind + 1:]

            for so2_ind, so2 in enumerate(occslice_list2):

                so2_len = so2.stop - so2.start
                js_bL = lib.empty_from_buf(
                    buffer_jb_d, (so2_len, nvir, naux), 'f8')
                js_bL.set(ovL[so2])
                t2 = lib.contraction(
                    'iaL', is_aL, 'jbL', js_bL, 'ijab', buf=buffer_t2_d)
                div_t2(
                    t2,
                    occ_energy[so1],
                    occ_energy[so2],
                    vir_energy,
                    vir_energy)

                t2_c = lib.contraction(
                    'ijab',
                    t2,
                    'if',
                    self.projector[so1],
                    'fjab',
                    buf=buffer_t2_c_d)
                if if_RDM:
                    tmp_t2_c_h = lib.empty_from_buf(
                        buffer_t2_c_h, t2_c.shape, 'f8')
                    t2_c.get(out=tmp_t2_c_h, blocking=True)

                lib.contraction(
                    'jbL',
                    js_bL,
                    'fjab',
                    t2_c,
                    'faL',
                    fvL_tmp,
                    alpha=2.0,
                    beta=1.0)
                lib.contraction(
                    'jaL',
                    js_bL,
                    'fjab',
                    t2_c,
                    'fbL',
                    fvL_tmp,
                    alpha=-1.0,
                    beta=1.0)

                if if_RDM:
                    self.t2c[:, so2] += tmp_t2_c_h
                cupy.cuda.get_current_stream().synchronize()

                t2_c = lib.contraction(
                    'ijab',
                    t2,
                    'jf',
                    self.projector[so2],
                    'fiba',
                    buf=buffer_t2_c_d)
                if if_RDM:
                    tmp_t2_c_h = lib.empty_from_buf(
                        buffer_t2_c_h, t2_c.shape, 'f8')
                    t2_c.get(out=tmp_t2_c_h, blocking=True)

                lib.contraction(
                    'iaL',
                    is_aL,
                    'fiba',
                    t2_c,
                    'fbL',
                    fvL_tmp,
                    alpha=2.0,
                    beta=1.0)
                lib.contraction(
                    'ibL',
                    is_aL,
                    'fiba',
                    t2_c,
                    'faL',
                    fvL_tmp,
                    alpha=-1.0,
                    beta=1.0)
                if if_RDM:
                    self.t2c[:, so1] += tmp_t2_c_h
                cupy.cuda.get_current_stream().synchronize()

            self.Logger.info(
                f'MP2 high-level solver get_truncation_T nocc:[{so1.start}:{so1.stop}]/{nocc}')

        self.e_corr = cupy.dot(fvL.ravel(), fvL_tmp.ravel().T).item()

        buffer_ia_d = buffer_jb_d = buffer_t2_d = buffer_t2_c_d = buffer_t2_c_h = fvL = fvL_tmp = None
        is_aL = t2 = t2_c = tmp_t2_c_h = js_bL = None

    def get_frag_correlation_energy(self, if_RDM=True):
        if if_RDM:
            return self.e_corr, self.e_corr * 2
        else:
            return self.e_corr, None
