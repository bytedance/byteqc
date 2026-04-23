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
import os
import time
from byteqc import lib
from byteqc.embyte.Localization import iao_pao_localization
from byteqc.embyte.Tools.tool_lib import fix_orbital_sign
from functools import reduce
import numpy
from byteqc.embyte.ERI import eri_trans
from byteqc.cuobc import scf
import pyscf
from pyscf import df
import cupy
cupy.cuda.set_pinned_memory_allocator(None)


def _normalize_kpts(kpts):
    if kpts is None:
        return None
    if hasattr(kpts, 'kpts'):
        kpts = kpts.kpts
    kpts = numpy.asarray(kpts, dtype=float)
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise ValueError(f'Invalid kpts shape {kpts.shape}, expected (nkpts, 3)')
    return kpts


def _kpts_match(kpts_ref, kpts_given, tol=1e-9):
    kpts_ref = _normalize_kpts(kpts_ref)
    kpts_given = _normalize_kpts(kpts_given)
    if kpts_ref is None or kpts_given is None:
        return False
    return kpts_ref.shape == kpts_given.shape and numpy.allclose(
        kpts_ref, kpts_given, atol=tol, rtol=0.0)


def _load_or_transform_pbc_ao_matrix(data_or_path, cell, kpts, kmesh, nao_sc, label):
    from pyscf.pbc.tools import k2gamma as pbc_k2gamma

    if isinstance(data_or_path, str):
        arr = numpy.load(data_or_path)
    else:
        arr = numpy.asarray(data_or_path)

    nkpts = len(kpts)
    nao = int(cell.nao_nr())
    if arr.shape == (nao_sc, nao_sc):
        return numpy.asarray(arr, dtype=numpy.float64)
    if arr.shape == (nkpts, nao, nao):
        return pbc_k2gamma.to_supercell_ao_integrals(
            cell, kpts, arr, kmesh=kmesh, force_real=True)
    raise ValueError(
        f'{label} has incompatible shape {arr.shape}; '
        f'expected either ({nao_sc}, {nao_sc}) or ({nkpts}, {nao}, {nao})'
    )


class low_level_info:
    '''
    Collect all the low level information from the mean field calculation.
    Commonly the mean field comes from the converged HF check file which is provided by pyscf.
    The low-level calculation would not be done in SIE workflow.
    '''

    def __init__(self, mol,
                 mf,
                 LG,
                 aux_basis=None,
                 jk_file=None,
                 with_eri=False,
                 oei=None,
                 local_orb_path=None,
                 ewald_correct=False,
                 kpts=None,
                 ):

        logger = LG.logger
        primitive_mol = mol
        primitive_mf = mf
        self.kpts = _normalize_kpts(kpts)
        self.kmesh = None
        self.eri_mol = primitive_mol
        self.eri_auxmol = None
        loaded_kpts = getattr(primitive_mf, 'kpts', None)
        if getattr(primitive_mol, 'pbc_intor', None) is not None and self.kpts is None:
            try:
                loaded_kpts = _normalize_kpts(loaded_kpts)
            except Exception:
                loaded_kpts = None
            if loaded_kpts is not None and len(loaded_kpts) > 1:
                raise ValueError(
                    'A multi-k mean-field / chk was detected. Please provide low_level_info(..., kpts=...)'
                )

        if getattr(primitive_mol, 'pbc_intor', None) is not None and self.kpts is not None:
            from pyscf.pbc.tools import k2gamma as pbc_k2gamma
            from byteqc.embyte.Tools import k2gamma as gpu_k2gamma

            if loaded_kpts is None:
                raise ValueError(
                    'kpts was provided, but the mean-field object does not contain k-point information'
                )
            if not _kpts_match(loaded_kpts, self.kpts):
                raise ValueError(
                    'The provided kpts does not match the kpts stored in the HF checkpoint / mean-field object'
                )

            self.kmesh = numpy.asarray(
                pbc_k2gamma.kpts_to_kmesh(primitive_mol, self.kpts),
                dtype=int,
            )
            mf = gpu_k2gamma.k2gamma(primitive_mf, kmesh=self.kmesh)
            mf.e_tot = primitive_mf.e_tot * len(self.kpts)
            mol = mf.cell
            logger.info(
                '----------- Use multi-k low-level path, kmesh=%s, nkpts=%d',
                tuple(self.kmesh.tolist()),
                len(self.kpts),
            )

        if getattr(primitive_mol, 'pbc_intor', None) is not None:
            if jk_file is None or oei is None:
                raise ValueError(
                    'Periodic systems require both jk_file and oei to be provided explicitly.'
                )

        if aux_basis is None:
            self.auxmol = df.addons.make_auxmol(mol, df.make_auxbasis(mol))
            if getattr(primitive_mol, 'pbc_intor', None) is not None and self.kpts is not None:
                self.eri_auxmol = df.addons.make_auxmol(
                    primitive_mol, df.make_auxbasis(primitive_mol))
        else:
            self.auxmol = df.addons.make_auxmol(mol, aux_basis)
            if getattr(primitive_mol, 'pbc_intor', None) is not None and self.kpts is not None:
                self.eri_auxmol = df.addons.make_auxmol(primitive_mol, aux_basis)

        if self.eri_auxmol is None:
            self.eri_auxmol = self.auxmol

        self.mol_full = mol

        mempool = cupy.get_default_memory_pool()
        mempool.free_all_blocks()

        blksize = int((lib.gpu_avail_bytes() / (8 * 2)) ** (1 / 3))
        if with_eri is False and getattr(mol, 'pbc_intor', None) is None:
            self.j2c = os.path.join(LG.filepath + 'j2c')
            lib.free_all_blocks()
            with h5py.File(self.j2c, 'w') as f:
                logger.info("--- start to get j2c")
                j2c, self.nL_un, self.nL = eri_trans.get_j2c(
                    logger, mol=self.mol_full, auxmol=self.auxmol)
                f.create_dataset('j2c', data=j2c, dtype='f8')
                del j2c
                logger.info(f'--- Save j2c to the disk {self.j2c}')
        else:

            self.j2c = None

        lib.free_all_blocks()

        if getattr(mol, 'pbc_intor', None) is not None and ewald_correct:
            from pyscf.pbc import tools
            self.madelung = tools.madelung(mf.cell, mf.kpt)
            self.ewald_correct = True
            logger.info("----------- Using Ewald correction. The results may not be reliable.")
        else:
            self.ewald_correct = False

        t_localized = time.time()

        if local_orb_path == 'lowdin':
            logger.info("----------- Using meta_lowdin")
            self.AOLO = cupy.asarray(
                pyscf.lo.orth.orth_ao(
                    mf, method='meta_lowdin'))
        elif local_orb_path is not None:
            logger.info("----------- Using load AOLO from file")
            self.AOLO = cupy.asarray(numpy.load(local_orb_path))
        else:
            logger.info("----------- Using IAO+PAO localizer")
            self.AOLO = fix_orbital_sign(iao_pao_localization(mol, mf))

        logger.info(
            '----------- localization_function time cost is %s' %
            (time.time() - t_localized))

        self.low_scf_energy = mf.e_tot
        mf_mo_coeff = cupy.asarray(mf.mo_coeff)
        low_scf_dm = reduce(
            cupy.dot,
            (
                mf_mo_coeff,
                cupy.diag(cupy.asarray(mf.mo_occ)),
                mf_mo_coeff.T
            )
        )

        self.onerdm_low_ao = low_scf_dm.get()
        if self.kpts is None:
            if getattr(mol, 'pbc_intor', None) is None:
                if jk_file is None:
                    blksize = int((lib.gpu_avail_bytes() / (8 * 2)) ** (1 / 4))
                    vhfopt = scf.hf._VHFOpt(mol, 'int2e')
                    vhfopt.build(group_size=blksize)
                    t_get_JK = time.time()
                    j, k = scf.hf.get_jk(mol, low_scf_dm.get(), vhfopt=vhfopt)
                    logger.info(
                        '----------- get_jk time cost is %s' %
                        (time.time() - t_get_JK))
                    del vhfopt
                    lib.free_all_blocks()
                    low_scf_twoint = cupy.asarray(j - 0.5 * k)
                else:
                    logger.info('----------- load JK from %s' % jk_file)
                    low_scf_twoint = cupy.asarray(numpy.load(jk_file))

                if oei is None:
                    self.low_scf_fock = cupy.asarray(
                        mf.get_hcore(mol) + low_scf_twoint.get())
                else:
                    assert isinstance(oei, str), f'The oei type is not right, expect str but {type(oei)} provided'
                    self.low_scf_fock = cupy.asarray(
                        numpy.load(oei) + low_scf_twoint.get())
            else:
                logger.info('----------- load JK from %s' % jk_file)
                low_scf_twoint = cupy.asarray(numpy.load(jk_file))
                assert isinstance(oei, str), f'The oei type is not right, expect str but {type(oei)} provided'
                self.low_scf_fock = cupy.asarray(
                    numpy.load(oei) + low_scf_twoint.get())

            self.ao_ovlp = cupy.asarray(mf.get_ovlp(mol))
        else:
            from pyscf.pbc.tools import k2gamma as pbc_k2gamma

            nao_sc = int(self.mol_full.nao_nr())
            low_scf_dm_k = numpy.asarray(primitive_mf.make_rdm1())
            self.onerdm_low_ao = pbc_k2gamma.to_supercell_ao_integrals(
                primitive_mol, self.kpts, low_scf_dm_k, kmesh=self.kmesh, force_real=True)

            logger.info('----------- load JK from %s and transform it to gamma supercell' % jk_file)
            low_scf_twoint = cupy.asarray(
                _load_or_transform_pbc_ao_matrix(
                    jk_file, primitive_mol, self.kpts, self.kmesh, nao_sc, 'jk_file'
                )
            )

            self.low_scf_fock = cupy.asarray(
                _load_or_transform_pbc_ao_matrix(
                    oei, primitive_mol, self.kpts, self.kmesh, nao_sc, 'oei'
                ) + low_scf_twoint.get()
            )

            s_k = numpy.asarray(primitive_mf.get_ovlp(primitive_mol))
            self.ao_ovlp = cupy.asarray(
                pbc_k2gamma.to_supercell_ao_integrals(
                    primitive_mol, self.kpts, s_k, kmesh=self.kmesh, force_real=True
                )
            )

        # Make sure the mo energy does not include the ewald correction
        self.mo_energy = mf._eigh(self.low_scf_fock.get(), self.ao_ovlp.get())[0]

        if not numpy.isclose(reduce(numpy.dot, (self.AOLO.T,
                                                self.ao_ovlp, self.AOLO)).sum(), mol.nao):
            logger.info(
                f'+++ localized orbitals may not orthogonal!')

        self.LOMO = reduce(
            cupy.dot, (self.AOLO.T, self.ao_ovlp, mf_mo_coeff))
        self.LOMO = cupy.asarray(fix_orbital_sign(self.LOMO.get()))
        self.onerdm_low = reduce(
            cupy.dot,
            (self.LOMO,
             cupy.diag(
                 cupy.asarray(
                     mf.mo_occ)),
                self.LOMO.T)).get()

        self.core_constant_energy = mf.mol.energy_nuc()
        self.oei_LO = reduce(
            cupy.dot,
            (self.AOLO.T,
             self.low_scf_fock
             - low_scf_twoint,
             self.AOLO)).get()
        self.fock_LO = reduce(
            cupy.dot,
            (self.AOLO.T,
             self.low_scf_fock,
             self.AOLO)).get()
        self.low_scf_fock = self.low_scf_fock.get()
        self.MOLO = reduce(cupy.dot, (mf_mo_coeff.T, self.ao_ovlp, self.AOLO))
        self.ao_ovlp = self.ao_ovlp.get()
        self.LOMO = self.LOMO.get()
        self.AOMO = mf_mo_coeff.get()
        del mf_mo_coeff

        self.num_occ = int(mol.nelectron / 2)

        self.MOLO = self.MOLO.get()

        del mf
        if hasattr(self.auxmol, 'stdout'):
            try:
                del self.auxmol.stdout
            except:
                pass
