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

from pyscf import gto
from pyscf.data.elements import is_ghost_atom
from pyscf.lo.iao import reference_mol
from pyscf.pbc import gto as pbcgto
from byteqc import lib
import numpy
import cupy
cupy.cuda.set_pinned_memory_allocator(None)


def _free_gpu_memory():
    cupy.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()


def get_vec_lowdin(c, s=1, tol=1e-14, orth_method='lowdin'):

    if numpy.isscalar(s):
        sc = c if s == 1 else c * s
    else:
        sc = cupy.dot(s, c)
    metric = cupy.dot(c.T, sc)
    if sc is not c:
        del sc
    metric = (metric + metric.T) * 0.5

    if orth_method == 'cholesky':
        chol = cupy.linalg.cholesky(metric)
        del metric
        c_t = cupy.ascontiguousarray(c.T)
        lo_t = lib.solve_triangular(chol, c_t, lower=True, overwrite_b=True)
        del chol, c_t
        lo_lowdin = lo_t.T
        _free_gpu_memory()
        return lo_lowdin

    if orth_method != 'lowdin':
        raise ValueError(f'Unknown orth_method {orth_method}')

    e, v = cupy.linalg.eigh(metric)
    del metric
    ind = e > tol
    ct = cupy.dot(v[:, ind] / (e[ind] ** 0.5), v[:, ind].T)
    del e, v, ind
    lo_lowdin = cupy.dot(c, ct)
    del ct
    _free_gpu_memory()

    return lo_lowdin


def reference_mol_get_mask(mol):
    mol_tmp = mol.copy()
    atoms = [atom for atom in gto.format_atom(mol_tmp.atom, unit=1)]
    mask = [i for i, atom in enumerate(atoms) if not is_ghost_atom(atom[0])]

    return mask


def iao_pao_localization(mol, mf, minao='minao', tol=1e-12, just_iao=False):

    mf_mo_occ = numpy.asarray(mf.mo_occ)
    nocc = int(numpy.count_nonzero(mf_mo_occ > 0))
    if isinstance(mf.mo_coeff, cupy.ndarray):
        aomo_coeff_occ = cupy.asarray(mf.mo_coeff[:, : nocc])
    else:
        aomo_coeff_occ = cupy.asarray(numpy.asarray(mf.mo_coeff)[:, : nocc])
    del mf_mo_occ

    ref_mol = reference_mol(mol, minao)

    if hasattr(mol, 'pbc_intor'):
        ao_ovlp = cupy.asarray(mol.pbc_intor('int1e_ovlp',
                                             hermi=1,
                                             kpts=None))
        ref_ao_ovlp = cupy.asarray(ref_mol.pbc_intor('int1e_ovlp',
                                                     hermi=1,
                                                     kpts=None))
        cross_ovlp = cupy.asarray(
            pbcgto.cell.intor_cross(
                'int1e_ovlp',
                mol,
                ref_mol,
                kpts=None))
    else:

        ao_ovlp = cupy.asarray(mol.intor_symmetric('int1e_ovlp'))
        ref_ao_ovlp = cupy.asarray(ref_mol.intor_symmetric('int1e_ovlp'))
        cross_ovlp = cupy.asarray(
            gto.mole.intor_cross(
                'int1e_ovlp',
                mol,
                ref_mol),
            dtype=cupy.float64)

    ao_ovlp_CD = cupy.linalg.cholesky(ao_ovlp)
    ref_ao_ovlp_CD = cupy.linalg.cholesky(ref_ao_ovlp)
    del ref_ao_ovlp
    _free_gpu_memory()

    tmp = cupy.dot(cross_ovlp.T, aomo_coeff_occ)
    tmp_solve = cupy.linalg.solve(ref_ao_ovlp_CD, tmp)
    del tmp
    coeff_inter = cupy.linalg.solve(ref_ao_ovlp_CD.T, tmp_solve)
    del tmp_solve, ref_ao_ovlp_CD
    _free_gpu_memory()

    tmp = cupy.dot(cross_ovlp, coeff_inter)
    del coeff_inter
    tmp_solve = cupy.linalg.solve(ao_ovlp_CD, tmp)
    del tmp
    coeff_inter = cupy.linalg.solve(ao_ovlp_CD.T, tmp_solve)
    del tmp_solve
    _free_gpu_memory()

    tmp = cupy.dot(ao_ovlp, coeff_inter)
    metric = cupy.dot(coeff_inter.T, tmp)
    del tmp
    e_tmp, v_tmp = cupy.linalg.eigh(metric)
    del metric
    ind_s = e_tmp > tol
    ct = v_tmp[:, ind_s] / (e_tmp[ind_s] ** 0.5)
    del e_tmp, v_tmp, ind_s
    coeff_inter_raw = coeff_inter
    coeff_inter = cupy.dot(coeff_inter_raw, ct)
    del coeff_inter_raw, ct
    _free_gpu_memory()

    occ_t_s = cupy.dot(aomo_coeff_occ.T, ao_ovlp)
    Cocc_ovlp = cupy.dot(aomo_coeff_occ, occ_t_s)
    del occ_t_s, aomo_coeff_occ
    _free_gpu_memory()

    inter_t_s = cupy.dot(coeff_inter.T, ao_ovlp)
    inter_ovlp = cupy.dot(coeff_inter, inter_t_s)
    del inter_t_s, coeff_inter
    _free_gpu_memory()

    tmp_solve = cupy.linalg.solve(ao_ovlp_CD, cross_ovlp)
    del cross_ovlp
    project_t = cupy.linalg.solve(ao_ovlp_CD.T, tmp_solve)
    del tmp_solve, ao_ovlp_CD
    _free_gpu_memory()

    inter_project = cupy.dot(inter_ovlp, project_t)
    del inter_ovlp
    cocc_project = cupy.dot(Cocc_ovlp, project_t)
    cocc_inter_project = cupy.dot(Cocc_ovlp, inter_project)
    del Cocc_ovlp
    _free_gpu_memory()

    coeff_iao = project_t
    del project_t
    coeff_iao -= cocc_project
    coeff_iao -= inter_project
    coeff_iao += cocc_inter_project
    coeff_iao += cocc_inter_project
    del cocc_project, inter_project, cocc_inter_project

    coeff_iao_raw = coeff_iao
    coeff_iao = get_vec_lowdin(coeff_iao_raw, ao_ovlp)
    del coeff_iao_raw
    _free_gpu_memory()

    if coeff_iao.shape[0] == coeff_iao.shape[1] or just_iao:
        # The mol has the same basis size with the IAO reference mol.
        del ao_ovlp
        _free_gpu_memory()
        return coeff_iao

    mol_ao_labels = numpy.asarray(mol.ao_labels())
    assert mol.nao == len(mol_ao_labels)
    ref_mol_tmp = mol.copy()
    ref_mol_tmp.basis = minao
    ref_mol_tmp.build()
    iao_labels = numpy.asarray([ao_label for ao_label in
                                ref_mol_tmp.ao_labels() if 'GHOST' not in ao_label])
    pao_ind = numpy.where(numpy.isin(mol_ao_labels,
                                     iao_labels,
                                     invert=True))[0].tolist()
    niao = len(iao_labels)
    npao = len(pao_ind)
    assert niao + npao == mol.nao

    iao_t_s = cupy.dot(coeff_iao.T, ao_ovlp)
    iao_ovlp = cupy.dot(coeff_iao, iao_t_s)
    del iao_t_s
    _free_gpu_memory()

    coeff_pao = -cupy.ascontiguousarray(iao_ovlp[:, pao_ind])
    del iao_ovlp
    pao_ind_gpu = cupy.asarray(pao_ind)
    coeff_pao[pao_ind_gpu, cupy.arange(npao)] += 1.0
    del pao_ind_gpu
    coeff_pao_raw = coeff_pao
    _free_gpu_memory()
    coeff_pao = get_vec_lowdin(coeff_pao_raw, ao_ovlp)
    del coeff_pao_raw, ao_ovlp
    _free_gpu_memory()

    mol_atom_orb_num = numpy.array(mol.ao_labels(fmt=False))
    mol_atom_orb_num = numpy.array(mol_atom_orb_num[:, 0], dtype=int)
    mol_atom_orb_num = numpy.bincount(mol_atom_orb_num)

    ref_mol_atom_orb_num = numpy.array(ref_mol.ao_labels(fmt=False))
    ref_mol_atom_orb_num = numpy.array(ref_mol_atom_orb_num[:, 0], dtype=int)
    ref_mol_atom_orb_num = numpy.bincount(ref_mol_atom_orb_num)

    if ref_mol_atom_orb_num.shape != mol_atom_orb_num.shape:
        ref_mol_atom_orb_num_temp = numpy.asarray(
            [0] * mol_atom_orb_num.shape[0])
        ref_mol_atom_orb_num_temp[reference_mol_get_mask(
            mol)] = ref_mol_atom_orb_num
        ref_mol_atom_orb_num = ref_mol_atom_orb_num_temp
        pass

    vir_atom_orb_num = mol_atom_orb_num - ref_mol_atom_orb_num
    num_occ_cumsum = numpy.cumsum(ref_mol_atom_orb_num)
    num_vir_cumsum = numpy.cumsum(vir_atom_orb_num)

    coeff_iao_split = cupy.split(coeff_iao, num_occ_cumsum, axis=1)[:-1]
    coeff_pao_split = cupy.split(coeff_pao, num_vir_cumsum, axis=1)[:-1]

    c_iao_parts = [
        cupy.hstack(
            (occ, vir)) for occ, vir in zip(
            coeff_iao_split, coeff_pao_split)]

    coeff_iao_pao = cupy.hstack(c_iao_parts)
    del c_iao_parts, coeff_iao_split, coeff_pao_split, coeff_iao, coeff_pao
    _free_gpu_memory()

    return coeff_iao_pao
