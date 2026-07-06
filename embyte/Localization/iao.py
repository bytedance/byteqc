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
from pyscf import gto
from pyscf.data.elements import is_ghost_atom
from pyscf.lo.iao import reference_mol
from pyscf.pbc import gto as pbcgto
import numpy
import cupy
cupy.cuda.set_pinned_memory_allocator(None)


def get_vec_lowdin(c, s=1, tol=1e-14):

    e, v = cupy.linalg.eigh(reduce(cupy.dot, (c.T, s, c)))
    ind = e > tol
    ct = cupy.dot(v[:, ind] / (e[ind] ** 0.5), v[:, ind].T)
    lo_lowdin = cupy.dot(c, ct)

    return lo_lowdin


def reference_mol_get_mask(mol):
    mol_tmp = mol.copy()
    atoms = [atom for atom in gto.format_atom(mol_tmp.atom, unit=1)]
    mask = [i for i, atom in enumerate(atoms) if not is_ghost_atom(atom[0])]

    return mask


def iao_pao_localization(mol, mf, minao='minao', tol=1e-12):

    nocc = numpy.where(cupy.asarray(mf.mo_occ) > 0)[0].size
    aomo_coeff_occ = cupy.asarray(mf.mo_coeff)[:, : nocc]

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

    coeff_inter = cupy.linalg.solve(ref_ao_ovlp_CD.T,
                                    cupy.linalg.solve(ref_ao_ovlp_CD,
                                                      cupy.dot(cross_ovlp.T, aomo_coeff_occ)))
    coeff_inter = cupy.linalg.solve(ao_ovlp_CD.T,
                                    cupy.linalg.solve(ao_ovlp_CD,
                                                      cupy.dot(cross_ovlp, coeff_inter)))

    e_tmp, v_tmp = cupy.linalg.eigh(reduce(cupy.dot,
                                           (coeff_inter.T, ao_ovlp, coeff_inter)))
    ind_s = e_tmp > tol
    ct = v_tmp[:, ind_s] / (e_tmp[ind_s] ** 0.5)
    coeff_inter = cupy.dot(coeff_inter, ct)

    Cocc_ovlp = reduce(cupy.dot, (aomo_coeff_occ, aomo_coeff_occ.T, ao_ovlp))
    inter_ovlp = reduce(cupy.dot, (coeff_inter, coeff_inter.T, ao_ovlp))

    project_t = cupy.linalg.solve(ao_ovlp_CD.T,
                                  cupy.linalg.solve(ao_ovlp_CD, cross_ovlp))

    coeff_iao = (project_t
                 + reduce(cupy.dot, (Cocc_ovlp, inter_ovlp, project_t)) * 2
                 - cupy.dot(Cocc_ovlp, project_t)
                 - cupy.dot(inter_ovlp, project_t))

    coeff_iao = get_vec_lowdin(coeff_iao, ao_ovlp)

    if coeff_iao.shape[0] == coeff_iao.shape[1]:
        # The mol has the same basis size with the IAO reference mol.
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

    iao_ovlp = reduce(cupy.dot, (coeff_iao, coeff_iao.T, ao_ovlp))

    coeff_pao = (cupy.eye(mol.nao) - iao_ovlp)[:, pao_ind]
    coeff_pao = get_vec_lowdin(coeff_pao, ao_ovlp)

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

    return coeff_iao_pao
