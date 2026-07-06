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
import itertools
from pyscf import gto
from copy import copy


def from_atom_to_orb_iao(mol, atom_list_frag, minao='minao'):
    '''
    Exchange the atom fragment list to the IAO fragemnt list.
    '''
    refer_mol = gto.Mole()
    refer_mol.atom = mol.atom
    refer_mol.basis = minao
    refer_mol.build()

    mol_atom_orb_num = numpy.array(mol.ao_labels(fmt=False))
    mol_atom_orb_num = numpy.array(mol_atom_orb_num[:, 0], dtype=int)
    mol_atom_orb_num = numpy.bincount(mol_atom_orb_num)
    refer_mol_atom_orb_num = numpy.array(refer_mol.ao_labels(fmt=False))
    refer_mol_atom_orb_num = numpy.array(
        refer_mol_atom_orb_num[:, 0], dtype=int)
    refer_mol_atom_orb_num = numpy.bincount(refer_mol_atom_orb_num)
    atom_orb_list = []

    for i_ind in range(mol.natm):
        i_now = int(numpy.sum(mol_atom_orb_num[:i_ind]))
        tmp = range(i_now, int(i_now + refer_mol_atom_orb_num[i_ind]))
        tmp = list(tmp)
        atom_orb_list.append(tmp)
    frag_orb_list = []
    for atom_frag in atom_list_frag:
        tmp = []
        for atom in atom_frag:
            tmp += atom_orb_list[atom]
        frag_orb_list.append(tmp)

    return frag_orb_list


class Fragment_group:
    '''
    Group all the fragment and give them a tag.
    Basiscally, the fragments which is equivalent to each other will be grouped together.
    '''

    def __init__(
        self,
        mol,
        Fragment_list,
    ):
        self.aoslice_by_atom = mol.aoslice_by_atom()[:, 2:4]
        self.aoslice_by_atom = [list(range(i[0], i[1]))
                                for i in self.aoslice_by_atom]
        self.fragments = Fragment_list
        self.atom_coord = mol.atom_coords(unit='A')
        self.orb_list = []
        self.equivalent_list = []
        self.symm_op = []
        self.pbc = bool(getattr(mol, 'pbc_intor', None))
        if self.pbc:
            self.lattice_vec = mol.a.copy()
            from cupy.linalg import inv
            self.atom_frac_coords = inv(
                cupy.asarray(
                    self.lattice_vec).T).dot(
                cupy.asarray(
                    self.atom_coord).T).T

        for frag in self.fragments:
            self.orb_list.append(frag['fragment_orb_list'])

            for atom_ind, atom_orb in enumerate(self.aoslice_by_atom):
                if numpy.intersect1d(
                        atom_orb, frag['fragment_orb_list']).size != 0:
                    frag['fragment_atom_list'].append(atom_ind)

            self.equivalent_list.append(frag['equivalent_level'])
            if frag['equivalent_operator'] != ['main']:
                if frag['equivalent_operator'] not in self.symm_op:
                    self.symm_op.append(frag['equivalent_operator'])

        if all(value is None for value in self.equivalent_list):
            self.equivalent_list = [i for i in range(len(self.orb_list))]

        self.equi_part = []

        for ind_i, level in enumerate(self.equivalent_list):
            while True:
                try:
                    self.equi_part[level].append(ind_i)
                except IndexError:
                    self.equi_part.append([])
                else:
                    break

    def check_atom_on_boundary(self, equi_group, atom_coord_equi_group):
        for eq_ind in range(len(equi_group)):
            atom_list_tmp = self.fragments[equi_group[eq_ind]
                                           ]['fragment_atom_list']
            atom_factor_coord = self.atom_frac_coords[atom_list_tmp]
            if_boundary = numpy.isclose(0.5, abs(atom_factor_coord))
            if if_boundary.any():
                all_possible = numpy.where(if_boundary is True)
                all_possible = list(
                    zip(all_possible[0].tolist(), all_possible[1].tolist()))
                all_combinations = [
                    comb for r in range(
                        1,
                        len(all_possible) + 1) for comb in itertools.combinations(
                        all_possible, r)]
                for plan_com in all_combinations:
                    equi_group.append(equi_group[eq_ind])
                    atom_coord_equi_group_tmp = atom_coord_equi_group[eq_ind].copy(
                    )
                    for atm_ind, axis_ind in plan_com:
                        atom_coord_equi_group_tmp[atm_ind] -= self.lattice_vec[axis_ind] * int(
                            atom_coord_equi_group_tmp[atm_ind, axis_ind] / abs(atom_coord_equi_group_tmp[atm_ind, axis_ind]))

                    atom_coord_equi_group.append(atom_coord_equi_group_tmp)

        return equi_group, atom_coord_equi_group

    def build(self):
        self.pair_relation_ship = []
        cross_check = cupy.ndarray(
            (len(
                self.fragments), len(
                self.fragments)), dtype=bool)
        cross_check[:] = False
        for equi_group_ind_x, equi_group_x in enumerate(self.equi_part):
            equi_group_x = copy(equi_group_x)
            frag_ind_x, frag_x = equi_group_x[0], self.fragments[equi_group_x[0]]

            atom_coord_x_equi_group = []
            for frag_ind_x_tmp in equi_group_x:
                atom_list_x_tmp = self.fragments[frag_ind_x_tmp]['fragment_atom_list']
                atom_coord_x_equi_group.append(
                    self.atom_coord[atom_list_x_tmp])

            for equi_group_y in self.equi_part[equi_group_ind_x:]:
                equi_group_y = copy(equi_group_y)
                atom_coord_y_equi_group = []
                for frag_ind_y_tmp in equi_group_y:
                    atom_list_y_tmp = self.fragments[frag_ind_y_tmp]['fragment_atom_list']
                    atom_coord_y_equi_group.append(
                        self.atom_coord[atom_list_y_tmp])

                for frag_ind_y in equi_group_y:
                    pair_relation_ship_x = []
                    frag_y = self.fragments[frag_ind_y]
                    if not cross_check[frag_ind_x, frag_ind_y]:
                        pair_relation_ship_x.append([frag_ind_x, frag_ind_y])
                        cross_check[frag_ind_x, frag_ind_y] = True
                        if frag_ind_x != frag_ind_y:
                            pair_relation_ship_x.append(
                                [frag_ind_y, frag_ind_x])
                            cross_check[frag_ind_y, frag_ind_x] = True
                    else:
                        continue

                    for ops in self.symm_op:
                        frag_x_eq_atom_coord = self.atom_coord[frag_x['fragment_atom_list']]
                        frag_y_eq_atom_coord = self.atom_coord[frag_y['fragment_atom_list']]
                        for op in ops:
                            if 'Reflection' not in op:
                                raise KeyboardInterrupt('Not support')
                            else:
                                frag_x_eq_atom_coord = self.Reflection(
                                    op[1], frag_x_eq_atom_coord)
                                frag_y_eq_atom_coord = self.Reflection(
                                    op[1], frag_y_eq_atom_coord)

                        mask_x = numpy.isclose(
                            atom_coord_x_equi_group,
                            frag_x_eq_atom_coord[0]).all(
                            axis=2).any(
                            axis=1)
                        frag_ind_x_eq = numpy.asarray(equi_group_x)[mask_x][0]

                        mask_y = numpy.isclose(
                            atom_coord_y_equi_group,
                            frag_y_eq_atom_coord[0]).all(
                            axis=2).any(
                            axis=1)
                        frag_ind_y_eq = numpy.asarray(equi_group_y)[mask_y][0]

                        if not cross_check[frag_ind_x_eq, frag_ind_y_eq]:
                            pair_relation_ship_x.append(
                                [frag_ind_x_eq, frag_ind_y_eq])
                            cross_check[frag_ind_x_eq, frag_ind_y_eq] = True
                            if frag_ind_x_eq != frag_ind_y_eq:
                                pair_relation_ship_x.append(
                                    [frag_ind_y_eq, frag_ind_x_eq])
                                cross_check[frag_ind_y_eq,
                                            frag_ind_x_eq] = True
                        else:
                            continue
                    if pair_relation_ship_x != []:
                        self.pair_relation_ship.append(pair_relation_ship_x)
        assert cross_check.get().all()

    def gourp_pair(self, group_size=200):
        self.group_pair_relation_ship = []
        pairing_ind_start_point = 0
        while True:

            if pairing_ind_start_point >= len(self.pair_relation_ship):
                break
            pair_slice = slice(
                pairing_ind_start_point,
                pairing_ind_start_point + group_size)
            group_pair_tmp = self.pair_relation_ship[pair_slice]
            if group_pair_tmp[0][0][0] == group_pair_tmp[-1][0][0]:
                self.group_pair_relation_ship.append(group_pair_tmp)
                pairing_ind_start_point += group_size
            else:
                match_frag = group_pair_tmp[0][0][0]
                for pair_ind, pair in enumerate(group_pair_tmp):
                    if pair[0][0] != match_frag:
                        self.group_pair_relation_ship.append(
                            group_pair_tmp[: pair_ind])
                        pairing_ind_start_point += pair_ind
                        break

        tmp_pair_list = []
        for pair_tmp in self.group_pair_relation_ship:
            tmp_pair_list += pair_tmp

        assert tmp_pair_list == self.pair_relation_ship

    def Reflection(self, plan, atom_coords):

        def Reflection_op(crood, axis, center=numpy.asarray([0.0, 0.0, 0.0])):

            axis = numpy.asarray(axis) / numpy.linalg.norm(axis)

            n = numpy.asarray(axis)
            p0 = numpy.asarray(center)
            q = numpy.asarray(crood)

            d = numpy.dot(n, q - p0) / numpy.linalg.norm(n)

            r = q - 2 * d * n / numpy.linalg.norm(n)

            return r

        for i_ind in range(len(atom_coords)):
            atom_coords[i_ind] = Reflection_op(atom_coords[i_ind], plan)

        return atom_coords


class Fragment:
    '''
    State a fragment with the the information of the
    IAOs/equivalent_level/equivalent_operator for this fragemnt.
    '''

    def __init__(
        self,
        fragment_orb_list,
        equivalent_level,
        equivalent_operator,
    ):

        if type(fragment_orb_list) is list:
            self.fragment_orb_list = fragment_orb_list
        else:
            raise ValueError

        if type(equivalent_level) in [int, numpy.int64]:
            self.equivalent_level = equivalent_level
        else:
            raise ValueError

        if type(equivalent_operator) is list:
            self.equivalent_operator = equivalent_operator
        else:
            raise ValueError

        self.fragment_atom_list = []

        pass

    def to_dict(self):
        return vars(self)
