# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# This file is part of ByteQC.
#
# ByteQC is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ByteQC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from pathlib import Path
from byteqc import cump2
import numpy
import os


def rotate_vector(v, k, theta):
    k = k / numpy.linalg.norm(k)
    v_rot = v * numpy.cos(theta) \
        + numpy.cross(k, v) * numpy.sin(theta) \
        + k * numpy.dot(k, v) * (1 - numpy.cos(theta))
    v_rot = numpy.where(numpy.abs(v_rot) < 1e-14, 0.0, v_rot)
    return v_rot


def water_on_graphene_PBC(k_mesh, ad_type, basis_set, Z=0, mol_type=2):
    V = numpy.asarray([1, 0, 0], dtype='float64')

    water = numpy.array([[0.00000000, 0.00000000, 3.15500000],
                         [0.00000000, 0.75668992, 2.56910806],
                         [0.00000000, -0.75668992, 2.56910806],
                         ])

    if ad_type == '2-leg':
        angle = 0
    elif ad_type == '0-leg':
        angle = 180 / 180 * numpy.pi
    else:
        assert isinstance(
            ad_type, int), 'The degree of the water rotation angle must be a number or a str 0-leg or 2-leg.'
        angle = ad_type

    mol_shift = water[0] + numpy.asarray([0, 0, 0.175])
    water += numpy.asarray([0, 0, 0.35])
    water -= mol_shift
    angle = angle * numpy.pi / 180
    water_rotate = rotate_vector(water, V, angle) + Z + mol_shift

    from pyscf.pbc import gto as pbcgto
    Lz = 20

    cell = pbcgto.Cell()
    CC_bond = 1.42
    length_cell = CC_bond / numpy.tan(numpy.pi / 6)
    cell.unit = 'A'
    a = numpy.asarray([length_cell * numpy.sin(numpy.pi / 6),
                      length_cell * numpy.cos(numpy.pi / 6), 0])
    b = numpy.asarray([-1 * length_cell * numpy.sin(numpy.pi / 6),
                      length_cell * numpy.cos(numpy.pi / 6), 0])
    c = numpy.asarray([0, 0, Lz])
    C1 = a * 1 / 3 + b * 1 / 3
    C2 = a * 2 / 3 + b * 2 / 3
    atom_list = water_rotate.tolist()
    for a_ind in range(k_mesh[0]):
        for b_ind in range(k_mesh[1]):
            atom_list.append((C1 + a_ind * a + b_ind * b).tolist())
            atom_list.append((C2 + a_ind * a + b_ind * b).tolist())
    a *= k_mesh[0]
    b *= k_mesh[1]
    atom_list = numpy.asarray(atom_list)
    atom_list[3:] -= numpy.asarray([0, a[1], 0])

    symbols = {
        0: {'O': 'O', 'H': 'H', 'C': 'ghost-C'},
        1: {'O': 'ghost-O', 'H': 'ghost-H', 'C': 'C'},
        2: {'O': 'O', 'H': 'H', 'C': 'C'},
    }

    atoms = []

    for atom_ind, axis in enumerate(atom_list):
        if atom_ind == 0:
            atoms.append([symbols[mol_type]['O'], axis])
        elif atom_ind > 0 and atom_ind < 3:
            atoms.append([symbols[mol_type]['H'], axis])
        else:
            atoms.append([symbols[mol_type]['C'], axis])

    cell.atom = atoms
    cell.basis = basis_set
    cell.a = numpy.asarray([a, b, c])

    cell.verbose = 6
    cell.exp_to_discard = 0.1

    if 'ccecp' in basis_set:
        cell.ecp = 'ccecp'

    cell.build()

    return cell


if __name__ == '__main__':
    # Set the GTO basis
    basis = 'ccecp-cc-pVDZ'
    # basis = 'ccecp-cc-pVTZ'

    mol_type = 2
    # mol_type = 2 for water + graphene
    # mol_type = 1 for ghost-water + graphene
    # mol_type = 0 for water + ghost-graphene

    boundary_condition = 'PBC'

    # Set the graphene size
    k = 4
    k_mesh = [k, k, 1]

    # Set the adsorption configuration for water monomer
    # ad_type should be '2-leg', '0-leg', or specific rotation angle.
    ad_type = '2-leg'

    cell = water_on_graphene_PBC(k_mesh, ad_type, basis, mol_type=mol_type)

    current = Path(__file__).resolve().parent
    if mol_type == 2:
        logdir = os.path.join(
            current, f'{boundary_condition}/{basis}/{ad_type}_graphene{k}{k}1/Full')
    elif mol_type == 1:
        logdir = os.path.join(
            current, f'{boundary_condition}/{basis}/{ad_type}_graphene{k}{k}1/ghost-water')
    elif mol_type == 0:
        logdir = os.path.join(
            current,
            f'{boundary_condition}/{basis}/{ad_type}_graphene{k}{k}1/ghost-graphene')
    else:
        assert False

    chkfile = os.path.join(logdir, 'HF_chkfile.chk')
    JK_file = os.path.join(logdir, 'JK_file.npy')
    cderi_file = os.path.join(logdir, 'cderi.h5')
    assert os.path.exists(chkfile) and os.path.exists(JK_file)

    logfile = os.path.join(logdir, 'MP2.log')
    cell.output = logfile
    cell.build()
    from pyscf.pbc import scf as pbcscf
    mf = pbcscf.RHF(cell, exxdiv=None).density_fit(auxbasis='weigend+etb')
    mf.__dict__.update(pbcscf.chkfile.load(chkfile, 'scf'))
    e_corr = cump2.DFKMP2(
        cell,
        mf,
        auxbasis='weigend+etb',
        cderi_path=cderi_file,
        with_rdm1=False)
