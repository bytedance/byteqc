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
from gpu4pyscf.dft import rks
from pyscf import gto
import numpy
import os


def rotate_vector(v, k, theta):
    k = k / numpy.linalg.norm(k)
    v_rot = v * numpy.cos(theta) \
        + numpy.cross(k, v) * numpy.sin(theta) \
        + k * numpy.dot(k, v) * (1 - numpy.cos(theta))
    v_rot = numpy.where(numpy.abs(v_rot) < 1e-14, 0.0, v_rot)
    return v_rot


def water_on_PAH_OBC(basis_set, mol_type, n, ad_type='2-leg', Z_t=0):
    # Rotate around x axis
    V = numpy.asarray([1, 0, 0])

    water = numpy.array([
        [0.00000000, 0.00000000, 3.15500000],
        [0.00000000, 0.75668992, 2.56910806],
        [0.00000000, -0.75668992, 2.56910806],
    ])
    Z = numpy.asarray([0., 0., float(Z_t)])
    mol_shift = water[0] + numpy.asarray([0, 0, 0.175])
    water += numpy.asarray([0, 0, 0.35])
    water -= mol_shift

    if ad_type == '2-leg':
        angle = 0
    elif ad_type == '0-leg':
        angle = 180 / 180 * numpy.pi
    else:
        assert isinstance(
            ad_type, int) or isinstance(
            ad_type, float), 'The degree of the water rotation angle must be a number or a str 0-leg or 2-leg.'
        angle = ad_type

    water_rotate = rotate_vector(water, V, angle) + Z + mol_shift

    geom_path = Path(__file__).resolve().parent.parent
    mol_ref = gto.M()
    mol_ref.atom = os.path.join(geom_path, f'Geometry/PAH{n}_0-leg.xyz')

    mol_ref.basis = basis_set
    mol_ref.verbose = 4
    mol_ref.build()

    mol_return = gto.M()
    if mol_type == 2:
        mol_return.atom \
            = [(mol_ref.atom_pure_symbol(atom_ind),
                mol_ref.atom_coord(atom_ind, 'angstrom').tolist()) if atom_ind > 2
                else (mol_ref.atom_pure_symbol(atom_ind), water_rotate[atom_ind].tolist())
                for atom_ind in range(mol_ref.natm)]
    elif mol_type == 1:
        mol_return.atom \
            = [(mol_ref.atom_pure_symbol(atom_ind),
                mol_ref.atom_coord(atom_ind, 'angstrom').tolist()) if atom_ind > 2
                else ('ghost-' + mol_ref.atom_pure_symbol(atom_ind), water_rotate[atom_ind].tolist())
                for atom_ind in range(mol_ref.natm)]
    elif mol_type == 0:
        mol_return.atom \
            = [('ghost-' + mol_ref.atom_pure_symbol(atom_ind),
                mol_ref.atom_coord(atom_ind, 'angstrom').tolist()) if atom_ind > 2
                else (mol_ref.atom_pure_symbol(atom_ind), water_rotate[atom_ind].tolist())
                for atom_ind in range(mol_ref.natm)]
    mol_return.unit = 'angstrom'
    mol_return.basis = basis_set
    mol_return.build()
    if 'ccecp' in basis_set:
        mol_return.ecp = 'ccecp'
        mol_return.build()
    mol_return.verbose = 4
    return mol_return


def water_on_PAH_OBC_opt(basis_set, mol_type, n, ad_type='2-leg'):
    current = Path(__file__).resolve().parent

    mol_ref = gto.M()
    mol_ref.atom = os.path.join(current, f'{xc}/{ad_type}/geom_opt.xyz')

    mol_ref.basis = basis_set
    mol_ref.verbose = 4
    mol_ref.build()

    mol_return = gto.M()
    if mol_type == 2:
        mol_return.atom = mol_ref.atom
    elif mol_type == 1:
        mol_return.atom \
            = [(mol_ref.atom_pure_symbol(atom_ind),
                mol_ref.atom_coord(atom_ind, 'angstrom').tolist()) if atom_ind > 2
                else ('ghost-' + mol_ref.atom_pure_symbol(atom_ind), mol_ref.atom_coord(atom_ind, 'angstrom').tolist())
                for atom_ind in range(mol_ref.natm)]
    elif mol_type == 0:
        mol_return.atom \
            = [('ghost-' + mol_ref.atom_pure_symbol(atom_ind),
                mol_ref.atom_coord(atom_ind, 'angstrom').tolist()) if atom_ind > 2
                else (mol_ref.atom_pure_symbol(atom_ind), mol_ref.atom_coord(atom_ind, 'angstrom').tolist())
                for atom_ind in range(mol_ref.natm)]
    mol_return.unit = 'angstrom'
    mol_return.basis = basis_set
    mol_return.build()
    if 'ccecp' in basis_set:
        mol_return.ecp = 'ccecp'
        mol_return.build()
    mol_return.verbose = 4
    return mol_return


if __name__ == '__main__':
    # Set the GTO basis
    basis = 'cc-pVQZ'

    boundary_condition = 'OBC'

    max_scf_cycles = 200
    screen_tol = 1e-14
    scf_tol = 1e-10

    n = 4

    xc_list = [
        'PBE',
        'PBE0',
        'B3LYP',
        'BLYP',
        'REVPBE',
        'REVPBE0',
        'WB97M-V',
        'B97M-V',
    ]
    for mol_type in [0, 1, 2]:
        for ad_type in ['0-leg', '2-leg']:
            for xc in xc_list:

                if xc not in ['WB97M-V', 'B97M-V']:
                    D3 = True
                else:
                    D3 = False

                mol = water_on_PAH_OBC(basis, mol_type, n, ad_type=ad_type)
                current = Path(__file__).resolve().parent

                if mol_type == 2:
                    logdir = os.path.join(current, f'{xc}/{ad_type}/Full')
                elif mol_type == 1:
                    logdir = os.path.join(
                        current, f'{xc}/{ad_type}/ghost-water')
                elif mol_type == 0:
                    logdir = os.path.join(
                        current, f'{xc}/{ad_type}/ghost-graphene')
                else:
                    assert False

                if not os.path.exists(logdir):
                    os.makedirs(logdir)

                logfile = os.path.join(logdir, 'energy_original.log')
                mol.output = logfile
                mol.build()
                mf = rks.RKS(mol, xc=xc).density_fit(auxbasis='weigend+etb')
                mf.conv_tol = scf_tol
                mf.max_cycle = max_scf_cycles
                mf.screen_tol = screen_tol
                mf.diis_space = 12
                mf.kernel()

                mol = water_on_PAH_OBC_opt(basis, mol_type, n, ad_type=ad_type)
                logfile = os.path.join(logdir, 'energy_optimized.log')
                mol.output = logfile
                mol.build()
                mf = rks.RKS(mol, xc=xc).density_fit(auxbasis='weigend+etb')
                mf.conv_tol = scf_tol
                mf.max_cycle = max_scf_cycles
                mf.screen_tol = screen_tol
                mf.diis_space = 12
                mf.kernel()
