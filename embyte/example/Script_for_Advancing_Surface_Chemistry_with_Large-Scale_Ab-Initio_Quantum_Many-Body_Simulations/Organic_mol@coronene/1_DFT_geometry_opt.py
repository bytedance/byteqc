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
from pyscf.geomopt.geometric_solver import optimize
from gpu4pyscf.dft import rks
from pyscf import gto
import os
# Please install gpu4pyscf before running this script.


def small_organic_mol_on_coronene(adsorbate, basis_set):
    geom_path = Path(__file__).resolve().parent
    mol = gto.M()
    mol.atom = os.path.join(geom_path, f'Geometry/coronene+{adsorbate}.xyz')

    mol.basis = basis_set
    mol.verbose = 4

    if 'ccecp' in basis_set:
        mol.ecp = 'ccecp'
    mol.build()
    mol.verbose = 4
    return mol


if __name__ == '__main__':
    # Set the GTO basis
    basis = 'cc-pVTZ'

    max_scf_cycles = 200
    screen_tol = 1e-14
    scf_tol = 1e-10

    xc1 = 'PBE'
    xc2 = 'WB97M-V'
    adsorbates_list = [
        'acetone',
        'acetonitrile',
        'dichloromethane',
        'ethanol',
        'ethylacetate',
        'toluene']

    for adsorbate in adsorbates_list:

        mol = small_organic_mol_on_coronene(adsorbate, basis)
        current = Path(__file__).resolve().parent
        logdir = os.path.join(current, f'{adsorbate}')

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        logfile = os.path.join(logdir, f'DFT_opt.log')

        mol.output = logfile
        mol.build()
        mf = rks.RKS(mol, xc=xc1).density_fit()
        mf.conv_tol = scf_tol
        mf.max_cycle = max_scf_cycles
        mf.screen_tol = screen_tol
        mf.diis_space = 12
        mf.disp = 'd3bj'

        gradients = []

        def callback(envs):
            gradients.append(envs['gradients'])

        mol = optimize(
            mf,
            maxsteps=max_scf_cycles * 10,
            callback=callback)

        mf = rks.RKS(mol, xc=xc2).density_fit()
        mf.conv_tol = scf_tol
        mf.max_cycle = max_scf_cycles
        mf.screen_tol = screen_tol
        mf.diis_space = 12

        mol_opt = optimize(
            mf,
            maxsteps=max_scf_cycles * 10,
            callback=callback)

        geom_file = os.path.join(logdir, 'geom_opt.xyz')
        with open(geom_file, 'w') as f:
            f.write(f'{mol.natm}\n')
            f.write(f'optimized by {xc2} by using {basis} in coronene\n')
            for atom_ind in range(mol_opt.natm):
                coord = mol_opt.atom_coord(atom_ind, 'angstrom').tolist()
                coord = [f'{c:12.8f}' for c in coord]
                f.write(f'{mol_opt.atom_pure_symbol(atom_ind)}{coord[0]}{coord[1]}{coord[2]}\n')
