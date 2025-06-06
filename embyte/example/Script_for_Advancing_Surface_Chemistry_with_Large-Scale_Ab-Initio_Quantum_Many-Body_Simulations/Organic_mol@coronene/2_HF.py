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

from pathlib import Path
from pyscf import gto
from byteqc.cuobc import scf
import numpy
import os


def small_organic_mol_on_coronene(adsorbate, basis_set, mol_type, n=2):
    current = Path(__file__).resolve().parent
    mol_path = os.path.join(current, f'{adsorbate}/geom_opt.xyz')

    mol_ref = gto.M()
    mol_ref.atom = mol_path

    mol_ref.basis = basis_set
    mol_ref.verbose = 6
    mol_ref.build()

    mol = mol_ref.copy()
    natm_ad = mol_ref.natm - 6 * n ** 2 - 6 * n

    if mol_type == 1:
        mol.atom = [(mol_ref.atom_pure_symbol(atom_ind),
                     mol_ref.atom_coord(atom_ind, 'angstrom').tolist()) if atom_ind > natm_ad - 1
                    else ('ghost-' + mol_ref.atom_pure_symbol(atom_ind),
                          mol_ref.atom_coord(atom_ind, 'angstrom').tolist())
                    for atom_ind in range(mol_ref.natm)]
    elif mol_type == 0:
        mol.atom = [('ghost-' + mol_ref.atom_pure_symbol(atom_ind),
                     mol_ref.atom_coord(atom_ind, 'angstrom').tolist()) if atom_ind > natm_ad - 1
                    else (mol_ref.atom_pure_symbol(atom_ind),
                          mol_ref.atom_coord(atom_ind, 'angstrom').tolist())
                    for atom_ind in range(mol_ref.natm)]
    elif mol_type == 2:
        pass
    else:
        assert False

    if 'ccecp' in basis_set:
        mol.ecp = 'ccecp'

    mol.build()

    return mol


if __name__ == '__main__':
    basis = 'cc-pVDZ'
    # basis = 'cc-pVTZ'

    mol_type = 2
    # mol_type = 1
    # mol_type = 0

    adsorbates_list = [
        'acetone',
        'acetonitrile',
        'dichloromethane',
        'ethanol',
        'ethylacetate',
        'toluene']
    adsorbate = adsorbates_list[0]

    mol = small_organic_mol_on_coronene(adsorbate, basis, mol_type)

    current = Path(__file__).resolve().parent
    if mol_type == 2:
        logdir = os.path.join(current, f'{adsorbate}/{basis}/Full')
    elif mol_type == 1:
        logdir = os.path.join(current, f'{adsorbate}/{basis}/ghost-adsorbate')
    elif mol_type == 0:
        logdir = os.path.join(current, f'{adsorbate}/{basis}/ghost-coronene')
    else:
        assert False

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    chkfile = os.path.join(logdir, 'HF_chkfile.chk')
    JK_file = os.path.join(logdir, 'JK_file.npy')
    logfile = os.path.join(logdir, 'HF.log')

    mol.output = logfile
    mol.build()
    mf = scf.RHF(mol)
    mf.chkfile = chkfile
    mf.diis_space = 12

    mf.kernel()

    veff = mf.get_veff()
    numpy.save(JK_file, veff)
