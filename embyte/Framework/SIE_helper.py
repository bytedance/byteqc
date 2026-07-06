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
from byteqc.embyte.Tools.tool_lib import fix_orbital_sign
import numpy
import cupy
cupy.cuda.set_pinned_memory_allocator(None)


def Get_bath(mol, fb_size_list, frag_list, rdm1_low):
    '''
    Calculate bath based on given framgment orbitals.
    '''

    norb_tot = mol.nao_nr()
    all_ind = numpy.arange(norb_tot)
    env_ind = numpy.setdiff1d(all_ind, frag_list)

    rdm1_env = cupy.asarray(rdm1_low[env_ind][:, env_ind])

    occupation_env, coeff_env = cupy.linalg.eigh(rdm1_env)
    coeff_env = fix_orbital_sign(coeff_env)
    occupation_env = occupation_env.get(blocking=True)
    coeff_env = coeff_env.get(blocking=True)
    new_ind = numpy.maximum(-occupation_env, occupation_env - 2.0).argsort()

    norb_bath = numpy.sum(-numpy.maximum(-occupation_env,
                          occupation_env - 2.0)[new_ind] > 1e-8)
    assert norb_bath < fb_size_list[0] or numpy.isclose(norb_bath, fb_size_list[0]), \
        f'bath orbitals number : {norb_bath}, fragment orbitals number : {fb_size_list[0]}, where bath size > fragment size.'
    fb_size_list.append(int(norb_bath))
    occupation_env = occupation_env[new_ind]
    coeff_env = coeff_env[:, new_ind]

    nelectron_frag = round(numpy.diag(
        rdm1_low[frag_list][:, frag_list]).sum().item())
    nelectron_bath = round(occupation_env[:norb_bath].sum().item())

    occupation_unentangle_env = occupation_env[fb_size_list[1]:]
    coeff_unentangle_env = coeff_env[:, fb_size_list[1]:]
    new_ind = (-1 * occupation_unentangle_env).argsort()

    coeff_env[:, fb_size_list[1]:] = coeff_unentangle_env[:, new_ind]
    occupation_unentangle_env = occupation_unentangle_env[new_ind]

    eo_occupation = numpy.hstack(
        (numpy.zeros([fb_size_list[0] + fb_size_list[1]]), occupation_unentangle_env))

    for orb in range(0, fb_size_list[0]):
        coeff_env = numpy.insert(coeff_env, orb, 0.0, axis=1)
    i_temp = 0

    temp_frag_list = frag_list.copy()
    temp_frag_list.sort()
    for orb_total in temp_frag_list:
        coeff_env = numpy.insert(coeff_env, orb_total, 0.0, axis=0)
    for orb_total in frag_list:
        coeff_env[orb_total, i_temp] = 1.0
        i_temp += 1

    LOEO = fix_orbital_sign(coeff_env.copy())

    return LOEO, eo_occupation, fb_size_list, [nelectron_frag, nelectron_bath]


def Impurity_1rdm(cluster_list, coeff, core_occupied,
                  number_active_electrons):
    '''
    Make the 1-RDM for the core part in environment.
    '''
    core_occupied = numpy.asarray(core_occupied).copy()
    core_occupied[cluster_list] = 0
    number_electrons = round(
        number_active_electrons
        - numpy.sum(core_occupied))
    try:
        core_occupied_onerdm = reduce(
            numpy.dot, (coeff, numpy.diag(core_occupied), coeff.T))
    except BaseException:
        core_occupied_onerdm = reduce(
            numpy.dot, (coeff.get(), numpy.diag(core_occupied), coeff.get().T))

    core_index = numpy.where(
        numpy.logical_not(
            numpy.isclose(
                core_occupied,
                0)))[0]
    core_occupied = core_occupied[core_index]
    core_occupied = core_occupied ** 0.5
    try:
        rdm1_core_coeff = reduce(
            numpy.dot, (coeff[:, core_index], numpy.diag(core_occupied)))
    except BaseException:
        rdm1_core_coeff = reduce(
            numpy.dot, (coeff.get()[:, core_index], numpy.diag(core_occupied)))

    assert numpy.allclose(
        numpy.dot(
            rdm1_core_coeff,
            rdm1_core_coeff.T),
        core_occupied_onerdm)

    return number_electrons, core_occupied_onerdm, rdm1_core_coeff
