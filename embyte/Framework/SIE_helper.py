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

from byteqc import lib
from byteqc.embyte.Tools.tool_lib import fix_orbital_sign
import numpy
import cupy
import cupyx
cupy.cuda.set_pinned_memory_allocator(None)


def _free_gpu_memory():
    cupy.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()


def _choose_gpu_tile(nrow, ncol, min_tile=128, max_tile=4096):
    avail = int(lib.gpu_avail_bytes())
    target = max(int(avail * 0.35), 64 * 1024 * 1024)
    tile = min(int(nrow), int(max_tile))
    while tile > min_tile:
        # Two skinny coefficient tiles and one square result tile.
        need = 8 * (2 * tile * max(int(ncol), 1) + tile * tile)
        if need < target:
            break
        tile //= 2
    return max(1, int(tile))


def _scaled_core_coeff_on_gpu(coeff, core_index, sqrt_core_occ):
    nrow = int(coeff.shape[0])
    ncore = int(len(core_index))
    out = cupyx.empty_pinned((nrow, ncore), dtype=numpy.float64, order='C')
    if ncore == 0:
        return out

    row_blksize = _choose_gpu_tile(nrow, ncore)
    sqrt_occ_d = cupy.asarray(sqrt_core_occ, dtype=cupy.float64)
    coeff_is_gpu = isinstance(coeff, cupy.ndarray)
    coeff_cpu = None if coeff_is_gpu else numpy.asarray(coeff)

    try:
        for p0 in range(0, nrow, row_blksize):
            p1 = min(nrow, p0 + row_blksize)
            if coeff_is_gpu:
                block = cupy.ascontiguousarray(coeff[p0:p1][:, core_index])
            else:
                block = cupy.asarray(
                    numpy.asarray(coeff_cpu[p0:p1][:, core_index],
                                  dtype=numpy.float64, order='C'))
            block *= sqrt_occ_d[None, :]
            block.get(out=out[p0:p1], blocking=True)
            del block
            _free_gpu_memory()
    finally:
        del sqrt_occ_d
        _free_gpu_memory()
    return out


def _core_dm_from_coeff_on_gpu(core_coeff):
    nrow, ncore = map(int, core_coeff.shape)
    out = cupyx.empty_pinned((nrow, nrow), dtype=numpy.float64, order='C')
    if ncore == 0:
        out[:] = 0.0
        return out

    tile = _choose_gpu_tile(nrow, ncore)
    for i0 in range(0, nrow, tile):
        i1 = min(nrow, i0 + tile)
        ci = cupy.asarray(core_coeff[i0:i1], dtype=cupy.float64)
        for j0 in range(0, i0 + 1, tile):
            j1 = min(nrow, j0 + tile)
            if j0 == i0:
                cj = ci
            else:
                cj = cupy.asarray(core_coeff[j0:j1], dtype=cupy.float64)
            dm_blk = cupy.dot(ci, cj.T)
            dm_host = cupyx.empty_pinned((i1 - i0, j1 - j0),
                                         dtype=numpy.float64, order='C')
            dm_blk.get(out=dm_host, blocking=True)
            out[i0:i1, j0:j1] = dm_host
            if i0 != j0:
                out[j0:j1, i0:i1] = dm_host.T
            del dm_blk, dm_host
            if j0 != i0:
                del cj
            _free_gpu_memory()
        del ci
        _free_gpu_memory()
    return out


def Get_bath(mol, fb_size_list, frag_list, rdm1_low, logger):
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
    if norb_bath > fb_size_list[0]:
        logger.info(f'bath orbitals number : {norb_bath}, fragment orbitals number : {fb_size_list[0]}, where bath size > fragment size with the therehold of 1e-8.')
        logger.info(f'Use fragment orbitals number as bath size.')
        index_tmp = -numpy.maximum(-occupation_env,
                                occupation_env - 2.0)[new_ind] > 1e-8
        occ_env = (-numpy.maximum(-occupation_env,
                                occupation_env - 2.0)[new_ind])[index_tmp]
        logger.info(f'Occupation of bath orbitals : {occ_env}')
        norb_bath = fb_size_list[0]
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

    core_index = numpy.where(
        numpy.logical_not(
            numpy.isclose(
                core_occupied,
                0)))[0]
    sqrt_core_occ = core_occupied[core_index] ** 0.5
    rdm1_core_coeff = _scaled_core_coeff_on_gpu(
        coeff, core_index, sqrt_core_occ)

    return number_electrons, rdm1_core_coeff
