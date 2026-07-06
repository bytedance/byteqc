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

import gc
from pyscf.lib import prange
from byteqc.cump2.dfmp2 import div_t2
from byteqc.embyte.Tools.tool_lib import fix_orbital_sign
from byteqc.embyte.ERI import eri_trans
from byteqc import lib
from functools import reduce
import numpy
import cupy
cupy.cuda.set_pinned_memory_allocator(None)


def SIE_BNO_builder(low_level_info, fb_size_list, LOEO,
                    EO_occupation, eri=None, vhfopt=None, logger=None):
    '''
    Build BNO in SIE way. See more details in the
    paper with doi 10.1103/physrevx.12.011046
    '''
    fb_norb = fb_size_list[0] + fb_size_list[1]
    fb_list = list(range(fb_norb))
    LO_fb_EO = cupy.asarray(LOEO[:, fb_list])
    fock_fb_EO = reduce(
        cupy.dot,
        (LO_fb_EO.T,
         cupy.asarray(
             low_level_info.fock_LO),
            LO_fb_EO))
    fb_mo_energy, fb_EO_fb_MO = cupy.linalg.eigh(fock_fb_EO)
    fb_EO_fb_MO = fix_orbital_sign(fb_EO_fb_MO)

    fb_n_electron = numpy.sum(EO_occupation[fb_list])
    fb_nocc = int(fb_n_electron // 2)

    tot_nocc = int(low_level_info.mol_full.nelectron // 2)

    assert fb_n_electron % 2 == 0, 'fragment + bath electron number' \
        + f' is {fb_n_electron} which is open-shell system!'
    assert not numpy.isclose(fb_norb, fb_n_electron / 2), \
        'all orb in fragment + bath is occupied orb. ' \
        + 'If you are using IAO to form fragment, please'\
        + ' select a larger basis for refernece mol'

    fb_MO_occ_list = numpy.arange(fb_nocc)
    fb_MO_vir_list = numpy.arange(fb_nocc, len(fb_list))
    LO_fb_MO = cupy.dot(LO_fb_EO, fb_EO_fb_MO)

    LO_fb_MO_occ = LO_fb_MO[:, fb_MO_occ_list].copy()
    LO_fb_MO_vir = LO_fb_MO[:, fb_MO_vir_list].copy()
    fb_EO_fb_MO = fock_fb_EO = LO_fb_EO = LO_fb_MO = None

    subspace_fb_occ_full_vir_LOMO = cupy.hstack(
        (LO_fb_MO_occ, cupy.asarray(low_level_info.LOMO[:, tot_nocc:])))
    subspace_full_occ_fb_vir_LOMO = cupy.hstack(
        (cupy.asarray(low_level_info.LOMO[:, : tot_nocc]), LO_fb_MO_vir))
    subspace_fb_occ_full_vir_mo_energy = list(
        fb_mo_energy.get()[: fb_nocc]) + list(low_level_info.mo_energy[tot_nocc:])
    subspace_full_occ_fb_vir_mo_energy = list(
        low_level_info.mo_energy[: tot_nocc]) + list(fb_mo_energy.get()[fb_nocc:])

    subspace_fb_occ_full_vir_AOMO = cupy.dot(
        low_level_info.AOLO,
        subspace_fb_occ_full_vir_LOMO).get(
        blocking=True)
    subspace_full_occ_fb_vir_AOMO = cupy.dot(
        low_level_info.AOLO,
        subspace_full_occ_fb_vir_LOMO).get(
        blocking=True)

    subspace_fb_occ_full_vir_LOMO = subspace_fb_occ_full_vir_LOMO.get(
        blocking=True)
    subspace_full_occ_fb_vir_LOMO = subspace_full_occ_fb_vir_LOMO.get(
        blocking=True)

    lib.free_all_blocks()
    gc.collect()

    if eri is None:

        ovL_fb_occ_full_vir, voL_full_occ_fb_vir = eri_trans.eri_OVL_SIE_MP2(
            low_level_info.mol_full,
            low_level_info.auxmol,
            subspace_fb_occ_full_vir_AOMO[:, : fb_nocc],
            subspace_fb_occ_full_vir_AOMO[:, fb_nocc:],
            subspace_full_occ_fb_vir_AOMO[:, : tot_nocc],
            subspace_full_occ_fb_vir_AOMO[:, tot_nocc:],
            low_level_info.j2c,
            logger,
            vhfopt=vhfopt,
        )
    else:
        ovL_fb_occ_full_vir, voL_full_occ_fb_vir = eri_trans.eri_ondisk_OVL_SIE_MP2(
            low_level_info.mol_full,
            eri,
            subspace_fb_occ_full_vir_AOMO[:, : fb_nocc],
            subspace_fb_occ_full_vir_AOMO[:, fb_nocc:],
            subspace_full_occ_fb_vir_AOMO[:, : tot_nocc],
            subspace_full_occ_fb_vir_AOMO[:, tot_nocc:],
            logger)

    lib.free_all_blocks()
    gc.collect()

    nocc, nvir, naux = ovL_fb_occ_full_vir.shape

    avail_memory = lib.gpu_avail_bytes() // 8
    a = nvir ** 2
    b = naux * nvir * 2
    c = -1 * avail_memory
    oblk = int(
        min(
            (-1 * (b / 2) + numpy.sqrt(((b / 2) ** 2 - 2 * 4 * a * c))) / (2 * 2 * a),
            nocc
        )
    )

    oslicelist = [slice(i[0], i[1]) for i in prange(0, nocc, oblk)]
    occ_energy = cupy.asarray(subspace_fb_occ_full_vir_mo_energy[: nocc])
    vir_energy = cupy.asarray(subspace_fb_occ_full_vir_mo_energy[nocc:])

    isa_d = cupy.empty((oblk, nvir, naux), dtype='f8')
    jsb_d = cupy.empty((oblk, nvir, naux), dtype='f8')
    t2s_d = cupy.empty((oblk, nvir, oblk, nvir), dtype='f8')
    tuas_d = cupy.empty((oblk, nvir, oblk, nvir), dtype='f8')
    gamma_vir_d = cupy.zeros((nvir, nvir), dtype='f8')
    logger.info('Start to build gamma_vir for the subspace of fb_occ + full_vir')
    for o_ind1, so1 in enumerate(oslicelist):

        so1_len = so1.stop - so1.start
        ia_d = lib.empty_from_buf(isa_d, ovL_fb_occ_full_vir[so1].shape, 'f8')
        ia_d.set(ovL_fb_occ_full_vir[so1])
        t2_d = lib.gemm(
            ia_d.reshape(
                (-1,
                 naux)),
            ia_d.reshape(
                (-1,
                 naux)),
            transb='T',
            buf=t2s_d).reshape(
                (so1_len,
                 nvir,
                 so1_len,
                 nvir))
        div_t2(t2_d, occ_energy[so1], vir_energy, occ_energy[so1], vir_energy)

        tua_d = lib.empty_from_buf(tuas_d, t2_d.shape, 'f8')
        tua_d[:] = t2_d
        tua_d *= 2
        tua_d -= t2_d.transpose(0, 3, 2, 1)
        lib.contraction(
            'iajc',
            t2_d,
            'ibjc',
            tua_d,
            'ab',
            gamma_vir_d,
            alpha=2.0,
            beta=1.0)

        oslicelist2 = oslicelist[o_ind1 + 1:]
        for so2 in oslicelist2:
            so2_len = so2.stop - so2.start
            jb_d = lib.empty_from_buf(
                jsb_d, ovL_fb_occ_full_vir[so2].shape, 'f8')
            jb_d.set(ovL_fb_occ_full_vir[so2])
            t2_d = lib.gemm(
                ia_d.reshape(
                    (-1,
                     naux)),
                jb_d.reshape(
                    (-1,
                     naux)),
                transb='T',
                buf=t2s_d).reshape(
                (so1_len,
                 nvir,
                 so2_len,
                 nvir))
            div_t2(
                t2_d,
                occ_energy[so1],
                vir_energy,
                occ_energy[so2],
                vir_energy)

            tua_d = lib.empty_from_buf(tuas_d, t2_d.shape, 'f8')
            tua_d[:] = t2_d
            tua_d *= 2
            tua_d -= t2_d.transpose(0, 3, 2, 1)
            lib.contraction(
                'iajc',
                t2_d,
                'ibjc',
                tua_d,
                'ab',
                gamma_vir_d,
                alpha=2.0,
                beta=1.0)
            lib.contraction(
                'icja',
                t2_d,
                'icjb',
                tua_d,
                'ab',
                gamma_vir_d,
                alpha=2.0,
                beta=1.0)

        logger.info('mp2 nocc:[%d:%d]/%d' % (so1.start, so1.stop, nocc))

    t2s_d = tuas_d = isa_d = jsb_d = ia_d = t2_d = tua_d = jb_d = vir_energy = occ_energy = None

    mo_vir_coeff = cupy.asarray(subspace_fb_occ_full_vir_LOMO[:, nocc:])
    eo_vir_coeff = cupy.asarray(
        LOEO[:, fb_norb:][:, numpy.isclose(EO_occupation[fb_norb:], 0)])
    moeo_vir_coeff = cupy.dot(mo_vir_coeff.T, eo_vir_coeff)
    gamma_vir_d = reduce(
        cupy.dot,
        (moeo_vir_coeff.T,
         gamma_vir_d,
         moeo_vir_coeff))

    ele_diff_vir, EOBNO_vir = cupy.linalg.eigh(gamma_vir_d)
    EOBNO_vir = fix_orbital_sign(EOBNO_vir)
    LOBNO_vir = cupy.dot(eo_vir_coeff, EOBNO_vir).get()
    ele_diff_vir = ele_diff_vir.get()
    if numpy.any(ele_diff_vir < 0):
        assert numpy.isclose(
            0, ele_diff_vir[numpy.where(ele_diff_vir < 0)[0]]).all()

    mo_vir_coeff = eo_vir_coeff = moeo_vir_coeff = gamma_vir_d \
        = subspace_fb_occ_full_vir_mo_energy = ovL_fb_occ_full_vir \
        = subspace_fb_occ_full_vir_LOMO = None

    lib.free_all_blocks()
    gc.collect()

    nvir, nocc, naux = voL_full_occ_fb_vir.shape

    avail_memory = lib.gpu_avail_bytes() // 8
    a = nocc ** 2
    b = naux * nocc * 2
    c = -1 * avail_memory
    vblk = int(
        min(
            (-1 * (b / 2) + numpy.sqrt(((b / 2) ** 2 - 2 * 4 * a * c))) / (2 * 2 * a),
            nvir
        )
    )

    vslicelist = [slice(i[0], i[1]) for i in prange(0, nvir, vblk)]
    occ_energy = cupy.asarray(subspace_full_occ_fb_vir_mo_energy[: nocc])
    vir_energy = cupy.asarray(subspace_full_occ_fb_vir_mo_energy[nocc:])

    asi_d = cupy.empty((vblk, nocc, naux), dtype='f8')
    bsj_d = cupy.empty((vblk, nocc, naux), dtype='f8')
    t2s_d = cupy.empty((vblk, nocc, vblk, nocc), dtype='f8')
    tuas_d = cupy.empty((vblk, nocc, vblk, nocc), dtype='f8')
    gamma_occ_d = cupy.zeros((nocc, nocc), dtype='f8')
    logger.info('Start to build gamma_occ for the subspace of full_occ + fb_vir')
    for v_ind1, sv1 in enumerate(vslicelist):

        sv1_len = sv1.stop - sv1.start
        ai_d = lib.empty_from_buf(asi_d, voL_full_occ_fb_vir[sv1].shape, 'f8')
        ai_d.set(voL_full_occ_fb_vir[sv1])
        t2_d = lib.gemm(
            ai_d.reshape(
                (-1,
                 naux)),
            ai_d.reshape(
                (-1,
                 naux)),
            transb='T',
            buf=t2s_d).reshape(
                (sv1_len,
                 nocc,
                 sv1_len,
                 nocc))
        div_t2(t2_d, vir_energy[sv1], occ_energy, vir_energy[sv1], occ_energy)

        tua_d = lib.empty_from_buf(tuas_d, t2_d.shape, 'f8')
        tua_d[:] = t2_d
        tua_d *= 2
        tua_d -= t2_d.transpose(0, 3, 2, 1)
        lib.contraction(
            'aibk',
            t2_d,
            'ajbk',
            tua_d,
            'ij',
            gamma_occ_d,
            alpha=-2.0,
            beta=1.0)

        vslicelist2 = vslicelist[v_ind1 + 1:]
        for sv2 in vslicelist2:
            sv2_len = sv2.stop - sv2.start
            bj_d = lib.empty_from_buf(
                bsj_d, voL_full_occ_fb_vir[sv2].shape, 'f8')
            bj_d.set(voL_full_occ_fb_vir[sv2])
            t2_d = lib.gemm(
                ai_d.reshape(
                    (-1,
                     naux)),
                bj_d.reshape(
                    (-1,
                     naux)),
                transb='T',
                buf=t2s_d).reshape(
                (sv1_len,
                 nocc,
                 sv2_len,
                 nocc))
            div_t2(
                t2_d,
                vir_energy[sv1],
                occ_energy,
                vir_energy[sv2],
                occ_energy)

            tua_d = lib.empty_from_buf(tuas_d, t2_d.shape, 'f8')
            tua_d[:] = t2_d
            tua_d *= 2
            tua_d -= t2_d.transpose(0, 3, 2, 1)
            lib.contraction(
                'aibk',
                t2_d,
                'ajbk',
                tua_d,
                'ij',
                gamma_occ_d,
                alpha=-2.0,
                beta=1.0)
            lib.contraction(
                'akbi',
                t2_d,
                'akbj',
                tua_d,
                'ij',
                gamma_occ_d,
                alpha=-2.0,
                beta=1.0)

        logger.info('mp2 nvir:[%d:%d]/%d' % (sv1.start, sv1.stop, nvir))

    t2s_d = tuas_d = asi_d = bsj_d = ai_d = t2_d = \
        tua_d = bj_d = vir_energy = occ_energy = None

    mo_occ_coeff = cupy.asarray(subspace_full_occ_fb_vir_LOMO[:, :nocc])
    eo_occ_coeff = cupy.asarray(
        LOEO[:, fb_norb:][:, numpy.isclose(EO_occupation[fb_norb:], 2)])
    moeo_occ_coeff = cupy.dot(mo_occ_coeff.T, eo_occ_coeff)
    gamma_occ_d = reduce(
        cupy.dot,
        (moeo_occ_coeff.T,
         gamma_occ_d,
         moeo_occ_coeff))

    ele_diff_occ, EOBNO_occ = cupy.linalg.eigh(gamma_occ_d)
    EOBNO_occ = fix_orbital_sign(EOBNO_occ)
    LOBNO_occ = cupy.dot(eo_occ_coeff, EOBNO_occ).get()
    ele_diff_occ = ele_diff_occ.get()
    assert not numpy.any(ele_diff_occ > 1e-10)
    ele_diff_occ = abs(ele_diff_occ)

    LOBNO_env = numpy.hstack((LOBNO_occ, LOBNO_vir))
    LOBNO = LOEO.copy()
    LOBNO[:, fb_norb:] = LOBNO_env
    ele_diff = list(ele_diff_occ) + list(ele_diff_vir)

    mo_occ_coeff = eo_occ_coeff = moeo_occ_coeff = gamma_occ_d \
        = subspace_full_occ_fb_vir_mo_energy = voL_full_occ_fb_vir \
        = subspace_full_occ_fb_vir_LOMO = None

    lib.free_all_blocks()
    gc.collect()

    return LOBNO, ele_diff
