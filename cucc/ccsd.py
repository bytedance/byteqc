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
#
# ByteQC includes code adapted from PySCF (https://github.com/pyscf/pyscf),
# which is licensed under the Apache License 2.0. The original copyright:
#     Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#
#     Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import cc
from byteqc.cucc.buffer import BufferPool, BufArr
from byteqc.cucc import culib, diis
from byteqc import cucc
from byteqc import lib
from pyscf.lib import (logger, prange, param)
import cupy
import numpy
from pyscf.scf import _vhf
from pyscf import ao2mo


def kernel(mycc, eris=None, t1=None, t2=None, max_cycle=50,
           tol=1e-8, tolnormt=1e-6, verbose=None, callback=None):
    log = logger.new_logger(mycc, verbose)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if t1 is None and t2 is None:
        t1, t2 = mycc.get_init_guess(eris)
    elif t2 is None:
        t2 = mycc.get_init_guess(eris)[1]
        t1 = mycc.pool.add('t1', mycc.pool.asarray(t1))
    else:
        t1 = mycc.pool.add('t1', mycc.pool.asarray(t1))
        if mycc.pool.status['t2'] == 0:
            t2 = mycc.pool.add('t2', mycc.pool.asarray(t2))
        else:
            t2 = mycc.pool.add('t2', t2)

    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    eold = 0
    eccsd = mycc.energy(t1, t2, eris)
    log.info('Init E_corr(CCSD) = %.15g', eccsd)

    if isinstance(mycc.diis, diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = diis.DIIS(
            mycc,
            mycc.diis_file,
            incore=mycc.incore_complete,
            pool=mycc.pool)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    conv = False
    t1new = t2new = None
    for istep in range(max_cycle):
        t1new, t2new = mycc.update_amps(t1, t2, eris, t1new, t2new)
        if callback is not None:
            callback(locals())
        normt = cupy.sqrt(culib.diffabs2_t1(t1, t1new)
                          + diffabs2_t2(mycc, t2, t2new))
        if mycc.iterative_damping < 1.0:
            alpha = mycc.iterative_damping
            culib.damp(alpha, t1new, t1)
            culib.damp(alpha, t2new, t2)
            lib.free_all_blocks()
        t1, t1new = t1new, t1
        t2, t2new = t2new, t2
        t1, t2 = mycc.run_diis(t1, t2, istep, normt, eccsd - eold, adiis)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        log.info(
            'cycle = %d  E_corr(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
            istep + 1, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if abs(eccsd - eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('CCSD', *cput0)
    lib.free_all_blocks()
    return conv, eccsd, t1, t2


def diffabs2_t2(mycc, t2, t2new):
    nocc, nvir = t2.shape[1:3]
    memory = mycc.pool.free_memory
    unit = 2 * nvir**2 * nocc
    blksize = min(nocc, int(memory / 8 / unit))
    buf = lib.ArrayBuffer(mycc.pool.empty(
        (blksize * nocc * nvir * nvir * 2 + 1024,), 'f8'))
    buf1 = buf.empty((blksize, nocc, nvir, nvir), 'f8')
    buf2 = buf.empty((blksize, nocc, nvir, nvir), 'f8')
    out = cupy.asarray(0.0)
    for p0, p1 in prange(0, nocc, blksize):
        culib.diffabs2_t2(p0 * nocc * nvir**2, nocc * nvir**2, nvir**2,
                          nvir, t2[p0:p1].ascupy(buf=buf1),
                          t2new[p0:p1].ascupy(buf=buf2), out)
    out = out.item()
    buf = buf1 = buf2 = None
    lib.free_all_blocks()
    return out


def update_amps(mycc, t1, t2, eris, t1new=None, t2new=None):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape

    if t1new is None:
        t1new = mycc.pool.new('t1new', t1.shape, t1.dtype)
    t1new[:] = 0
    if t2new is None:
        t2new = mycc.pool.new('t2new', t2.shape, t2.dtype)
    mycc._add_vvvv(t1, t2, eris, t2sym='jiba', out=t2new)
    t2new *= .5  # *.5 because t2+t2.transpose(1,0,3,2) in the end
    time1 = log.timer_debug1('vvvv', *time0)

    fock = mycc.pool.asarray(eris.fock)
    mo_e = mycc.pool.asarray(eris.mo_energy)
    mo_e_o = mo_e[:nocc]
    mo_e_v = mo_e[nocc:]
    mo_e = None
    mo_e_v += mycc.level_shift

    fov = fock[:nocc, nocc:].copy()
    t1new += fov

    foo = fock[:nocc, :nocc] - cupy.diag(mo_e_o)
    culib.gemm('N', 'T', fov, t1, foo, 0.5, 1)

    fvv = fock[nocc:, nocc:] - cupy.diag(mo_e_v)
    culib.gemm('T', 'N', t1, fov, fvv, -0.5, 1)
    fock = None

    fwVOov, fwVooV = _add_ovvv(mycc, t1, t2, eris, fvv, t1new, t2new)
    time2 = time1 = log.timer_debug1('ovvv', *time1)

    woooo = mycc.pool.new("woooo", (nocc, nocc, nocc, nocc), eris.oooo.dtype)

    memory = mycc.pool.free_memory
    itemsize = t2.dtype.itemsize
    nmax = max(nocc, nvir)
    unit = nocc**2 * (3 * nvir + 2 * nmax)
    blksize = min(nvir, int(memory / itemsize / unit))
    buf = lib.ArrayBuffer(mycc.pool.empty((blksize * unit + 1024,), t2.dtype))
    buf.tag('init')
    log.info('nvir(%d) is sliced to %d with memory %.2fGB' % (
        nvir, blksize, memory / 1e9))

    blkooo = min(nocc, int(buf.bufsize / itemsize / nocc ** 3))
    for p0, p1 in prange(0, nocc, blkooo):
        woooo[p0:p1] = eris.oooo.getitem(
            numpy.s_[p0:p1], buf=buf.arr)

    buf1 = buf.empty((blksize, nocc, nocc, nmax), t2.dtype)
    buf2 = buf.empty((blksize, nocc, nocc, nvir), t2.dtype)
    buf3 = buf.empty((blksize, nocc, nocc, nvir), t2.dtype)
    buf4 = buf.empty((blksize, nocc, nocc, nmax), t2.dtype)
    buf5 = buf.empty((blksize, nocc, nocc, nvir), t2.dtype)
    for p0, p1 in prange(0, nvir, blksize):
        eris_ovoo = eris.ovoo.getitem(numpy.s_[:, p0:p1], order='C',
                                      buf=buf1)

        culib.contraction('kc', t1[:, p0:p1], 'kcji', eris_ovoo, 'ij', foo,
                          alpha=2.0, beta=1.0)
        culib.contraction('icjk', eris_ovoo, 'kc', t1[:, p0:p1], 'ij', foo,
                          alpha=-1.0, beta=1.0)

        culib.contraction('la', t1[:, p0:p1], 'jaik', eris_ovoo,
                          'ljki', woooo, beta=1.0)
        culib.contraction('jaik', eris_ovoo, 'la', t1[:, p0:p1], 'kilj', woooo,
                          beta=1.0)

        wVOov = fwVOov[p0:p1].ascupy(order='C', buf=buf2)
        culib.contraction('jbik', eris_ovoo, 'ka', t1, 'bjia', wVOov,
                          alpha=-1.0, beta=1.0)
        wVooV = fwVooV[p0:p1].ascupy(order='C', buf=buf3)
        culib.contraction('ka', t1, 'kbij', eris_ovoo, 'bija', wVooV, beta=1.0)
        eris_ovoo = None

        eris_oovv = eris.oovv.getitem(numpy.s_[:, :, p0:p1], order='C',
                                      buf=buf1)
        culib.contraction('jb', t1, 'jiab', eris_oovv, 'ia', t1new[:, p0:p1],
                          alpha=-1.0, beta=1.0)
        lib.elementwise_binary('bcad', eris_oovv, 'abcd', wVooV,
                               alpha=-1.0, gamma=1.0)
        tmp = culib.contraction('ic', t1, 'kjbc', eris_oovv, 'ibkj', buf=buf4)
        eris_oovv = None

        eris_ovov = eris.ovov.getitem(numpy.s_[:, p0:p1], order='C',
                                      buf=buf1)
        culib.contraction('jb', t1, 'iajb', eris_ovov, 'ia', t1new[:, p0:p1],
                          alpha=2.0, beta=1.0, opb='CONJ')

        culib.contraction('jbkc', eris_ovov, 'ic', t1, 'jbki', tmp,
                          beta=1.0, opa='CONJ')
        t2newp = t2new[:, :, p0:p1]
        t2newp.enter_kwg = {'buf': buf5}
        with t2newp as arr:
            lib.elementwise_binary('cabd', wVOov, 'abcd', arr, gamma=1.0)
            culib.contraction('ka', t1, 'jbki', tmp, 'jiba', arr,
                              alpha=-1.0, beta=1.0)
            lib.elementwise_binary('acbd', eris_ovov, 'abcd', arr,
                                   alpha=0.5, gamma=1.0)
        t2newp = arr = tmp = None

        lib.axpy(wVooV, wVOov, a=0.5)
        tmp = culib.contraction('kbic', eris_ovov, 'jc', t1, 'kbij',
                                beta=1.0, buf=buf4)
        culib.contraction('kbij', tmp, 'ka', t1, 'bija', wVooV,
                          beta=1.0)
        culib.contraction('kbic', eris_ovov, 'jkca', t2, 'bija', wVooV,
                          alpha=0.5, beta=1.0)
        tmp = None

        culib.contraction('kc', t1, 'iakc', eris_ovov, 'ia', fov[:, p0:p1],
                          alpha=2.0, beta=1.0, opb='CONJ')
        culib.contraction('kaic', eris_ovov, 'kc', t1, 'ia', fov[:, p0:p1],
                          alpha=-1.0, beta=1.0, opa='CONJ')

        tau = t2[:, :, p0:p1].ascupy(order='C', buf=buf4)
        culib.contraction('ia', t1[:, p0:p1], 'jb', t1, 'ijab', tau,
                          alpha=0.5, beta=1.0)

        culib.contraction('jica', tau, 'jcib', eris_ovov, 'ab', fvv,
                          alpha=-2.0, beta=1.0, opb='CONJ')
        culib.contraction('ijca', tau, 'jcib', eris_ovov, 'ab', fvv,
                          beta=1.0, opb='CONJ')

        culib.contraction('iakb', eris_ovov, 'jkab', tau, 'ij', foo,
                          alpha=2.0, beta=1.0, opa='CONJ')
        culib.contraction('iakb', eris_ovov, 'kjab', tau, 'ij', foo,
                          alpha=-1.0, beta=1.0, opa='CONJ')

        culib.contraction('ia', t1[:, p0:p1], 'jb', t1, 'ijab', tau,
                          alpha=0.5, beta=1.0)
        culib.contraction('ijab', tau, 'kalb', eris_ovov, 'ikjl',
                          woooo, beta=1.0, opb='CONJ')

        culib.contraction('ia', t1[:, p0:p1], 'jb', t1, 'ijab', tau,
                          alpha=-1.0, beta=1.0)
        t2p, tau = tau, None

        culib.contraction('jkca', t2p, 'ckib', wVooV,
                          'ijab', t2new, beta=1.0)
        culib.contraction('jkca', t2p, 'ckib', wVooV, 'jiab',
                          t2new, alpha=0.5, beta=1.0)
        wVooV = None

        lib.elementwise_binary('bacd', eris_ovov, 'abcd', wVOov, out=wVOov,
                               alpha=1.0, gamma=1.0, opa='CONJ')
        eris_OVov = lib.empty_from_buf(buf3, (nocc, p1 - p0, nocc, nvir),
                                       dtype=t2.dtype)
        lib.elementwise_binary('cbad', eris_ovov, 'abcd', eris_ovov,
                               out=eris_OVov, alpha=-0.5, gamma=1.0,
                               opa='CONJ', opc='CONJ')
        eris_ovov = None
        tmp = culib.contraction('iakc', eris_OVov, 'jc', t1,
                                'aijk', alpha=-1.0, buf=buf1)
        culib.contraction('aijk', tmp, 'kb', t1, 'aijb', wVOov, beta=1.0)
        culib.contraction('iakc', eris_OVov, 'kjbc', t2, 'aijb', wVOov,
                          alpha=-0.5, beta=1.0)
        tmp = None
        culib.contraction('iakc', eris_OVov, 'jkbc', t2,
                          'aijb', wVOov, beta=1.0)
        eris_OVov = None
        culib.contraction('ckjb', wVOov, 'ikca', t2p,
                          'ijab', t2new, alpha=-1.0, beta=1.0)
        culib.contraction('ckjb', wVOov, 'kica', t2p,
                          'ijab', t2new, alpha=2.0, beta=1.0)
        wVOov = t2p = None
        time2 = log.timer_debug1('voov [%d:%d]' % (p0, p1), *time2)
    fwVOov = fwVooV = None
    buf1 = buf2 = buf3 = buf4 = buf5 = None
    cupy.cuda.Device().synchronize()
    log.timer_debug1('voov', *time1)

    culib.contraction('jb', fov, 'jiba', t2, 'ia', t1new,
                      alpha=2.0, beta=1.0)
    culib.contraction('jb', fov, 'ijba', t2, 'ia', t1new,
                      alpha=-1.0, beta=1.0)

    buf.loadtag('init')
    blksize = min(nvir, int(buf.bufsize / itemsize / nocc**2 / (nocc + nvir)))
    buf1 = buf.empty((nocc, blksize, nocc, nocc))
    buf2 = buf.empty((nocc, nocc, blksize, nvir))
    for p0, p1 in prange(0, nvir, blksize):
        eris_ovoo = eris.ovoo.getitem(numpy.s_[:, p0:p1], buf=buf1)
        t2p = t2[:, :, p0:p1].ascupy(order='C', buf=buf2)
        culib.contraction('jbki', eris_ovoo, 'jkba', t2p, 'ia',
                          t1new, alpha=-2.0, beta=1.0)
        culib.contraction('jbki', eris_ovoo, 'kjba', t2p, 'ia',
                          t1new, beta=1.0)
        eris_ovoo = t2p = None
    buf1 = buf2 = None

    buf.loadtag('init')
    blksize = min(
        nocc, int(buf.bufsize / itemsize / nocc / nvir / (nocc + nvir)))
    buf1 = buf.empty((blksize, nocc, nvir, nocc))
    buf2 = buf.empty((blksize, nocc, nvir, nvir))
    for p0, p1 in prange(0, nocc, blksize):
        tmp = culib.contraction('ikjl', woooo[p0:p1], 'ka', t1, 'ijal',
                                alpha=0.5, buf=buf1)
        t2newp = t2new[p0:p1]
        t2newp.enter_kwg = {'buf': buf2}
        with t2newp as arr:
            culib.contraction('ijal', tmp, 'lb', t1, 'ijab', arr, beta=1.0)
            culib.contraction('ikjl', woooo[p0:p1], 'klab', t2, 'ijab',
                              arr, alpha=0.5, beta=1.0)
        tmp = t2newp = arr = None
    woooo = buf1 = buf2 = None

    buf.loadtag('init')
    ft_ij = buf.empty(foo.shape, foo.dtype)
    ft_ij[:] = foo
    culib.gemm('N', 'T', fov, t1, ft_ij, alpha=0.5, beta=1.0)
    ft_ab = buf.empty(fvv.shape, fvv.dtype)
    ft_ab[:] = fvv
    culib.gemm('T', 'N', t1, fov, ft_ab, alpha=-0.5, beta=1.0)
    fov = None
    blksize = min(
        nocc, int(buf.bufsize / itemsize / nocc / nvir / nvir / 2))
    buf1 = buf.empty((blksize, nocc, nvir, nvir))
    buf2 = buf.empty((blksize, nocc, nvir, nvir))
    for p0, p1 in prange(0, nocc, blksize):
        t2newp = t2new[p0:p1]
        t2newp.enter_kwg = {'buf': buf1}
        with t2newp as arr:
            for q0, q1 in prange(0, nocc, blksize):
                t2q = t2[q0:q1].ascupy(buf=buf2)
                culib.contraction('ki', ft_ij[q0:q1, p0:p1], 'kjab', t2q,
                                  'ijab', arr, alpha=-1.0, beta=1.0)
                if p0 == q0:
                    culib.contraction('ijac', t2q, 'bc', ft_ab,
                                      'ijab', arr, beta=1.0)
        t2newp = arr = t2q = None
    buf = buf1 = buf2 = ft_ij = ft_ab = None

    culib.gemm('N', 'T', t1, fvv, t1new, beta=1.0)
    culib.gemm('T', 'N', foo, t1, t1new, alpha=-1.0, beta=1.0)
    foo = fvv = None
    ker = cupy.ElementwiseKernel(
        'int64 m, raw T eo, raw T ev', 'T out', '''
            size_t a = i / m;
            size_t b = i % m;
            out /= eo[a] - ev[b];
        ''', 't1new_mo_e')
    with t1new as arr:
        ker(nvir, mo_e_o, mo_e_v, arr)
    arr = None

    # t2new = t2new + t2new.transpose(1,0,3,2)
    ker1 = cupy.ElementwiseKernel(
        'int64 m, int64 ind, raw T eo, raw T ev', 'T out', '''
            size_t c = i % m;
            size_t a = i / m;
            size_t b = a % m;
            a /= m;
            out /= eo[ind]  + eo[a] - ev[b] - ev[c];
        ''', 't2new_eia_1')
    ker2 = cupy.ElementwiseKernel(
        'int64 m, T eo, raw T ev', 'T out', '''
            size_t a = i / m;
            size_t b = i % m;
            out /= 2*eo - ev[a] - ev[b];
        ''', 't2new_eia_2')
    for i in range(nocc):
        if i > 0:
            with t2new[i, :i] as arr:
                arr += t2new[:i, i].ascupy().transpose(0, 2, 1)
                ker1(nvir, i, mo_e_o, mo_e_v, arr)
                t2new[:i, i] = arr.transpose(0, 2, 1)
        with t2new[i, i] as arr:
            arr += arr.T
            ker2(nvir, mo_e_o[i], mo_e_v, arr)
    mo_e_o = mo_e_v = arr = None
    time0 = log.timer_debug1('update t1 t2', *time0)
    lib.free_all_blocks()
    return t1new, t2new


class EriWrapper:
    '''
    A unified interface for CD-ERI and ERI.
    '''

    def __init__(self, l1, l2=None):
        '''`l2=None` means that the ERI is without density-fitting.'''
        self.l1 = l1
        self.l2 = l2

    def getitem(self, k, **kwgs):
        '''Automatically check wether density-fitting is used. If yes a gemm is performed. If not a slicing is performed. `**kwgs` is passed to `BufArr.ascupy`.'''
        if isinstance(k, slice):
            k = (k,)
        if self.l2 is None:
            return self.l1[*k].ascupy(**kwgs)
        else:
            s = [slice(None)] * 4
            s[:len(k)] = k
            l1 = self.l1[:, *s[:2]]
            l2 = self.l2[:, *s[2:]]
            naux = l1.shape[0]
            r = culib.gemm('T', 'N', l1.reshape(naux, -1),
                           l2.reshape(naux, -1), buf=kwgs.get('buf'))
            return r.reshape((*l1.shape[1:], *l2.shape[1:]))

    def __getitem__(self, k, **kwgs):
        '''See `EriWrapper.getitem`.'''
        return self.getitem(k, **kwgs)

    def __setitem__(self, k, val):
        assert self.l2 is None, "setitem is not allowed when l2 is not None"
        return self.l1.__setitem__(k, val)

    def __getattr__(self, name):
        return self.l1.__getattribute__(name)

    def __broad_as__(self):
        l1 = cupy.asarray(self.l1)
        if self.l2 is None:
            return l1, EriWrapper(l1)
        l2 = cupy.asarray(self.l2)
        return (l1, l2), EriWrapper(l1, l2)

    def __broad_new__(self):
        l1 = cupy.empty(self.l1.shape, dtype=self.l1.dtype)
        if self.l2 is None:
            return l1, EriWrapper(l1)
        l2 = cupy.empty(self.l2.shape, dtype=self.l2.dtype)
        return (l1, l2), EriWrapper(l1, l2)


class CCSD(cc.ccsd.CCSD):

    update_amps = update_amps

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 gpulim=None, cpulim=None, pool=None, path=param.TMPDIR,
                 mem_ratio=0.7):
        cc.ccsd.CCSD.__init__(self, mf, frozen=frozen, mo_coeff=mo_coeff,
                              mo_occ=mo_occ)
        if pool is not None:
            self.pool = pool
        else:
            self.pool = BufferPool(gpulim, cpulim, path, verbose=self.verbose)

        self.mem_ratio = mem_ratio
        self._keys.update(['pool, mem_ratio'])

    def density_fit(self, auxbasis=None, with_df=None):
        from byteqc.cucc import dfccsd
        mycc = dfccsd.RCCSD(self._scf, self.frozen, self.mo_coeff, self.mo_occ,
                            pool=getattr(self, 'pool', None))
        if with_df is not None:
            mycc.with_df = with_df
        if mycc.with_df.auxbasis != auxbasis:
            import copy
            mycc.with_df = copy.copy(mycc.with_df)
            mycc.with_df.auxbasis = auxbasis
        return mycc

    def dump_flags(self, verbose=None):
        cc.ccsd.CCSD.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        r = culib.current_memory()
        log.info('max_GPU_memory %d MB (current use %d MB)', r[1], r[0])

    def ccsd(self, t1=None, t2=None, eris=None):
        assert (self.mo_coeff is not None)
        assert (self.mo_occ is not None)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)

        self.e_hf = getattr(eris, 'e_hf', None)
        if self.e_hf is None:
            self.e_hf = self._scf.e_tot

        self.converged, self.e_corr, self.t1, self.t2 = \
            kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                   tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                   verbose=self.verbose, callback=self.callback)
        self._finalize()
        return self.e_corr, self.t1, self.t2

    def init_amps(self, eris=None):
        time0 = logger.process_clock(), logger.perf_counter()
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        nocc = self.nocc
        nvir = eris.mo_energy.size - nocc

        t1 = self.pool.asarray(eris.fock[:nocc, nocc:])
        t2 = self.pool.new('t2', (nocc, nocc, nvir, nvir),
                           dtype=eris.ovov.dtype)
        mo_e = self.pool.asarray(eris.mo_energy)
        ker = cupy.ElementwiseKernel(
            'int64 no, int64 nv, raw T e', 'T t1',
            '''
                size_t b = i % nv;
                size_t a = i / nv;
                t1 /= (e[a] - e[no+b]);
            ''', 'init_amps_t1')
        with t1 as t1arr:
            ker(nocc, nvir, mo_e, t1arr)

        memory = self.pool.free_memory
        itemsize = t2.dtype.itemsize
        blksize = min(nocc, int(memory / itemsize / (2 * nvir**2 * nocc)))
        ker = cupy.ElementwiseKernel(
            'int64 no, int64 nv, int64 p0, raw T ovov, raw T e',
            'T t2', '''
                size_t d = i % nv;
                size_t a = i / nv;
                size_t c = a % nv;
                a /= nv;
                size_t b = a % no;
                a /= no;
                t2 = ovov[a*nv*no*nv + c*no*nv + b*nv + d] /
                        (e[b] - e[no+d] + e[a+p0] - e[no+c]);
            ''', 'init_amps_t2')

        emp2 = 0
        eris_ovov = None
        for p0, p1 in prange(0, nocc, blksize):
            eris_ovov = eris.ovov.getitem(numpy.s_[p0:p1], buf=eris_ovov)
            with t2[p0:p1] as t2arr:
                ker(nocc, nvir, p0, eris_ovov, mo_e, t2arr)
                emp2 += culib.contraction(
                    'ijab', t2arr, 'iajb', eris_ovov, '', alpha=2.0).item()
                eris_ovov = eris.ovov.getitem(numpy.s_[:, :, p0:p1],
                                              buf=eris_ovov)
                emp2 -= culib.contraction(
                    'jiab', t2arr, 'iajb', eris_ovov, '', alpha=1.0).item()
            t2arr = None
        eris_ovov = mo_e = None
        self.emp2 = emp2.real

        e_hf = self.e_hf or eris.e_hf

        logger.info(self, 'Init t2, MP2 energy = %.15g  E_corr(MP2) %.15g',
                    e_hf + self.emp2, self.emp2)
        logger.timer(self, 'init mp2', *time0)
        lib.free_all_blocks()
        return self.emp2, t1, t2

    def energy(mycc, t1=None, t2=None, eris=None):
        '''CCSD correlation energy'''
        if t1 is None:
            t1 = mycc.t1
        if t2 is None:
            t2 = mycc.t2
        if eris is None:
            eris = mycc.ao2mo()

        nocc, nvir = t1.shape
        fock = cupy.asarray(eris.fock[:nocc, nocc:])
        e = culib.contraction('ia', fock, 'ia', t1, '').item() * 2
        fock = None

        memory = mycc.pool.free_memory
        itemsize = t2.dtype.itemsize
        blksize = min(nocc, int(memory / itemsize / (2 * nvir**2 * nocc)))
        buf = lib.ArrayBuffer(mycc.pool.empty(
            (blksize * 2 * nocc * nvir**2 + 1024), dtype=t2.dtype))
        buf1 = buf.empty((blksize, nvir, nocc, nvir), dtype=t2.dtype)
        buf2 = buf.empty((blksize, nocc, nvir, nvir), dtype=t2.dtype)
        for p0, p1 in prange(0, nocc, blksize):
            eris_ovov = eris.ovov.getitem(numpy.s_[p0:p1], buf=buf1)
            tau = t2[p0:p1].ascupy(copy=True, buf=buf2)
            tau = culib.contraction(
                'ia', t1[p0:p1], 'jb', t1, 'ijab', tau, beta=1.0)
            e += culib.contraction('ijab', tau, 'iajb',
                                   eris_ovov, '', alpha=2.0).item()
            eris_ovov = eris.ovov.getitem(
                numpy.s_[:, :, p0:p1], buf=eris_ovov)
            e -= culib.contraction('iajb', eris_ovov, 'jiab', tau, '').item()
        buf = buf1 = buf2 = eris_ovov = tau = None
        lib.free_all_blocks()
        if abs(e.imag) > 1e-4:
            logger.warn(
                mycc, 'Non-zero imaginary part found in CCSD energy %s', e)
        return e.real

    def _add_vvvv(mycc, t1, t2, eris, out=None, with_ovvv=None, t2sym=None):
        if t2sym in ('jiba', '-jiba', '-jiab'):
            log = logger.Logger(mycc.stdout, mycc.verbose)
            return eris._contract_vvvv_t2(mycc, t1, t2, mycc.direct, out, log)
        else:
            assert False, "Not implemented"

    def run_diis(self, t1, t2, istep, normt, de, adiis):
        if (adiis and istep >= self.diis_start_cycle and abs(
                de) < self.diis_start_energy_diff):
            vec = adiis.update(amplitudes_to_vector(self, t1, t2))
            vector_to_amplitudes(self, vec, self.nmo, self.nocc, t1=t1, t2=t2)
            vec = None
            lib.free_all_blocks()
            logger.debug1(self, 'DIIS for step %d', istep)
        return t1, t2

    def ao2mo(self, mo_coeff=None):
        if getattr(self._scf, 'with_df', None):
            raise RuntimeError(
                'use dfccsd.CCSD for the DF-CCSD calculations')
        return _make_eris_incore(self, mo_coeff)

    def tocpu(self):
        if isinstance(self, cucc.dfccsd.RCCSD):
            from pyscf.cc import dfccsd
            c = dfccsd.RCCSD(self._scf)
        elif isinstance(self, cucc.ccsd.RCCSD):
            c = cc.ccsd.RCCSD(self._scf)
        else:
            assert False, 'Not implement'
        for k in self.__dict__:
            v = getattr(self, k, None)
            if v is not None:
                if getattr(v, 'asnumpy', None) is not None:
                    setattr(c, k, v.asnumpy())
                else:
                    setattr(c, k, v)
        return c

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        '''Un-relaxed 1-particle density matrix in MO space'''
        from byteqc.cucc import ccsd_rdm
        if t1 is None:
            t1 = self.t1
        if t2 is None:
            t2 = self.t2
        if l1 is None:
            l1 = self.l1
        if l2 is None:
            l2 = self.l2
        if l1 is None:
            l1, l2 = self.solve_lambda(t1, t2)
        return ccsd_rdm.make_rdm1(self, t1, t2, l1, l2, ao_repr=ao_repr)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        '''2-particle density matrix in MO space.  The density matrix is
        stored as

        dm2[p,r,q,s] = <p^+ q^+ s r>
        '''
        from byteqc.cucc import ccsd_rdm
        if t1 is None:
            t1 = self.t1
        if t2 is None:
            t2 = self.t2
        if l1 is None:
            l1 = self.l1
        if l2 is None:
            l2 = self.l2
        if l1 is None:
            l1, l2 = self.solve_lambda(t1, t2)
        return ccsd_rdm.make_rdm2(self, t1, t2, l1, l2, ao_repr=ao_repr)

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, eris=None):
        from byteqc.cucc import ccsd_lambda
        if t1 is None:
            t1 = self.t1
        if t2 is None:
            t2 = self.t2
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
            ccsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                               max_cycle=self.max_cycle,
                               tol=self.conv_tol_normt,
                               verbose=self.verbose)
        return self.l1, self.l2


def _add_ovvv(mycc, t1, t2, eris, fvv, t1new, t2new):
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nmax = max(nocc, nvir)

    wVOov = mycc.pool.new('wVOov', (nvir, nocc, nocc, nvir), 'f8')
    wVooV = mycc.pool.new('wVooV', (nvir, nocc, nocc, nvir), 'f8')
    memory = mycc.pool.free_memory
    itemsize = t2.dtype.itemsize
    blk1 = min(
        nocc, int(memory / itemsize / (nvir * nmax**2 + nocc * nmax * nvir)))
    if blk1 < 2:
        blk1 = 2
        blk2 = min(nvir, int(memory / itemsize / (nmax**2 + nocc * nmax)))
        log.info(
            'nocc(%d) and nvir(%d) is sliced to %d and %d with memory %.2fGB'
            % (nocc, nvir, blk1, blk2, memory / 1e9))
    else:
        blk2 = nvir
        log.info('nocc(%d) is sliced to %d with memory %.2fGB' % (
            nocc, blk1, memory / 1e9))

    buf1 = mycc.pool.empty((blk1, nmax, nmax, blk2), 'f8')
    buf2 = mycc.pool.empty((blk1, nocc, nmax, blk2), 'f8')
    for p0, p1 in prange(0, nocc, blk1):
        for q0, q1 in prange(0, nvir, blk2):
            ovvv = eris.ovvv.getitem(numpy.s_[p0:p1, :, :, q0:q1], buf=buf1)

            culib.contraction('kc', t1[p0:p1], 'kcab', ovvv, 'ab',
                              fvv[:, q0:q1], alpha=2.0, beta=1.0)
            culib.contraction('kbca', ovvv, 'kc', t1[p0:p1], 'ab', fvv[q0:q1],
                              alpha=-1.0, beta=1.0)

            culib.contraction('ibac', ovvv, 'jc', t1[:, q0:q1], 'bija',
                              wVOov[:, p0:p1])
            culib.contraction('ja', t1, 'iabc', ovvv, 'bijc',
                              wVooV[:, p0:p1, :, q0:q1], alpha=-1.0)

            culib.contraction('bacj', ovvv, 'biac', t2[p0:p1], 'ij',
                              t1new[:, q0:q1], alpha=2.0, beta=1.0)
            culib.contraction('ibac', t2[:, p0:p1], 'bacj', ovvv, 'ij',
                              t1new[:, q0:q1], alpha=-1.0, beta=1.0)

            tmp = culib.contraction('kabc', ovvv, 'ijab', t2, 'ijkc',
                                    buf=buf2)
            culib.contraction('ijkc', tmp, 'kd', t1[p0:p1], 'ijdc',
                              t2new[:, :, :, q0:q1], alpha=-1.0, beta=1.0)

            tmp = culib.contraction('kabc', ovvv, 'ia', t1, 'kibc', buf=buf2)
            tmp = culib.contraction('kibc', tmp, 'jb', t1, 'kijc', buf=buf1)
            tmp = culib.contraction('kijc', tmp, 'kd', t1[p0:p1], 'ijdc',
                                    t2new[:, :, :, q0:q1], alpha=-1.0,
                                    beta=1.0)
            if blk2 != nvir:
                time1 = log.timer_debug1(
                    'ovvv [%d:%d]-[%d:%d]' % (p0, p1, q0, q1), *time1)
        if blk2 == nvir:
            time1 = log.timer_debug1('ovvv [%d:%d]' % (p0, p1), *time1)

    buf1 = buf2 = tmp = ovvv = None
    lib.free_all_blocks()
    return wVOov, wVooV


def amplitudes_to_vector(self, t1, t2, out=None):
    nocc, nvir = t1.shape
    nov = nocc * nvir
    if out is None:
        out = self.pool.new('vec', (nov + nov * (nov + 1) // 2,), t1.dtype)
    else:
        assert isinstance(out, BufArr), 'out must be a BufArr'
    out[:nov] = t1.ravel()
    ker = cupy.ElementwiseKernel(
        'int64 nocc, int64 nvir, int64 p, int64 off, raw T t2', 'T out',
        '''
            size_t j = i + off;
            size_t a = floor(sqrt(0.25+2*j)-0.5);
            size_t c = j-a*(a+1)/2;
            size_t b = a % nvir;
            a = a / nvir - p;
            size_t d = c % nvir;
            c = c / nvir;
            out = t2[a*nvir*nvir*nocc + c*nvir*nvir + b*nvir + d];
        ''', 'amp2vec')
    memory = self.pool.free_memory
    unit = nocc * nvir * nvir * 2
    blk = min(nocc, int(memory / 8 / unit))
    buf = lib.ArrayBuffer(self.pool.empty(
        (blk * nocc * nvir * nvir * 2 + 1024,), 'f8'))
    buf1 = buf.empty((blk, nocc, nvir, nvir), 'f8')
    buf2 = buf.empty((blk, nocc, nvir, nvir), 'f8')
    for p0, p1 in prange(0, nocc, blk):
        n0 = p0 * nvir
        off0 = n0 * (n0 + 1) // 2
        n1 = p1 * nvir
        off1 = n1 * (n1 + 1) // 2
        outp = out[off0 + nov:off1 + nov]
        outp.enter_kwg = {'buf': buf1}
        with outp as arr:
            ker(nocc, nvir, p0, off0,
                t2[p0:p1].ascupy(order='C', buf=buf2), arr)
        outp = arr = None
        cupy.cuda.Device().synchronize()
    buf = buf1 = buf2 = None
    lib.free_all_blocks()
    return out


def vector_to_amplitudes(mycc, vector, nmo, nocc, t1, t2):
    nvir = nmo - nocc
    nov = nocc * nvir
    t1[:] = vector[:nov].reshape((nocc, nvir))
    assert not cupy.iscomplexobj(vector)
    ker = cupy.ElementwiseKernel(
        'int64 nvir, int64 p0, int64 p1, int64 off, raw T v', 'T t2', '''
            size_t d = i % nvir;
            size_t j = i / nvir;
            size_t c =  j % nvir;
            j /= nvir;
            size_t b = j % p1;
            size_t a = j / p1 + p0;
            size_t x = a * nvir + c;
            size_t y = b * nvir + d;
            if (x >= y)
                t2 = v[x * (x + 1) / 2 + y - off];
            else
                t2 = v[y * (y + 1) / 2 + x - off];
        ''', 'vec2amp')
    memory = mycc.pool.free_memory
    unit = nocc * nvir * nvir * 2
    blk = min(nocc, int(memory / 8 / unit))
    buf = lib.ArrayBuffer(mycc.pool.empty(
        (blk * nocc * nvir * nvir * 2 + 1024,), 'f8'))
    buf1 = buf.empty((blk, nocc, nvir, nvir), 'f8')
    buf2 = buf.empty((blk, nocc, nvir, nvir), 'f8')
    for p0, p1 in prange(0, nocc, blk):
        n0 = p0 * nvir
        off0 = n0 * (n0 + 1) // 2
        n1 = p1 * nvir
        off1 = n1 * (n1 + 1) // 2
        t2p = lib.empty_from_buf(buf1, (p1 - p0, p1, nvir, nvir))
        v = vector[off0 + nov:off1 + nov].ascupy(order='C', buf=buf2)
        ker(nvir, p0, p1, off0, v, t2p)
        t2[p0:p1, :p1] = t2p
        if p0 != 0:
            t2q = lib.empty_from_buf(buf2, (p0, p1 - p0, nvir, nvir))
            lib.elementwise_binary('badc', t2p[:, :p0], 'abcd', t2q)
            t2[:p0, p0:p1] = t2q
        t2p = t2q = v = None
    buf = buf1 = buf2 = None
    lib.free_all_blocks()
    return t1, t2


# modified funcs for _ChemistsERIs._common_init_
# region
def get_frozen_mask(mp):
    '''Get boolean mask for the restricted reference orbitals.

    In the returned boolean (mask) array of frozen orbital indices, the
    element is False if it corresonds to the frozen orbital.
    '''
    moidx = cupy.ones(mp.mo_occ.size, dtype=cupy.bool_)

    if mp._nmo is not None:
        moidx[mp._nmo:] = False
    elif mp.frozen is None:
        pass
    elif isinstance(mp.frozen, (int, numpy.integer, cupy.integer)):
        moidx[:mp.frozen] = False
    elif len(mp.frozen) > 0:
        moidx[list(mp.frozen)] = False
    else:
        raise NotImplementedError
    return moidx


def dot_eri_dm(eri, dm, hermi=0, with_j=True, with_k=True):
    nao = dm.shape[-1]
    if eri.dtype == numpy.complex128 or eri.size == nao**4:
        dm = cupy.asarray(dm)
        eri = cupy.asarray(eri)
        eri = eri.reshape((nao, ) * 4)
        dms = dm.reshape(-1, nao, nao)
        vj = vk = None
        if with_j:
            vj = cupy.einsum('ijkl,xji->xkl', eri, dms)
            vj = vj.reshape(dm.shape)
        if with_k:
            vk = cupy.einsum('ijkl,xjk->xil', eri, dms)
            vk = vk.reshape(dm.shape)
    else:
        # vj, vk = incore(eri, dm.real, hermi, with_j, with_k)
        # # this extremly un optimized kernel is slow than CPU version
        vj, vk = _vhf.incore(eri, dm.get().real, hermi, with_j, with_k)
        vj = cupy.asarray(vj)
        vk = cupy.asarray(vk)
        if dm.dtype == numpy.complex128:
            # vs = incore(eri, dm.imag, 0, with_j, with_k)
            vs0, vs1 = _vhf.incore(eri, dm.get().imag, hermi, with_j, with_k)
            vs = (cupy.asarray(vs0), cupy.asarray(vs1))
            if with_j:
                vj = vj + vs[0] * 1j
            if with_k:
                vk = vk + vs[1] * 1j
    return vj, vk
# endregion


class _ChemistsERIs(cc.ccsd._ChemistsERIs):
    def _contract_vvvv_t2(
            self, mycc, t1, t2, vvvv_or_direct=False, out=None, verbose=None):
        if isinstance(vvvv_or_direct, numpy.ndarray):
            vvvv = vvvv_or_direct
        elif vvvv_or_direct:  # AO-direct contraction
            vvvv = None
        else:
            vvvv = self.vvvv
        return _contract_vvvv_t2(mycc, self.mol, vvvv, t1, t2, out, verbose)


def _make_eris_incore(mycc, mo_coeff=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    mycc.pool.memory_status(nocc, nvir, mem_ratio=mycc.mem_ratio)

    # eri1 = full(cupy.asarray(mycc._scf._eri), eris.mo_coeff)
    eri1 = ao2mo.incore.full(mycc._scf._eri, eris.mo_coeff)
    if eri1.ndim == 4:
        eri1 = ao2mo.restore(4, eri1, nmo)
    eri1 = mycc.pool.asarray(eri1)

    eris.oooo = EriWrapper(mycc.pool.new(
        'oooo', (nocc, nocc, nocc, nocc), 'f8'))
    eris.ovoo = EriWrapper(mycc.pool.new(
        'ovoo', (nocc, nvir, nocc, nocc), 'f8'))
    eris.ovov = EriWrapper(mycc.pool.new(
        'ovov', (nocc, nvir, nocc, nvir), 'f8'))
    eris.oovv = EriWrapper(mycc.pool.new(
        'oovv', (nocc, nocc, nvir, nvir), 'f8'))
    eris.ovvv = EriWrapper(mycc.pool.new(
        'ovvv', (nocc, nvir, nvir, nvir), 'f8'))
    eris.vvvv = EriWrapper(mycc.pool.new(
        'vvvv', (nvir, nvir, nvir, nvir), 'f8'))

    ij = 0
    outbuf = mycc.pool.empty((nmo, nmo, nmo))
    for i in range(nocc):
        buf = lib.unpack_tril(eri1[ij:ij + i + 1], out=outbuf[:i + 1])
        eris.oooo[i, :i + 1] = buf[:, :nocc, :nocc]
        eris.oooo[:i + 1, i] = buf[:, :nocc, :nocc]
        eris.oovv[i, :i + 1] = buf[:, nocc:, nocc:]
        eris.oovv[:i + 1, i] = buf[:, nocc:, nocc:]
        ij += i + 1

    for i in range(nocc, nmo):
        buf = lib.unpack_tril(eri1[ij:ij + i + 1], out=outbuf[:i + 1])
        eris.ovoo[:, i - nocc] = buf[:nocc, :nocc, :nocc]
        eris.ovov[:, i - nocc] = buf[:nocc, :nocc, nocc:]
        eris.ovvv[:, i - nocc] = buf[:nocc, nocc:, nocc:]
        eris.vvvv[:i - nocc + 1, i - nocc] = buf[nocc:i + 1, nocc:, nocc:]
        eris.vvvv[i - nocc, :i - nocc + 1] = buf[nocc:i + 1, nocc:, nocc:]
        ij += i + 1
    logger.timer(mycc, 'CCSD integral transformation', *cput0)
    return eris


def _contract_vvvv_t2(mycc, mol, vvvv, t1, t2, out=None, verbose=None):
    '''Ht2 = numpy.einsum('ijcd,acbd->ijab', t2, vvvv)
    '''
    assert vvvv is not None, 'ERI without df should have vvvv'
    nocc, nvir = t2.shape[1:3]
    culib.contraction('ijcd', t2, 'acbd', vvvv[:], 'ijab', out)
    if t1 is not None:
        tmp = mycc.pool.empty((nocc, nvir, nvir, nvir), t1.dtype)
        culib.contraction('jd', t1, 'acbd', vvvv[:], 'jacb', tmp)
        culib.contraction('ic', t1, 'jacb', tmp, 'ijab', out, beta=1.0)
        tmp = None
    lib.free_all_blocks()
    return out
