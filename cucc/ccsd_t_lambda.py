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

'''
Spin-free lambda equation of RHF-CCSD(T)

Ref:
JCP 147, 044104 (2017); DOI:10.1063/1.4994918
'''

from byteqc.cucc import culib, ccsd_rdm, ccsd_lambda, ccsd_t
import numpy
import cupy
import cupyx
from pyscf.lib import logger, prange, direct_sum
from pyscf.cc.ccsd import BLKMIN
from byteqc import lib
import numba
import concurrent.futures

# generate the [off:off+n] indices for [a,b,c] with a >= b >= c with n the length of output.
gen_abc = cupy.ElementwiseKernel(
    'int64 off', 'T outa, T outb, T outc',
    '''
        size_t y = i + off;
        size_t a = floor(cbrt(6.0*y));
        size_t t = a*(a+1)*(a+2)/6;
        if (t > y)
        {
            a -= 1;
            y -= a*(a+1)*(a+2)/6;
        }
        else
            y -= t;
        size_t b = floor(sqrt(2.0*y));
        t = b*(b+1)/2;
        size_t c;
        if (t > y)
        {
            b -= 1;
            c = y - b*(b+1)/2;
        }
        else
            c = y - t;
        outa = a;
        outb = b;
        outc = c;
    ''', 'gen_abc')

# Divided the energy and alpha in-place.
# divided by E_ij^ab

# divided by D_ijk^abc
div_d3 = cupy.ElementwiseKernel(
    'int64 nocc, raw int32 a, raw int32 b, raw int32 c, raw T eocc, '
    'raw T evir', 'T out',
    '''
        int K = i % nocc;
        size_t iabc = i / nocc;
        int J = iabc % nocc;
        iabc /= nocc;
        int I = iabc % nocc;
        
        iabc /= nocc;
        int A = a[iabc];
        int B = b[iabc];
        int C = c[iabc];

        out /= (eocc[I] + eocc[J] + eocc[K] - evir[A] - evir[B] - evir[C]);
    ''', 'div_d3')

# if i != j != k, then multiply the matrix element by 2
sym_multiply = cupy.ElementwiseKernel(
    'int64 n, raw int32 indi, raw int32 indj, raw int32 indk',
    'T out',
    '''
    int C = i % n;
    size_t temp = i / n;
    int B = temp % n;
    temp /= n;
    int A = temp % n;
        
    size_t n_idx = temp / n;
    int I = indi[n_idx];
    int J = indj[n_idx];
    int K = indk[n_idx];

    T alpha = 1.0;
    if (I != J && J != K)
        alpha = 2.0;

    out = out * alpha;
    ''', 'sym_multiply')

# if i != j != k, no change, else divide by 2
sym_divide = cupy.ElementwiseKernel(
    'int64 n, raw int32 indi, raw int32 indj, raw int32 indk',
    'T out',
    '''
    int C = i % n;
    size_t temp = i / n;
    int B = temp % n;
    temp /= n;
    int A = temp % n;
        
    size_t n_idx = temp / n;
    int I = indi[n_idx];
    int J = indj[n_idx];
    int K = indk[n_idx];

    T alpha = 1.0;
    if (I == J || J == K || I == K)
        alpha = 2.0;

    out = out / alpha;
    ''', 'sym_divide')

# exta, extb = r.shape
# for i in range(n):
#   out[i, :] = r[:, indb[i]]
take01 = cupy.ElementwiseKernel(
    'int64 exta, int64 extb, raw int32 indb, raw T r', 'T out', '''
        int a = i % exta;
        int m = i / exta;
        out = r[a * extb  + indb[m]];
    ''', 'take01')

# exta, extb = r.shape
# for i in range(n):
#   out[i, :] = r[indb[i], :]
take10 = cupy.ElementwiseKernel(
    'int64 extb, raw int32 inda, raw T r', 'T out', '''
        int b = i % extb;
        int m = i / extb;
        out = r[inda[m] * extb + b];
    ''', 'take10')

# exta, extb, extc = r.shape
# for i in range(n):
#   out[i, :, :] = r[:, indb[i], :]
take010 = cupy.ElementwiseKernel(
    'int64 exta, int64 extb, int64 extc, raw int32 indb, raw T r', 'T out', '''
        int c = i % extc;
        size_t ind = i / extc;
        int a = ind % exta;
        int m = ind / exta;
        out = r[a * extb * extc + indb[m] * extc + c];
    ''', 'take010')

# nocc, nocc, nvir, nvir = r.shape
# for i in range(n):
#   out[i, :, :, :] = r[:, :, indb[i], :].transpose((1,0,2,3))
take0010_t = cupy.ElementwiseKernel(
    'int64 nocc, int64 nvir, raw int32 indb, raw T r', 'T out', '''
        int c = i % nvir;
        size_t ind = i / nvir;
        int b = ind % nocc;
        ind /= nocc;
        int a = ind % nocc;
        int m = ind / nocc;
        out = r[(b * nocc + a) * nvir * nvir + indb[m] * nvir + c];
    ''', 'take0010_t')

# nocc, extb, extc = r.shape
# for i in range(n):
#   out[i, (:, :), :] = r[(:, :), indb[i], :].transpose((1,0,2,3))
#   #(:, :) are sliced by p0:p1
take0010_s_t = cupy.ElementwiseKernel(
    'int64 nocc, int64 nvir, int64 p0, int64 p1, raw int32 indb,'
    ' raw T r', 'raw T out', '''
        int c = i % nvir;
        size_t ind = i / nvir;
        int p = ind % (p1 - p0);
        int m = ind / (p1 - p0);
        int b = (p + p0) % nocc;
        int a = (p + p0) / nocc;
        out[m * nocc * nocc * nvir + (b * nocc + a) * nvir + c] = \
            r[(a * nocc + b - p0) * nvir * nvir + indb[m] * nvir + c];
    ''', 'take0010_s_t')

# nocc, nvir, nocc, nocc = r.shape
# for i in range(n):
#   out[i, :, :, :] = r[:, indb[i], :, :].transpose((2,1,0,3))
take0100_t = cupy.ElementwiseKernel(
    'int64 nocc, int64 nvir, raw int32 indb, raw T r', 'T out', '''
        int d = i % nocc;
        size_t ind = i / nocc;
        int c = ind % nocc;
        ind /= nocc;
        int a = ind % nocc;
        int m = ind / nocc;
        out = r[c * nvir * nocc * nocc + indb[m] * nocc * nocc + a * nocc + d];
    ''', 'take0100_t')

# exta, extb, extc = r.shape
# for i in range(n):
#   out[i, off:off+exta, :, :] = r[:, indb[i], :]
take010_s = cupy.ElementwiseKernel(
    'int64 exta, int64 p0, int64 p1, int64 extb, int64 extc, raw int32 indb,'
    ' raw T r', 'raw T out', '''
        int c = i % extc;
        size_t ind = i / extc;
        int a = ind % (p1 - p0);
        int m = ind / (p1 - p0);
        out[m * exta * extc + (a + p0) * extc + c] = \
            r[a * extb * extc + indb[m] * extc + c];
    ''', 'take010_s')

# exta, extb, extc = r.shape
# for i in range(n):
#   out[i, :] = r[:, indb[i], indc[i]]
take011 = cupy.ElementwiseKernel(
    'int64 exta, int64 extb, int64 extc, raw int32 indb, raw int32 indc, '
    'raw T r',
    'T out', '''
        int a = i % exta;
        int m = i / exta;
        out = r[a * extb * extc + indb[m] * extc + indc[m]];
    ''', 'take011')

# exta, extb, extc = r.shape
# for i in range(n):
#   out[:, off:off+exta][i, :] = r[:, indb[i], indc[i]]
take011_s = cupy.ElementwiseKernel(
    'int64 exta, int64 p0, int64 p1, int64 extb, int64 extc, raw int32 indb,'
    ' raw int32 indc, raw T r', 'raw T out', '''
        int a = i % (p1 - p0);
        int m = i / (p1 - p0);
        out[m * exta + a + p0] = \
            r[a * extb * extc + indb[m] * extc + indc[m]];
    ''', 'take011_s')

# exta, extb, extc = r.shape
# for i in range(n):
#   out[i, :] = r[inda[i], indb[i], :]
take110 = cupy.ElementwiseKernel(
    'int64 exta, int64 extb, int64 extc, raw int32 inda, raw int32 indb, raw T r',
    'T out', '''
    int c = i % extc;
    int m = i / extc;
    out = r[(inda[m] * extb + indb[m]) * extc + c];
    ''', 'take110')

# exta, extb, extc, extd = r.shape
# for i in range(n):
#   out[i, :, :] = r[:, indb[i], indc[i], :]
take0110 = cupy.ElementwiseKernel(
    'int64 exta, int64 extb, int64 extc, int64 extd, raw int32 indb, '
    'raw int32 indc, raw T r',
    'T out', '''
        int d = i % extd;
        size_t ind = i / extd;
        int a = ind % exta;
        int m = ind / exta;
        out = r[a * extb * extc * extd + indb[m] * extc * extd
                + indc[m]* extd + d];
    ''', 'take0110')

# exta, extb, extc, extd = r.shape
# for i in range(n):
#   out[i, :, :] = r[:, indb[i], :, indd[i]]
take0101 = cupy.ElementwiseKernel(
    'int64 exta, int64 extb, int64 extc, int64 extd, raw int32 indb, '
    'raw int32 indd, raw T r',
    'T out', '''
        int c = i % extc;
        size_t ind = i / extc;
        int a = ind % exta;
        int m = ind / exta;
        out = r[a * extb * extc * extd + indb[m] * extc * extd
                + c * extd + indd[m]];
    ''', 'take0101')

# exta, extb, extc, extd = r.shape
# for i in range(n):
#   out[i, :, :] = r[inda[i], :, indc[i], :]
take1010 = cupy.ElementwiseKernel(
    'int64 exta, int64 extb, int64 extc, int64 extd, raw int32 inda, raw int32 indc, raw T r',
    'T out', '''
    int d = i % extd;
    size_t temp = i / extd;
    int b = temp % extb; 
    size_t m = temp / extb;   
    
    int a_target = inda[m];
    int c_target = indc[m];
    
    size_t r_idx = ((((size_t)a_target * extb) + b) * extc + c_target) * extd + d;
    out = r[r_idx];
    ''', 'take1010')

def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    return ccsd_lambda.kernel(mycc, eris, t1, t2, l1, l2, max_cycle, tol,
                              verbose, make_intermediates, update_lambda)

def make_intermediates(mycc, t1, t2, eris):
    imds = ccsd_lambda.make_intermediates(mycc, t1, t2, eris)
    nocc, nvir = t1.shape
    nabc = nvir * (nvir + 1) * (nvir + 2) // 6
    pool = mycc.pool
    ovvv = eris.ovvv
    ovoo = eris.ovoo
    ovov = eris.ovov
    fov = cupy.asarray((eris.fock)[nocc:,:nocc])
    mo_e = pool.asarray(eris.mo_energy)
    e_occ, e_vir = mo_e[:nocc], mo_e[nocc:]
     
    def take_ovov(eri_ovov,ind1,ind2,lov_1,lov_2):
        if ovov.l2 is None: # turn off density fitting
            take0101(nocc, nvir, nocc, nvir, ind1, ind2, ovov, eri_ovov)
        else:
            naux = ovov.l1.shape[0]
            take01(naux * nocc, nvir, ind1, ovov.l1, lov_1)
            take01(naux * nocc, nvir, ind2, ovov.l1, lov_2)
            lib.contraction('nli',lov_1,'nlj',lov_2,'nij',eri_ovov)

    def take_ovvv(eri_ovvv,ind1,ind2,lov,lvv): # for ovvv, take0110 = take0101 due to the symmetry (ia|bc) = (ia|cb)
        if ovvv.l2 is None: # turn off density fitting
            take0110(nocc, nvir, nvir, nvir, ind1, ind2, ovvv, eri_ovvv)
        else: # ovvv density fitting is wrong!
            naux = ovvv.l1.shape[0]
            take01(naux * nocc, nvir, ind1, ovvv.l1, lov)
            take010(naux, nvir, nvir, ind2, ovvv.l2, lvv) # ovvv.l2 might be in CPU?
            lib.contraction('nlo', lov, 'nlv', lvv, 'nov', eri_ovvv)
    
    def take_ovoo(eri_ovoo,ind1,lov,lpq):
        if ovoo.l2 is None: # turn off density fitting
            take010(nocc,nvir,nocc*nocc,ind1,ovoo,eri_ovoo)
        else:
            naux = ovoo.l1.shape[0]
            #lov = tmpbuf.empty((n, naux, nocc), 'f8')
            #lpq = tmpbuf.empty((naux,nocc,nocc),'f8')
            take01(naux * nocc, nvir, ind1, ovoo.l1, lov)
            lpq = ovoo.l2.ascupy(buf=lpq)
            lib.contraction('nlo', lov, 'lpq', lpq, 'nopq', eri_ovoo)
    
    
    if t2.dev == 0: # t2 can be stored in GPU
        def take010_t2(out, ind, t2, buf, isTrans=False):
            if isTrans:
                take0010_t(nocc, nvir, ind, t2, out)
            else:
                take010(nocc * nocc, nvir, nvir, ind, t2, out)
        def take011_t2(out, ind1, ind2, t2, buf):
            take011(nocc * nocc, nvir, nvir, ind1, ind2, t2, out)  
    else:
        # t2 can't be stored in GPU
        def take010_t2(out, ind, t2, buf, isTrans=False):
            blk = min(nocc * nocc, int(buf.bufsize / 8 / nvir**2))
            n = len(ind)
            t2 = t2.reshape(nocc**2, nvir, nvir)
            buf.tag('t2')
            buf1 = buf.left()
            for p0, p1 in prange(0, nocc * nocc, blk):
                t2p = t2[p0:p1].ascupy(buf=buf1)
                if isTrans:
                    take0010_s_t(nocc, nvir, p0, p1, ind, t2p, out,
                                 size=n * (p1 - p0) * nvir)
                else:
                    take010_s(nocc * nocc, p0, p1, nvir, nvir, ind, t2p, out,
                              size=n * (p1 - p0) * nvir)
            buf.untag('t2')
        def take011_t2(out, ind1, ind2, t2, buf):
            n = len(ind1)
            t2 = t2.reshape(nocc**2, nvir, nvir)
            blk = min(nocc * nocc, int(buf.bufsize / 8 / nvir**2))
            buf.tag('t2')
            buf1 = buf.left()
            for p0, p1 in prange(0, nocc * nocc, blk):
                t2p = t2[p0:p1].ascupy(buf=buf1)
                take011_s(nocc * nocc, p0, p1, nvir, nvir, ind1,
                          ind2, t2p, out, size=n * (p1 - p0))
            buf.untag('t2')
    
    pool.status['l1_t'] = 0 # store in gpu
    pool.status['l2_t'] = 1 # store in cpu
    pool.status['j'] = 1 # store in cpu
    pool.status['k'] = 1 # store in cpu
    imds.l1_t = pool.new('l1_t', (nocc,nvir), 'f8')
    imds.l2_t = pool.new('l2_t', (nocc,nocc,nvir,nvir), 'f8')
    joovv = pool.new('j', (nocc,nocc,nvir,nvir), 'f8')
    koovv = pool.new('k', (nocc,nocc,nvir,nvir), 'f8')
    imds.l1_t[:] = 0.0
    imds.l2_t[:] = 0.0  
    joovv[:] = 0.0
    koovv[:] = 0.0

    memory = pool.free_memory
    print("Free memory before ccsd_t_lambda: ", memory)
    print(f"nocc = {nocc}, nvir = {nvir}")
    unit_occ = 3 * nocc * nocc * nocc + nocc * nocc * nvir + nocc * (nocc + nvir) + 3
    if eris.ovvv.l2 is not None: # density fitting is turned on
        naux = eris.ovvv.l1.shape[0]
        unit_occ += naux * nocc * 2
        unit_occ += naux * nocc * nocc + naux * nvir
    blksize = min(nabc, int(memory / 8 / unit_occ))

    # allocate memory for intermediate tensors 
    buf = lib.ArrayBuffer(pool.empty((blksize * unit_occ + 10 * 1024), 'f8'))
    a = buf.empty((blksize), 'i4')
    b = buf.empty((blksize), 'i4')
    c = buf.empty((blksize), 'i4')
    buf1 = buf.empty((blksize, nocc, nocc, nocc), 'f8')
    buf2 = buf.empty((blksize, nocc, nocc, nocc), 'f8')
    bufleft = buf.left()
    cpu_buf1 = lib.empty((blksize, nocc, nocc, nvir), 'f8',type=1) # pinned memory
    cpu_buf2 = lib.empty((blksize, nocc, nocc), 'f8',type=1) # pinned memory
    cpu_buf1_asy = lib.empty((blksize, nocc, nocc, nvir), 'f8',type=1) # pinned memory
    cpu_buf2_asy = lib.empty((blksize, nocc, nocc), 'f8',type=1) # pinned memory
    # p6 permutation for slicing (a,b,c)
    inds = numpy.asarray([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1],[2, 1, 0]])
    modes = numpy.asarray(["nijk", "nikj", "njik", "nkij", "njki", "nkji"])

    # evaluate lambda vectors by slicing (a,b,c)
    for p0, p1 in list(prange(0, nabc, blksize)): # loop for all blks
        n = p1 - p0
        gen_abc(p0, a[:n], b[:n], c[:n])
        abc = [a[:n], b[:n], c[:n]]
        def add_w(abc,_bufw):
            w = None
            n = len(abc[0])
            tmpbuf = lib.ArrayBuffer(bufleft)
            tmpbuf.tag()
            perm = [3, 5, 1, 4, 0, 2]
            _inds = inds[perm]
            _modes = modes[perm]
            # w = lib.contraction('iafb',eris_ovvv,'kjcf',t2, 'ijkabc')
            for i in range(6): # p6 permutation --> loop for all 6 modes
                a, b, c = [abc[j] for j in _inds[i]] 
                mode = _modes[i]
                tmpbuf.loadtag()
                eri_ovvv = tmpbuf.empty((n, nocc, nvir), 'f8')
                lov = lvv = None
                if ovvv.l2 is not None: # turn on density fitting
                    naux = ovvv.l1.shape[0]
                    lov = tmpbuf.empty((n, naux, nocc), 'f8')
                    lvv = tmpbuf.empty((n, naux, nvir), 'f8')
                take_ovvv(eri_ovvv, a, b, lov, lvv)
                lov = lvv = None

                inc = 'njkf' if i == 2 or i == 3 else 'nkjf'
                if i % 2 == 0:
                    t2s = tmpbuf.empty((n, nocc, nocc, nvir), 'f8')
                    take010_t2(t2s, c, t2, tmpbuf,
                               isTrans=True if i == 2 else False)
                    tmpbuf.tag('t2')
                tmpbuf.loadtag('t2')

                if w is None:
                    w = lib.contraction('nif', eri_ovvv, inc, t2s, mode, buf=_bufw)
                else:
                    w = lib.contraction('nif', eri_ovvv, inc, t2s, mode, w, beta=1.0)
        
            # w -= lib.contraction('iajm',eris_ovoo,'mkbc',t2, 'ijkabc')
            for i in range(6): # p6 permutation --> loop for all 6 modes
                a, b, c = [abc[j] for j in inds[i]]
                mode = modes[i]
                tmpbuf.loadtag()
                if i not in [1, 3]:
                    eri_ovoo = tmpbuf.empty((n, nocc, nocc, nocc), 'f8')
                    if ovoo.l2 is None:
                        if i in [0, 5]:
                            take010(nocc, nvir, nocc * nocc, a, ovoo, eri_ovoo)
                        else:
                            take0100_t(nocc, nvir, a, ovoo, eri_ovoo)
                    else:
                        naux = ovoo.l1.shape[0]
                        lov = tmpbuf.empty((n, naux, nocc), 'f8')
                        take01(naux * nocc, nvir, a, ovoo.l1, lov)
                        lib.contraction('nlo', lov, 'lpq',
                                        ovoo.l2, 'nopq' if i in [0, 5] else 'npoq', eri_ovoo)
                    tmpbuf.tag('ovoo')
                tmpbuf.loadtag('ovoo')
                t2ss = tmpbuf.empty((n, nocc, nocc), 'f8')
                take011_t2(t2ss, c, b, t2, tmpbuf)
                inda = 'nijm' if i in [0, 1, 5] else 'njim'
                lib.contraction(inda, eri_ovoo, 'nkm', t2ss, mode, w,
                                beta=1.0, alpha=-1.0)
            return w
        
        def r3(input, output):
            '''
            for Restricted-CCSD only
            Original 6-D expression:
            r3(w) = 4 * w + 1 * w.transpose(0,1,2,4,5,3) + 1 * w.transpose(0,1,2,5,3,4)
                          - 2 * w.transpose(0,1,2,5,4,3) - 2 * w.transpose(0,1,2,3,5,4)
                          - 2 * w.transpose(0,1,2,4,3,5)
            Sliced expression:
            output = 4 * input + input.transpose(1,2,0) + input.transpose(2,0,1)
                   - 2 * input.transpose(2,1,0) - 2 * input.transpose(0,2,1)
                   - 2 * input.transpose(1,0,2).
            Boosted by cuTENSOR.
            '''
            lib.elementwise_binary('nkij', input, 'nijk', output, gamma=4.0)
            lib.elementwise_binary('njki', input, 'nijk', output, gamma=1.0)
            lib.elementwise_binary('nkji', input, 'nijk', output, alpha=-2.0, gamma=1.0)
            lib.elementwise_binary('nikj', input, 'nijk', output, alpha=-2.0, gamma=1.0)
            lib.elementwise_binary('njik', input, 'nijk', output, alpha=-2.0, gamma=1.0)
            return output
        
        bufw = buf1[:n] 
        r = buf2[:n]
        w = add_w(abc,bufw)
        div_d3(nocc, *abc, e_occ, e_vir, w)
        r[:] = w
        r3(w,r) 
        tmpbuf = lib.ArrayBuffer(bufleft)

        # calculate l1_t via r
        tmpbuf.tag()
        tmp_l1_t = tmpbuf.empty((n, nocc), 'f8')
        eri_ovov = tmpbuf.empty((n, nocc, nocc), 'f8')
        
        # l1_t = numpy.einsum('jbkc,ijkabc->ia', eris_ovov, r6(w)) / eia * .5
        lov_1 = lov_2 = None
        if ovov.l2 is not None:
            naux = ovov.l1.shape[0]
            lov_1 = tmpbuf.empty((n, naux, nocc), 'f8')
            lov_2 = tmpbuf.empty((n, naux, nocc), 'f8')

        sym_multiply(nocc, *abc, r)
        take_ovov(eri_ovov,b,c,lov_1,lov_2)
        lib.contraction('njk',eri_ovov,'nijk',r,'ni',tmp_l1_t,alpha=0.5)
        cupyx.scatter_add(imds.l1_t, (slice(None), a[:n]), tmp_l1_t.T)
        take_ovov(eri_ovov,a,c,lov_1,lov_2)
        lib.contraction('nik',eri_ovov,'nijk',r,'nj',tmp_l1_t,alpha=0.5)
        cupyx.scatter_add(imds.l1_t, (slice(None), b[:n]), tmp_l1_t.T)
        take_ovov(eri_ovov,a,b,lov_1,lov_2)
        lib.contraction('nij',eri_ovov,'nijk',r,'nk',tmp_l1_t,alpha=0.5)
        cupyx.scatter_add(imds.l1_t, (slice(None), c[:n]), tmp_l1_t.T)
        lov_1 = lov_2 = None
        tmpbuf.loadtag()    # free tmp_l1_t and eri_ovov
        tmpbuf = None

        # calculate l2 via m and r'
        def as_r6(input,output,mode0='nijk',mode1='nkji',mode2='nikj'):
            # When making derivative over t2, r6 should be called on the 6-index
            # tensor. It gives the equation for lambda2, but not corresponding to
            # the lambda equation used by RCCSD-lambda code.  A transformation was
            # applied in RCCSD-lambda equation  F(lambda)_{ijab} = 0:
            #       2/3 * # F(lambda)_{ijab} + 1/3 * F(lambda)_{jiab} = 0
            # Combining this transformation with r6 operation, leads to the
            # transformation code below
            # return m * 2 - m.transpose(0,1,2,5,4,3) - m.transpose(0,1,2,3,5,4)
            ### Important things in our tensor slicing code:
            # as_r6 will break the 6-fold permutation symmetry, 
            # so we have to manually apply the upper-triangle symmetry.
            lib.elementwise_binary(mode0, input, 'nijk', output, alpha=2.0)
            lib.elementwise_trinary(mode1, input, mode2, input, 'nijk', output, gamma=1.0, alpha=-1.0, beta=-1.0)
            return output
        
        @numba.njit(parallel=True, fastmath=True)
        def fast_scatter_add_j1(target, idx_c, update):
            n, nocc, _, nvir = update.shape
            for i in numba.prange(nocc):
                for k in range(n):
                    c = idx_c[k]
                    for jj in range(nocc):
                        for v in range(nvir):
                            target[i, jj, c, v] += update[k, i, jj, v]

        @numba.njit(parallel=True, fastmath=True)
        def fast_scatter_add_k(target, idx_c1, idx_c2, update):
            n, nocc, _ = update.shape
            for i in numba.prange(nocc):
                for k in range(n):
                    c1 = idx_c1[k]
                    c2 = idx_c2[k]
                    for jj in range(nocc):
                        target[i, jj, c1, c2] += update[k, i, jj]   
        
        # In the next section, r = as_r6(w), is different from r tensor in the l1_t equation.  
        def add_k(abc,k,cpu_bufk):
            a,b,c = abc[0],abc[1],abc[2]
            cpu_a, cpu_b, cpu_c = cupy.asnumpy(a), cupy.asnumpy(b), cupy.asnumpy(c)
            tmpbuf = lib.ArrayBuffer(bufleft)
            bufk = tmpbuf.empty((n, nocc, nocc), 'f8')
            fs = tmpbuf.empty((n, nocc), 'f8')
            
            take01(nocc,nvir,c,fov,fs)
            as_r6(w,r,'nijk','nkji','nikj') # (v1,v2,v3) = (A,B,C); r = 2W(A,B,C) - W(C,B,A) - W(A,C,B)
            lib.contraction('nk',fs,'nijk',r,'nij',bufk)
            bufk.get(out=cpu_bufk,blocking=True)
            fast_scatter_add_k(k, cpu_a, cpu_b, cpu_bufk)
            as_r6(w,r,'njik','njki','nkij') # (v1,v2,v3) = (B,A,C), r = 2W(B,A,C) - W(C,A,B) - W(B,C,A)
            lib.contraction('nk',fs,'nijk',r,'nij',bufk)
            bufk.get(out=cpu_bufk,blocking=True)
            fast_scatter_add_k(k, cpu_b, cpu_a, cpu_bufk)

            take01(nocc,nvir,b,fov,fs)
            as_r6(w,r,'nikj','nkij','nijk') # (v1,v2,v3) = (A,C,B), r = 2W(A,C,B) - W(B,C,A) - W(A,B,C)
            lib.contraction('nk',fs,'nijk',r,'nij',bufk)
            bufk.get(out=cpu_bufk,blocking=True)
            fast_scatter_add_k(k, cpu_a, cpu_c, cpu_bufk)
            as_r6(w,r,'njki','njik','nkji') # (v1,v2,v3) = (C,A,B), r = 2W(C,A,B) - W(B,A,C) - W(C,B,A)
            lib.contraction('nk',fs,'nijk',r,'nij',bufk)
            bufk.get(out=cpu_bufk,blocking=True)
            fast_scatter_add_k(k, cpu_c, cpu_a, cpu_bufk)
            
            take01(nocc,nvir,a,fov,fs)
            as_r6(w,r,'nkij','nikj','njik') # (v1,v2,v3) = (B,C,A), r = 2W(B,C,A) - W(A,C,B) - W(B,A,C)
            lib.contraction('nk',fs,'nijk',r,'nij',bufk)
            bufk.get(out=cpu_bufk,blocking=True)
            fast_scatter_add_k(k, cpu_b, cpu_c, cpu_bufk)
            as_r6(w,r,'nkji','nijk','njki') # (v1,v2,v3) = (C,B,A), r = 2W(C,B,A) - W(A,B,C) - W(C,A,B))
            lib.contraction('nk',fs,'nijk',r,'nij',bufk)
            bufk.get(out=cpu_bufk,blocking=True)
            fast_scatter_add_k(k, cpu_c, cpu_b, cpu_bufk)
            return
        
        def add_z(abc,_bufv):
            # build v using _bufv, then add w to it to get z.
            v = None
            tmpbuf = lib.ArrayBuffer(bufleft)
            tmpbuf.tag('z')
            perm = [0, 1, 2, 3, 4, 5]
            _inds = inds[perm]
            _modes = modes[perm]
            # v = lib.contraction('iajb',eris_ovov.conj(),'kc',t1, 'ijkabc')
            for i in range(6): # p6 permutation --> loop for all 6 modes
                a, b, c = [abc[j] for j in _inds[i]] 
                mode = _modes[i]
                tmpbuf.loadtag('z')
                eri_ovov = tmpbuf.empty((n, nocc, nocc), 'f8')
                t1s = tmpbuf.empty((n, nocc), 'f8')
                lov_1 = lov_2 = None
                if ovov.l2 is not None:
                    naux = ovov.l1.shape[0]
                    lov_1 = tmpbuf.empty((n, naux, nocc), 'f8')
                    lov_2 = tmpbuf.empty((n, naux, nocc), 'f8')
                take_ovov(eri_ovov, a, b, lov_1,lov_2)
                lov_1 = lov_2 = None
                take01(nocc,nvir,c,t1,t1s)
                # contraction
                if v is None:
                    v = lib.contraction('nij',eri_ovov, 'nk',t1s,mode,buf=_bufv)
                else:
                    lib.contraction('nij',eri_ovov,'nk',t1s,mode,v,beta=1.0)

            # v += lib.contraction('ck',eris.fock[nocc:,:nocc],'ijab',t2, 'ijkabc')
            for i in range(6): # p6 permutation --> loop for all 6 modes
                a, b, c = [abc[j] for j in _inds[i]] 
                mode = _modes[i]
                tmpbuf.loadtag('z')
                fs = tmpbuf.empty((n, nocc), 'f8')
                t2s = tmpbuf.empty((n, nocc, nocc), 'f8')
                take01(nocc,nvir,c,fov,fs) # slice fock, (kc) -> (nk)
                take011_t2(t2s,a,b,t2,tmpbuf) # slice t2, (ijab) -> (nij)
                lib.contraction('nk',fs,'nij',t2s,mode,v,beta=1.0) # contraction
            
            div_d3(nocc, *abc, e_occ, e_vir, v)
            sym_divide(nocc,*abc,v)
            lib.elementwise_binary('nijk', w, 'nijk', v, alpha=2.0, gamma=0.5) # z = 2w + 0.5v
            tmpbuf.untag('z')
            return v
        
        sym_divide(nocc,*abc,w) 
        add_k(abc,koovv,cpu_buf2[:n]) # Buffer allocation now: buf1 -> w; buf2 -> r
        del r
        bufz = buf2[:n]
        z = add_z(abc,bufz)   # Buffer allocation now: buf1 -> w; buf2 -> z
        del w, bufw
        m = buf1[:n]
        # Buffer allocation now: buf1 -> m; buf2 -> z

        def add_j(abc,j,cpu_bufj1,cpu_bufj2,cpu_bufj1_asy,cpu_bufj2_asy):
            a,b,c = abc[0],abc[1],abc[2]
            n = len(a)
            cpu_a = cupy.asnumpy(a)
            cpu_b = cupy.asnumpy(b)
            cpu_c = cupy.asnumpy(c)
            tmpbuf = lib.ArrayBuffer(bufleft)
            bufj1 = tmpbuf.empty((n, nocc, nocc, nvir), 'f8')
            bufj2 = tmpbuf.empty((n, nocc, nocc), 'f8')
            eri_ovvv = tmpbuf.empty((n, nocc, nvir), 'f8')
            eri_ovoo = tmpbuf.empty((n, nocc, nocc, nocc), 'f8')
            lov_ovoo = lpq = lov = lvv = None
            if ovoo.l2 is not None:
                naux = ovoo.l1.shape[0]
                lov_ovoo = tmpbuf.empty((n, naux, nocc), 'f8')
                lpq = tmpbuf.empty((naux, nocc, nocc),'f8')
            if ovvv.l2 is not None: # turn on density fitting
                naux = ovvv.l1.shape[0]
                lov = tmpbuf.empty((n, naux, nocc), 'f8')
                lvv = tmpbuf.empty((n, naux, nvir), 'f8')
            
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            task_tracker = {} 
            def dispatch_j1(cpu_buf, idx_c):
                if id(cpu_buf) in task_tracker:
                   task_tracker[id(cpu_buf)].result() 
                bufj1.get(out=cpu_buf, blocking=True)
                task_tracker[id(cpu_buf)] = executor.submit(fast_scatter_add_j1, j, idx_c, cpu_buf)
            def dispatch_k(cpu_buf, idx_c1, idx_c2):
                if id(cpu_buf) in task_tracker:
                   task_tracker[id(cpu_buf)].result()
                bufj2.get(out=cpu_buf, blocking=True)
                task_tracker[id(cpu_buf)] = executor.submit(fast_scatter_add_k, j, idx_c1, idx_c2, cpu_buf)

            # (a_n,b_n,c_n) = (v1,e,f)(ijk)
            as_r6(z,m,'nijk','nkji','nikj')
            take_ovoo(eri_ovoo,c,lov_ovoo,lpq)
            take_ovvv(eri_ovvv,c,b,lov,lvv)
            lib.contraction('nkv',eri_ovvv,'nijk',m,'nijv',bufj1)
            dispatch_j1(cpu_bufj1, cpu_a)
            lib.contraction('nkjl',eri_ovoo,'nijk',m,'nil',bufj2,alpha=-1.0)
            dispatch_k(cpu_bufj2, cpu_a, cpu_b)
            
            # (a_n,b_n,c_n) = (e,v1,f)(jik)
            as_r6(z,m,'njik','njki','nkij')
            take_ovvv(eri_ovvv,c,a,lov,lvv)
            lib.contraction('nkv',eri_ovvv,'nijk',m,'nijv',bufj1)
            dispatch_j1(cpu_bufj1_asy, cpu_b)
            lib.contraction('nkjl',eri_ovoo,'nijk',m,'nil',bufj2,alpha=-1.0)
            dispatch_k(cpu_bufj2_asy, cpu_b, cpu_a)

            # (a_n,b_n,c_n) = (v1,f,e)(ikj)
            as_r6(z,m,'nikj','nkij','nijk')
            take_ovoo(eri_ovoo,b,lov_ovoo,lpq)
            take_ovvv(eri_ovvv,b,c,lov,lvv)
            lib.contraction('nkv',eri_ovvv,'nijk',m,'nijv',bufj1)
            dispatch_j1(cpu_bufj1, cpu_a)
            lib.contraction('nkjl',eri_ovoo,'nijk',m,'nil',bufj2,alpha=-1.0)
            dispatch_k(cpu_bufj2, cpu_a, cpu_c)

            # (a_n,b_n,c_n) = (e,f,v1)(jki)
            as_r6(z,m,'njki','njik','nkji')
            take_ovvv(eri_ovvv,b,a,lov,lvv)
            lib.contraction('nkv',eri_ovvv,'nijk',m,'nijv',bufj1)
            dispatch_j1(cpu_bufj1_asy, cpu_c)
            lib.contraction('nkjl',eri_ovoo,'nijk',m,'nil',bufj2,alpha=-1.0)
            dispatch_k(cpu_bufj2_asy, cpu_c, cpu_a)

            # (a_n,b_n,c_n) = (f,v1,e)(kij)
            as_r6(z,m,'nkij','nikj','njik')
            take_ovoo(eri_ovoo,a,lov_ovoo,lpq)
            take_ovvv(eri_ovvv,a,c,lov,lvv)
            lib.contraction('nkv',eri_ovvv,'nijk',m,'nijv',bufj1)
            dispatch_j1(cpu_bufj1, cpu_b)
            lib.contraction('nkjl',eri_ovoo,'nijk',m,'nil',bufj2,alpha=-1.0)
            dispatch_k(cpu_bufj2, cpu_b, cpu_c)

            # (a_n,b_n,c_n) = (f,e,v1)(kji)
            as_r6(z,m,'nkji','nijk','njki')
            take_ovvv(eri_ovvv,a,b,lov,lvv)
            lib.contraction('nkv',eri_ovvv,'nijk',m,'nijv',bufj1)
            dispatch_j1(cpu_bufj1_asy, cpu_c)
            lib.contraction('nkjl',eri_ovoo,'nijk',m,'nil',bufj2,alpha=-1.0)
            dispatch_k(cpu_bufj2_asy, cpu_c, cpu_b)

            for future in task_tracker.values():
                future.result()
            executor.shutdown()
            lov_ovoo = lpq = lov = lvv = None
            return
        
        add_j(abc,joovv,cpu_buf1[:n],cpu_buf2[:n],cpu_buf1_asy[:n],cpu_buf2_asy[:n])
        m = z = bufz = None

    # calculate l2_t
    joovv = joovv + joovv.transpose(1,0,3,2) # performed on cpu
    imds.l2_t = joovv + koovv
    joovv = koovv = None

    # div_d1 for l1_t (in gpu)
    eia = cupy.asarray(e_occ)[:, None] - cupy.asarray(e_vir)[None, :]
    imds.l1_t /= eia
    eia = None
    
    # div_d2 for l2_t (in cpu)
    e_occ = cupy.asnumpy(e_occ)
    e_vir = cupy.asnumpy(e_vir)
    for i in numba.prange(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    e_ijab = e_occ[i] + e_occ[j] - e_vir[a] - e_vir[b]                   
                    imds.l2_t[i, j, a, b] /= e_ijab

    buf = bufleft = tmpbuf = tmp_l1_t = eri_ovov = buf1 = buf2 = None
    abc = a = b = c = None
    cpu_buf1 = cpu_buf2 = None
    lib.free_all_blocks()
    return imds

def update_lambda(mycc, t1, t2, l1, l2, eris=None, imds=None): 
    if eris is None: eris = mycc.ao2mo()
    if imds is None: imds = make_intermediates(mycc, t1, t2, eris)
    l1, l2 = ccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
    l1 += imds.l1_t
    l2 += imds.l2_t
    lib.free_all_blocks()
    return l1, l2