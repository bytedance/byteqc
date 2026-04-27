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

from byteqc.cucc import ccsd_rdm
import numpy
import cupy
import cupyx
from pyscf.lib import logger, prange
from pyscf.cc.ccsd import BLKMIN
from byteqc import lib

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
div_d3_abc = cupy.ElementwiseKernel(
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
    ''', 'div_d3_abc')

div_d3_ijk = cupy.ElementwiseKernel(
    'int64 nvir, raw int32 indi, raw int32 indj, raw int32 indk, raw T eocc, '
    'raw T evir', 'T out',
    '''
        int C = i % nvir;
        size_t temp = i / nvir;
        int B = temp % nvir;
        temp /= nvir;
        int A = temp % nvir;
        
        size_t n_idx = temp / nvir;
        int I = indi[n_idx];
        int J = indj[n_idx];
        int K = indk[n_idx];

        out /= (eocc[I] + eocc[J] + eocc[K] - evir[A] - evir[B] - evir[C]);
    ''', 'div_d3_ijk')

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

def _gamma1_intermediates(mycc, t1, t2, l1, l2, eris=None):
    doo, dov, dvo, dvv = ccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)
    if eris is None: eris = mycc.ao2mo()

    nocc, nvir = t1.shape
    nmax = max(nocc, nvir)
    nabc = nvir * (nvir + 1) * (nvir + 2) // 6
    pool = mycc.pool
    ovvv = eris.ovvv
    ovoo = eris.ovoo
    ovov = eris.ovov
    fov = cupy.asarray((eris.fock)[nocc:,:nocc])
    mo_e = pool.asarray(eris.mo_energy)
    e_occ, e_vir = mo_e[:nocc], mo_e[nocc:]

    # evaluate goo by slicing (a,b,c)
    goo = pool.empty((nocc,nocc), 'f8')
    gvv = pool.empty((nvir,nvir), 'f8')
    goo[:] = 0.0
    gvv[:] = 0.0

    if t2.dev == 0: # t2 can be stored in GPU
        def take010_t2(out, ind, t2, buf, isTrans=False):
            if isTrans:
                take0010_t(nocc, nvir, ind, t2, out)
            else:
                take010(nocc * nocc, nvir, nvir, ind, t2, out)
        def take011_t2(out, ind1, ind2, t2, buf):
            take011(nocc * nocc, nvir, nvir, ind1, ind2, t2, out)
        
        def take110_t2(out, ind1, ind2, t2, buf):
            take110(nocc, nocc, nvir * nvir, ind1, ind2, t2, out)
        
        def take010_t2v(out, ind, t2, buf):
            take010(nocc, nocc, nvir * nvir, ind, t2, out)
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
        def take110_t2(out, ind1, ind2, t2, buf):
            i_cpu = cupy.asnumpy(ind1)
            j_cpu = cupy.asnumpy(ind2)
            ## TODO: using pre-memory allcation to speed up
            # lib.empty((n, nvir, nvir), type=1) 
            t2_sliced_cpu = t2[i_cpu, j_cpu, :, :]
            out[:] = cupy.asarray(t2_sliced_cpu)
        def take010_t2v(out, ind, t2, buf):
            ind_cpu = cupy.asnumpy(ind)
            t2_sliced_cpu = t2[:, ind_cpu, :, :]
            out[:] = cupy.asarray(t2_sliced_cpu)

    memory = pool.free_memory
    print("Free memory before ccsd_t_rdm: ",memory)
    unit_occ = 3 * nocc * nocc * nocc + nmax * nocc * nvir + nmax * nocc + 3
    if eris.ovvv.l2 is not None: # density fitting is turned on
        naux = eris.ovvv.l1.shape[0]
        unit_occ += naux * nocc * 2
    blksize = min(nabc, int(memory / 8 / unit_occ))
    # allocate memory for intermediate tensors 
    buf = lib.ArrayBuffer(pool.empty((blksize * unit_occ + 10 * 1024), 'f8'))
    a = buf.empty((blksize), 'i4')
    b = buf.empty((blksize), 'i4')
    c = buf.empty((blksize), 'i4')
    bufw = buf.empty((blksize, nocc, nocc, nocc), 'f8')
    bufz = buf.empty((blksize, nocc, nocc, nocc), 'f8')
    bufleft = buf.left()
    # p6 permutation for slicing (a,b,c) 
    inds = numpy.asarray([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1],[2, 1, 0]])
    modes = numpy.asarray(["nijk", "nikj", "njik", "nkij", "njki", "nkji"])
    
    # evaluate goo by slicing (a,b,c)
    for p0, p1 in list(prange(0, nabc, blksize)): # loop for all blks
        n = p1 - p0
        gen_abc(p0, a[:n], b[:n], c[:n])
        abc = [a[:n], b[:n], c[:n]] 
        def add_w(abc):
            w = None
            n = len(abc[0])
            tmpbuf = lib.ArrayBuffer(bufleft)
            tmpbuf.tag()
            perm = [3, 5, 1, 4, 0, 2]
            _inds = inds[perm]
            _modes = modes[perm]

            # w = lib.contraction('iafb',eris_ovvv,'kjcf',t2, 'ijkabc')
            for i in range(6): # p6 permutation --> loop for all 6 modes
                a, b, c = [abc[j] for j in _inds[i]] # a, b, c is the index ARRAY
                mode = _modes[i]
                tmpbuf.loadtag()

                # determine the slice of eris_ovvv
                eri_ovvv = tmpbuf.empty((n, nocc, nvir), 'f8')
                if ovvv.l2 is None: 
                    take0110(nocc, nvir, nvir, nvir, a, b, ovvv, eri_ovvv)
                else:
                    naux = ovvv.l1.shape[0]
                    lov = tmpbuf.empty((n, naux, nocc), 'f8')
                    lvv = tmpbuf.empty((n, naux, nvir), 'f8')
                    take01(naux * nocc, nvir, a, ovvv.l1, lov)
                    take010(naux, nvir, nvir, b, ovvv.l2, lvv)
                    lib.contraction('nlo', lov, 'nlv', lvv, 'nov', eri_ovvv)

                # determine the slice of t2 amplitude
                '''
                For i in range(6) the contraction time t_i is in order
                   t_0<t_5<t_3<t_1<t_4<t_2
                Expensive t2s is read only if i % 2 == 0 to reduce I/O time.
                Switching two nocc dims of t 2s for i = 2, 3 can save time
                After switching the total time is
                   t_0+t_1+t_4+t_1+t_4+t_5
                '''
                inc = 'njkf' if i == 2 or i == 3 else 'nkjf'
                if i % 2 == 0:
                    t2s = tmpbuf.empty((n, nocc, nocc, nvir), 'f8')
                    take010_t2(t2s, c, t2, tmpbuf,
                               isTrans=True if i == 2 else False)
                    tmpbuf.tag('t2')
                tmpbuf.loadtag('t2')
            
                # conduct contraction between two slices
                if w is None:
                    w = lib.contraction('nif', eri_ovvv, inc, t2s, mode, buf=bufw)
                else:
                    w = lib.contraction('nif', eri_ovvv, inc, t2s, mode, w, beta=1.0)
        
            # w -= lib.contraction('iajm',eris_ovoo,'mkbc',t2, 'ijkabc')
            for i in range(6): # p6 permutation --> loop for all 6 modes
                a, b, c = [abc[j] for j in inds[i]]
                mode = modes[i]
                tmpbuf.loadtag()
                # For i in range(6) the contraction time t_i is in order
                #   t_0<t_1<t_5<t_2<t_4<t_3
                # Readind eri_ovoo is cheap, only reuse it for i = 1, 3
                # Switching two nocc dims of t2s for i = 2, 4, 3 can save time.
                # After switching the total time is
                #   t_0+t_1+t_0+t_5+t_1+t_5
                if i not in [1, 3]:
                    eri_ovoo = tmpbuf.empty((n, nocc, nocc, nocc), 'f8')
                    if ovoo.l2 is None:
                        if i in [0, 5]:
                            take010(nocc, nvir, nocc * nocc,
                                    a, ovoo, eri_ovoo)
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
        
        def add_z(abc,w): # build tensor v, and then add it to w to obtain z
            v = None
            tmpbuf = lib.ArrayBuffer(bufleft)
            tmpbuf.tag()
            # perm = [3, 5, 1, 4, 0, 2]
            perm = [0, 1, 2, 3, 4, 5]
            _inds = inds[perm]
            _modes = modes[perm]

            # v = lib.contraction('iajb',eris_ovov.conj(),'kc',t1, 'ijkabc')
            for i in range(6): # p6 permutation --> loop for all 6 modes
                a, b, c = [abc[j] for j in _inds[i]] 
                mode = _modes[i]
                tmpbuf.loadtag()
                eri_ovov = tmpbuf.empty((n, nocc, nocc), 'f8')
                t1s = tmpbuf.empty((n, nocc), 'f8')

                # slice ovov
                if ovov.l2 is None: # turn off density fitting
                    take0101(nocc, nvir, nocc, nvir, a, b, ovov, eri_ovov)
                else: # turn on density fitting
                    naux = ovov.l1.shape[0]
                    lov_a = tmpbuf.empty((n, naux, nocc), 'f8')
                    lov_b = tmpbuf.empty((n, naux, nocc), 'f8')
                    take01(naux * nocc, nvir, a, ovov.l1, lov_a)
                    take01(naux * nocc, nvir, b, ovov.l1, lov_b)
                    lib.contraction('nli',lov_a,'nlj',lov_b,'nij',eri_ovov)

                # slice t1
                take01(nocc,nvir,c,t1,t1s)

                # contraction
                if v is None:
                    v = lib.contraction('nij',eri_ovov, 'nk',t1s,mode,buf=bufz)
                else:
                    lib.contraction('nij',eri_ovov,'nk',t1s,mode,v,beta=1.0)

            # v += lib.contraction('ck',eris.fock[nocc:,:nocc],'ijab',t2, 'ijkabc')
            for i in range(6): # p6 permutation --> loop for all 6 modes
                a, b, c = [abc[j] for j in _inds[i]] 
                mode = _modes[i]
                tmpbuf.loadtag()
                fs = tmpbuf.empty((n, nocc), 'f8')
                t2s = tmpbuf.empty((n, nocc, nocc), 'f8')
                take01(nocc,nvir,c,fov,fs) # slice fock, (kc) -> (nk)
                take011_t2(t2s,a,b,t2,tmpbuf) # slice t2, (ijab) -> (nij)
                lib.contraction('nk',fs,'nij',t2s,mode,v,beta=1.0) # contraction

            # v(z) = 0.5 * v + w, modified in-place.
            lib.elementwise_binary('nijk', w, 'nijk', v, alpha=1.0, gamma=0.5) 
            return v
        
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
        
        w = add_w(abc)
        z = add_z(abc,w) 
        tmpbuf = lib.ArrayBuffer(bufleft)
        r = tmpbuf.empty((n, nocc, nocc, nocc), 'f8')
        r[:] = w
        r3(w,r) 
        div_d3_abc(nocc, *abc, e_occ, e_vir, z)
        div_d3_abc(nocc, *abc, e_occ, e_vir, r)
        sym_multiply(nocc, *abc, r)
        lib.contraction('nikl',z,'njkl',r,'ij',goo,beta=1.0)
        lib.contraction('nkil',z,'nkjl',r,'ij',goo,beta=1.0) 
        lib.contraction('nkli',z,'nklj',r,'ij',goo,beta=1.0)

        # lib.contraction('ijab',t2.conj(),'ijkabc',r,'ck',dvo,alpha=0.5,beta=1.0)
        gvo = tmpbuf.empty((n, nocc), 'f8')
        t2s = tmpbuf.empty((n, nocc, nocc), 'f8')
        take011_t2(t2s,a[:n],b[:n],t2,tmpbuf)
        lib.contraction('nij',t2s,'nijk',r,'nk',gvo,alpha=0.5)
        cupyx.scatter_add(dvo,c[:n],gvo)
        take011_t2(t2s,a[:n],c[:n],t2,tmpbuf)
        lib.contraction('nij',t2s,'nikj',r,'nk',gvo,alpha=0.5)
        cupyx.scatter_add(dvo,b[:n],gvo)
        take011_t2(t2s,b[:n],c[:n],t2,tmpbuf)
        lib.contraction('nij',t2s,'nkij',r,'nk',gvo,alpha=0.5)
        cupyx.scatter_add(dvo,a[:n],gvo)

    a = b = c = abc = w = z = r = bufw = bufz = gvo = t2s = bufleft = buf = tmpbuf = None
    lib.free_all_blocks()

    nijk = nocc * (nocc + 1) * (nocc + 2) // 6
    memory = pool.free_memory
    unit_vir = 3 * nvir * nvir * nvir + nmax * nvir * nvir + nmax * nvir + 3
    if eris.ovvv.l2 is not None: # density fitting is turned on
        naux = eris.ovvv.l1.shape[0]
        unit_vir += naux * nvir * nvir * 2
    blksize = min(nijk, int(memory / 8 / unit_vir))
    # allocate memory for intermediate tensors 
    buf = lib.ArrayBuffer(pool.empty((blksize * unit_vir + 10 * 1024), 'f8'))
    i = buf.empty((blksize), 'i4')
    j = buf.empty((blksize), 'i4')
    k = buf.empty((blksize), 'i4')
    bufw = buf.empty((blksize, nvir, nvir, nvir), 'f8')
    bufz = buf.empty((blksize, nvir, nvir, nvir), 'f8')
    bufleft = buf.left()
    # p6 permutation for slicing (i,j,k)
    inds = numpy.asarray([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1],[2, 1, 0]])
    modes = numpy.asarray(["nabc", "nacb", "nbac", "ncab", "nbca", "ncba"])

    # evaluate gvv by slicing (i,j,k)
    for p0, p1 in list(prange(0, nijk, blksize)): # loop for all blks
        n = p1 - p0
        gen_abc(p0, i[:n], j[:n], k[:n])
        ijk = [i[:n], j[:n], k[:n]] 
        def add_w(ijk):
            w = None
            n = len(ijk[0])
            perm = [3, 5, 1, 4, 0, 2]
            _inds = inds[perm]
            _modes = modes[perm]
            tmpbuf = lib.ArrayBuffer(bufleft)
            if ovvv.l2 is not None:
                naux = ovvv.l1.shape[0]
                lvv = tmpbuf.empty((naux,nvir,nvir),'f8')
                lvv = ovvv.l2.ascupy(buf=lvv)
            tmpbuf.tag()

            # w = lib.contraction('iafb',eris_ovvv,'kjcf',t2, 'ijkabc')
            for idx in range(6): # p6 permutation --> loop for all 6 modes
                i,j,k = [ijk[_idx] for _idx in _inds[idx]] 
                mode = _modes[idx]
                tmpbuf.loadtag()
                # determine the slice of eris_ovvv
                # ##### TODO:
                ## reordering reading order of ovvv!
                eri_ovvv = tmpbuf.empty((n, nvir, nvir, nvir), 'f8')
                if ovvv.l2 is None: 
                    take10(nvir * nvir * nvir, i, ovvv, eri_ovvv)
                else:
                    naux = ovvv.l1.shape[0]
                    lov = tmpbuf.empty((n, naux, nvir), 'f8')
                    take010(naux, nocc, nvir, i, ovvv.l1, lov)
                    lib.contraction('nla', lov, 'lbc', lvv, 'nabc', eri_ovvv)
                # determine the slice of t2
                # ##### TODO:
                ## IMPLEMENT sliced version of take_110_t2
                t2s = tmpbuf.empty((n, nvir, nvir), 'f8')
                take110_t2(t2s,k,j,t2,tmpbuf)
                
                # contraction
                if w is None:
                    w = lib.contraction('nabf', eri_ovvv, 'ncf', t2s, mode, buf=bufw)
                else:
                    w = lib.contraction('nabf', eri_ovvv, 'ncf', t2s, mode, w, beta=1.0)
            
            # w -= lib.contraction('iajm',eris_ovoo,'mkbc',t2, 'ijkabc')
            for idx in range(6): # p6 permutation --> loop for all 6 modes
                i,j,k = [ijk[_idx] for _idx in _inds[idx]] 
                mode = _modes[idx]
                tmpbuf.loadtag()

                # determine the slice of eris_ovoo, ### add density fitting for ovoo!!!
                eri_ovoo = tmpbuf.empty((n, nvir, nocc), 'f8')
                if ovoo.l2 is None:
                    take1010(nocc,nvir,nocc,nocc,i,j,ovoo,eri_ovoo)
                else: # density fitting is turned on
                    naux = ovoo.l1.shape[0]
                    lov = tmpbuf.empty((n, naux, nvir), 'f8')
                    lom = tmpbuf.empty((n, naux, nocc), 'f8')
                    take010(naux,nocc,nvir,i,ovoo.l1,lov)
                    take010(naux,nocc,nocc,j,ovoo.l2,lom)
                    lib.contraction('nla',lov,'nlm',lom,'nam',eri_ovoo)

                # determine the slice of t2
                # ##### TODO:
                ## 1. IMPLEMENT sliced version of take_110_t2
                ## 2. optimize t2 reading order 
                t2ss = tmpbuf.empty((n,nocc,nvir,nvir), 'f8')
                take010_t2v(t2ss,k,t2,tmpbuf)

                # contraction
                w = lib.contraction('nam', eri_ovoo, 'nmbc', t2ss, mode, w, beta=1.0, alpha=-1.0)
            
            return w
        
        def add_z(ijk,w):
            v = None
            tmpbuf = lib.ArrayBuffer(bufleft)
            tmpbuf.tag()
            # perm = [3, 5, 1, 4, 0, 2]
            perm = [0, 1, 2, 3, 4, 5]
            _inds = inds[perm]
            _modes = modes[perm]

            # v = lib.contraction('iajb',eris_ovov.conj(),'kc',t1, 'ijkabc')
            for idx in range(6): # p6 permutation --> loop for all 6 modes
                i, j, k = [ijk[_idx] for _idx in _inds[idx]] 
                mode = _modes[idx]
                tmpbuf.loadtag()
                eri_ovov = tmpbuf.empty((n, nvir, nvir), 'f8')
                t1s = tmpbuf.empty((n, nvir), 'f8')
                # slice ovov
                if ovov.l2 is None: # turn off density fitting
                    take1010(nocc,nvir,nocc,nvir,i,j,ovov,eri_ovov)
                else: # turn on density fitting
                    naux = ovov.l1.shape[0]
                    lov_a = tmpbuf.empty((n, naux, nvir), 'f8')
                    lov_b = tmpbuf.empty((n, naux, nvir), 'f8')
                    take010(naux, nocc, nvir, i, ovov.l1, lov_a)
                    take010(naux, nocc, nvir, j, ovov.l1, lov_b)
                    lib.contraction('nla',lov_a,'nlb',lov_b,'nab',eri_ovov)
                
                # slice t1
                take10(nvir,k,t1,t1s)
                # contraction
                if v is None:
                    v = lib.contraction('nab',eri_ovov, 'nc',t1s,mode,buf=bufz)
                else:
                    lib.contraction('nab',eri_ovov,'nc',t1s,mode,v,beta=1.0)
            
            # v += lib.contraction('ck',eris.fock[nocc:,:nocc],'ijab',t2, 'ijkabc')
            for idx in range(6): # p6 permutation --> loop for all 6 modes
                i, j, k = [ijk[_idx] for _idx in _inds[idx]] 
                mode = _modes[idx]
                tmpbuf.loadtag()
                fs = tmpbuf.empty((n, nvir), 'f8')
                t2s = tmpbuf.empty((n, nvir, nvir), 'f8')
                take10(nvir,k,fov,fs) # slice fock, (kc) -> (nc)
                take110_t2(t2s,i,j,t2,tmpbuf) # slice t2, (ijab) -> (nab)
                lib.contraction('nc',fs,'nab',t2s,mode,v,beta=1.0) # contraction

            # v(z) = 0.5 * v + w, modified in-place.
            lib.elementwise_binary('nabc', w, 'nabc', v, alpha=1.0, gamma=0.5) 
            return v

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
            lib.elementwise_binary('nbca', input, 'nabc', output, gamma=4.0)
            lib.elementwise_binary('ncab', input, 'nabc', output, gamma=1.0)
            lib.elementwise_binary('ncba', input, 'nabc', output, alpha=-2.0, gamma=1.0)
            lib.elementwise_binary('nacb', input, 'nabc', output, alpha=-2.0, gamma=1.0)
            lib.elementwise_binary('nbac', input, 'nabc', output, alpha=-2.0, gamma=1.0)
            return output
        
        w = add_w(ijk) 
        z = add_z(ijk,w) 
        tmpbuf = lib.ArrayBuffer(bufleft)
        r = tmpbuf.empty((n, nvir, nvir, nvir), 'f8')
        r[:] = w
        r3(w,r)
        div_d3_ijk(nvir, *ijk, e_occ, e_vir, z)
        div_d3_ijk(nvir, *ijk, e_occ, e_vir, r)
        sym_multiply(nvir, *ijk, z)
        lib.contraction('nacd',z,'nbcd',r,'ab',gvv,beta=1.0)
        lib.contraction('ncad',z,'ncbd',r,'ab',gvv,beta=1.0) 
        lib.contraction('ncda',z,'ncdb',r,'ab',gvv,beta=1.0)

    # adding to doo and dvv 
    goo_diag = cupy.diag(goo)
    gvv_diag = cupy.diag(gvv)
    doo[numpy.diag_indices(nocc)] -= 0.5 * goo_diag
    dvv[numpy.diag_indices(nvir)] += 0.5 * gvv_diag
    goo_diag = gvv_diag = None
    del buf,tmpbuf,i,j,k,bufw,bufz,bufleft
    lib.free_all_blocks()
    return doo, dov, dvo, dvv

def make_rdm1(mycc, t1, t2, l1, l2, eris=None, ao_repr=False):
    d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
    return ccsd_rdm._make_rdm1(mycc, d1, True, ao_repr=ao_repr)
    