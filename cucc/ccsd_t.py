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

import cupy
import numpy
from pyscf.lib import logger, prange
from byteqc import lib
from byteqc.lib import Mg


inds = numpy.asarray([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1],
                      [2, 1, 0]])
modes = ["nijk", "nikj", "njik", "nkij", "njki", "nkji"]


def r3(input, output):
    '''
    output = 4 * input + input.transpose(1,2,0) + input.transpose(2,0,1)
             - 2 * input.transpose(2,1,0) - 2 * input.transpose(0,2,1)
             - 2 * input.transpose(1,0,2).
    Boosted by cuTENSOR.
    '''
    lib.elementwise_binary('nkij', input, 'nijk', output, gamma=4.0)
    lib.elementwise_binary('njki', input, 'nijk', output, gamma=1.0)
    lib.elementwise_binary('nkji', input, 'nijk', output,
                           alpha=-2.0, gamma=1.0)
    lib.elementwise_binary('nikj', input, 'nijk', output,
                           alpha=-2.0, gamma=1.0)
    lib.elementwise_binary('njik', input, 'nijk', output,
                           alpha=-2.0, gamma=1.0)
    return output


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
div_d3 = cupy.ElementwiseKernel(
    'int64 nocc, raw int32 a, raw int32 b, raw int32 c, raw T eocc, '
    'raw T evir', 'T out',
    '''
        int K = i % nocc;
        size_t iabc = i / nocc;
        int J = iabc % nocc;
        iabc /= nocc;
        int I = iabc %nocc;
        iabc /= nocc;

        int A = a[iabc];
        int B = b[iabc];
        int C = c[iabc];

        T alpha;
        if (A == B)
            alpha = B == C ? 6.0 : 2.0;
        else
            alpha = B == C ? 2.0 : 1.0;

        out /= alpha * (eocc[I] + eocc[J] + eocc[K]
                        - evir[A] - evir[B] - evir[C]);
    ''', 'div_d3')

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
        out = r[inda[m] * extb  + b];
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

# exta, extb, extc = r.shape
# for i in range(n):
#   out[:, off:off+exta][i, :, :] = r[:, indb[i], :]
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


def kernel(mycc, eris, t1=None, t2=None, projector=None):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2

    nocc, nvir = t1.shape
    nmax = max(nocc, nvir)
    nabc = nvir * (nvir + 1) * (nvir + 2) // 6
    pool = mycc.pool
    mo_e = pool.asarray(eris.mo_energy)
    e_occ, e_vir = mo_e[:nocc], mo_e[nocc:]
    fvo = pool.asarray(eris.fock[nocc:, :nocc])

    memory = pool.free_memory
    unit = nocc * nocc * (nmax + nocc) + nocc * nmax + 3
    if projector is not None:
        unit += nocc ** 3

    if eris.ovvv.l2 is not None:
        naux = eris.ovvv.l1.shape[0]
        unit += naux * (nocc + nmax)
    blksize = min(nabc, int(memory / 8 / unit))
    buf = lib.ArrayBuffer(pool.empty((blksize * unit + 10 * 1024), 'f8'))

    t1, ovoo, ovov, ovvv, e_occ, e_vir, fvo, buf = Mg.broadcast(
        t1, eris.ovoo, eris.ovov, eris.ovvv, e_occ, e_vir, fvo, buf)

    if t2.dev == 0:
        # t2 can store in GPU
        t2 = Mg.broadcast(t2)

        def take010_t2(out, ind, t2, buf):
            igpu = Mg.getgid()
            take010(nocc * nocc, nvir, nvir, ind, t2[igpu], out)

        def take011_t2(out, ind1, ind2, t2, buf):
            igpu = Mg.getgid()
            take011(nocc * nocc, nvir, nvir, ind1, ind2, t2[igpu], out)
    else:
        # t2 can't store in GPU
        def take010_t2(out, ind, t2, buf):
            blk = min(nocc * nocc, int(buf.bufsize / 8 / nvir**2))
            n = len(ind)
            t2 = t2.reshape(nocc**2, nvir, nvir)
            buf.tag('t2')
            buf1 = buf.left()
            for p0, p1 in prange(0, nocc * nocc, blk):
                t2p = t2[p0:p1].ascupy(buf=buf1)
                take010_s(nocc * nocc, p0, p1, nvir, nvir, ind, t2p, out,
                          size=n * (p1 - p0) * nvir)
            buf.untag('t2')

        def take011_t2(out, ind1, ind2, t2, buf):
            blk = min(nocc * nocc, int(buf.bufsize / 8 / nvir**2))
            n = len(ind1)
            t2 = t2.reshape(nocc**2, nvir, nvir)
            buf.tag('t2')
            buf1 = buf.left()
            for p0, p1 in prange(0, nocc * nocc, blk):
                t2p = t2[p0:p1].ascupy(buf=buf1)
                take011_s(nocc * nocc, p0, p1, nvir, nvir, ind1,
                          ind2, t2p, out, size=n * (p1 - p0))
            buf.untag('t2')

    a = Mg.mapgpu(lambda buf: buf.empty((blksize), numpy.int32), buf)
    b = Mg.mapgpu(lambda buf: buf.empty((blksize), numpy.int32), buf)
    c = Mg.mapgpu(lambda buf: buf.empty((blksize), numpy.int32), buf)
    bufw = Mg.mapgpu(lambda buf: buf.empty(
        (blksize, nocc, nocc, nocc), 'f8'), buf)
    bufleft = Mg.mapgpu(lambda buf: buf.left(), buf)

    def add_w(abc, w, mode):
        igpu = Mg.getgid()
        a, b, c = abc
        n = len(a)
        tmpbuf = lib.ArrayBuffer(bufleft[igpu])
        tmpbuf.tag()
        t2s = tmpbuf.empty((n, nocc, nocc, nvir), 'f8')
        take010_t2(t2s, c, t2, tmpbuf)
        eri_ovvv = tmpbuf.empty((n, nocc, nvir), 'f8')
        if ovvv[igpu].l2 is None:
            take0110(nocc, nvir, nvir, nvir, a, b, ovvv[igpu], eri_ovvv)
        else:
            naux = ovvv[igpu].l1.shape[0]
            lov = tmpbuf.empty((n, naux, nocc), 'f8')
            lvv = tmpbuf.empty((n, naux, nvir), 'f8')
            take01(naux * nocc, nvir, a, ovvv[igpu].l1, lov)
            take010(naux, nvir, nvir, b, ovvv[igpu].l2, lvv)
            lib.contraction('nlo', lov, 'nlv', lvv, 'nov', eri_ovvv)
        if w is None:
            w = lib.contraction('nif', eri_ovvv, 'nkjf',
                                t2s, mode, buf=bufw[igpu])
        else:
            w = lib.contraction('nif', eri_ovvv, 'nkjf', t2s, mode, w,
                                beta=1.0)

        tmpbuf.loadtag()
        t2ss = tmpbuf.empty((n, nocc, nocc), 'f8')
        take011_t2(t2ss, c, b, t2, tmpbuf)
        eri_ovoo = tmpbuf.empty((n, nocc, nocc, nocc), 'f8')
        if ovoo[igpu].l2 is None:
            take010(nocc, nvir, nocc * nocc, a, ovoo[igpu], eri_ovoo)
        else:
            naux = ovoo[igpu].l1.shape[0]
            lov = tmpbuf.empty((n, naux, nocc), 'f8')
            take01(naux * nocc, nvir, a, ovoo[igpu].l1, lov)
            lib.contraction('nlo', lov, 'lpq', ovoo[igpu].l2, 'nopq', eri_ovoo)

        lib.contraction('nijm', eri_ovoo, 'nkm', t2ss, mode, w,
                        beta=1.0, alpha=-1.0)
        return w

    def r3_d3_e(abc, wr6, et):
        igpu = Mg.getgid()
        tmpbuf = lib.ArrayBuffer(bufleft[igpu])
        vr6 = tmpbuf.empty(wr6.shape, 'f8')
        vr6[:] = wr6
        if projector is None:
            r3(vr6, wr6)
            div_d3(nocc, *abc, e_occ[igpu], e_vir[igpu], wr6)
            lib.contraction('nijk', wr6, 'nijk', vr6, '', et, beta=1.0)
        else:
            r3(wr6, vr6)
            wr6_tmp = tmpbuf.empty(wr6.shape, 'f8')
            lib.contraction('nijp', vr6, 'pk', projector, 'nijk', wr6_tmp)
            div_d3(nocc, *abc, e_occ[igpu], e_vir[igpu], wr6_tmp)
            lib.contraction('nijk', wr6_tmp, 'nijk', wr6, '', et, beta=1.0)
            wr6[:] = wr6_tmp

    def add_v_e(abc, wr6, mode, x, y, tmpbuf):
        igpu = Mg.getgid()
        a, b, c = abc
        n = len(a)
        tmpbuf.loadtag()
        blkoo = tmpbuf.empty((2, n, nocc, nocc), 'f8')
        take011_t2(blkoo[0], a, b, t2, tmpbuf)
        if ovov[igpu].l2 is None:
            take0101(nocc, nvir, nocc, nvir, a, b, ovov[igpu], blkoo[1])
        else:
            naux = ovov[igpu].l1.shape[0]
            lova = tmpbuf.empty((n, naux, nocc), 'f8')
            lovb = tmpbuf.empty((n, naux, nocc), 'f8')
            take01(naux * nocc, nvir, a, ovov[igpu].l1, lova)
            take01(naux * nocc, nvir, b, ovov[igpu].l2, lovb)
            lib.contraction('nlo', lova, 'nlO', lovb, 'noO', blkoo[1])

        take10(nocc, c, fvo[igpu], y[0])
        take01(nocc, nvir, c, t1[igpu], y[1])

        lib.contraction('bnij', blkoo, mode, wr6, 'bnk', x, alpha=0.5)

    def reducetask(et, p):
        if et is None:
            et = cupy.asarray(0, 'f8')
        if p is None:
            return et
        p0, p1 = p
        igpu = Mg.getgid()

        time1 = logger.process_clock(), logger.perf_counter()
        n = p1 - p0
        gen_abc(p0, a[igpu][:n], b[igpu][:n], c[igpu][:n])
        abc = [a[igpu][:n], b[igpu][:n], c[igpu][:n]]
        wr6 = None
        for (ind, mode) in zip(inds, modes):
            wr6 = add_w([abc[i] for i in ind], wr6, mode)

        r3_d3_e(abc, wr6, et)

        tmpbuf = lib.ArrayBuffer(bufleft[igpu])
        x = tmpbuf.empty((12, n, nocc), 'f8')
        y = tmpbuf.empty((12, n, nocc), 'f8')
        tmpbuf.tag()
        q0, q1 = 0, 2
        for (ind, mode) in zip(inds, modes):
            add_v_e([abc[i] for i in ind], wr6, mode, x[q0:q1], y[q0:q1],
                    tmpbuf)
            q0, q1 = q1, q1 + 2
        lib.contraction('bnk', x, 'bnk', y, '', et, beta=1.0)

        time1 = log.timer_debug1(
            'ccsdt GPU%d [%d:%d]/%d' % (Mg.gpus[igpu], p0, p1, nabc), *time1)
        return et

    et = Mg.reduce(reducetask, list(prange(0, nabc, blksize)))
    et = et.item() * 2
    log.timer_debug1('ccsdt', *time0)
    return et
