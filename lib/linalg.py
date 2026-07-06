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
# ByteQC includes code adapted from CuPy (https://github.com/cupy/cupy/),
# which is licensed under the MIT license.
# The original copyright:
#     Copyright (c) 2015 Preferred Infrastructure, Inc.
#     Copyright (c) 2015 Preferred Networks, Inc.

#     Permission is hereby granted, free of charge, to any person obtaining a
#     copy of this software and associated documentation files
#     (the "Software"), to deal in the Software without restriction, including
#     without limitation the rights to use, copy, modify, merge, publish,
#     distribute, sublicense, and/or sell copies of the Software, and to permit
#     persons to whom the Software is furnished to do so, subject to the
#     following conditions:

#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.

#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy
import scipy
import cupy
import cupy_backends.cuda.libs.cutensor as cutensorlib
import cupy.cutensor as cutensor
import nvmath
from nvmath.bindings import cublas as nvcublas
from nvmath.bindings import cusolverDn as nvcusolverDn
from cupy_backends.cuda.libs import cublas, cusolver
from cupy.cublas import _trans_to_cublas_op, _get_scalar_ptr, _decide_ld_and_trans
from cupy.linalg import _util
from cupy.cuda import device
import os.path
from threading import Lock

from byteqc.lib.array import empty_pinned, empty_from_buf, MemoryTypeDevice, \
    MemoryTypeHost
from byteqc.lib.multigpu import Mg
from byteqc.lib.utils import hasnan

# Max int32
INT32MAX = 2147483647


class WsHost:
    def __init__(self):
        self.size = 0
        self.host = empty_pinned((0,), numpy.int8)
        self.lock = Lock()
        self.host_lock = Lock()

    def __call__(self, size):
        with self.lock:
            if self.size < size:
                self.host = None
                self.size = size
                self.host = empty_pinned((size,), numpy.int8)
                self.host_lock = Lock()
                self.host_lock.acquire()
                return self.host, self.host_lock
            else:
                if self.host_lock.acquire(blocking=False):
                    return self.host, self.host_lock
        return empty_pinned((size,), numpy.int8), None


# The global buffer of the host waorspace of cuTENSORMG. The maximum one will be memorized.  # NOQA
DEFAULT_WS_HOST = WsHost()

# A numpy array with size below this threshold will be copied to the device first.  # NOQA
MG_NBYTES_THRESHOLD = 1073741824  # 1GB


def arrayind(*x):
    return numpy.asarray(
        [(isinstance(ix, numpy.ndarray) + 1 - isinstance(ix, cupy.ndarray)) / 2
         for ix in x])


def blaschar(x):
    x = numpy.dtype(x)
    if x.char == 'f':
        return 's'
    if x.char == 'd':
        return 'd'
    if x.char == 'S':
        return 'c'
    if x.char == 'D':
        return 'z'
    raise TypeError('Invalid dtype %s' % str(x))


class CuTensorModel(dict):
    """
    A dict-like class to store the models that how a tensor can be sliced to bypass the CUTENSOR_STATUS_NOT_SUPPORTED error.
    """

    def __init__(self, path='cuTensorModel.dat'):
        self.path = path
        if os.path.isfile(path):
            with open(path, 'r') as f:
                content = f.read()
            if len(content) != 0:
                self.update(eval(content))

    def __setitem__(self, k, v):
        r = super().__setitem__(k, v)
        with open(self.path, 'w') as f:
            f.write(str(self))
        return r

    def __delitem__(self, k):
        r = super().__delitem__(k)
        with open(self.path, 'w') as f:
            f.write(str(self))
        return r


model = CuTensorModel()


ops = {
    'IDENTITY': cutensorlib.OP_IDENTITY,
    'SQRT': cutensorlib.OP_SQRT,
    'RELU': cutensorlib.OP_RELU,
    'CONJ': cutensorlib.OP_CONJ,
    'RCP': cutensorlib.OP_RCP,
    'SIGMOID': cutensorlib.OP_SIGMOID,
    'TANH': cutensorlib.OP_TANH,
    'EXP': cutensorlib.OP_EXP,
    'LOG': cutensorlib.OP_LOG,
    'ABS': cutensorlib.OP_ABS,
    'NEG': cutensorlib.OP_NEG,
    'SIN': cutensorlib.OP_SIN,
    'COS': cutensorlib.OP_COS,
    'TAN': cutensorlib.OP_TAN,
    'SINH': cutensorlib.OP_SINH,
    'COSH': cutensorlib.OP_COSH,
    'ASIN': cutensorlib.OP_ASIN,
    'ACOS': cutensorlib.OP_ACOS,
    'ATAN': cutensorlib.OP_ATAN,
    'ASINH': cutensorlib.OP_ASINH,
    'ACOSH': cutensorlib.OP_ACOSH,
    'ATANH': cutensorlib.OP_ATANH,
    'CEIL': cutensorlib.OP_CEIL,
    'FLOOR': cutensorlib.OP_FLOOR,
    'ADD': cutensorlib.OP_ADD,
    'MUL': cutensorlib.OP_MUL,
    'MAX': cutensorlib.OP_MAX,
    'MIN': cutensorlib.OP_MIN
}


def getop(op, x):
    if op == 'CONJ' and cupy.isrealobj(x):
        return cutensorlib.OP_IDENTITY
    return ops[op]


def _complex2real(inda, a):
    dtype = numpy.dtype(a.dtype.char.lower())
    shape = (*a.shape, 2)
    strides = (*a.strides, dtype.itemsize)
    assert '~' not in inda
    inda = inda + '~'
    return inda, empty_from_buf(a, shape, dtype, strides=strides)


def complex2real(inda, a, indb, b, indc, c):
    '''
    Convert complex-real contractions real-real contractions.
    '''
    if cupy.isrealobj(a):
        assert not cupy.isrealobj(b)
        indb, b = _complex2real(indb, b)
    else:
        assert cupy.isrealobj(b)
        inda, a = _complex2real(inda, a)
    if c is None:
        return inda, a, indb, b, indc + '~', c
    else:
        assert not cupy.isrealobj(c)
        indc, c = _complex2real(indc, c)
        return inda, a, indb, b, indc, c


def real2complex(indc, c):
    '''
    Convert resutls of real-real contractions to the origianl compelx one.
    '''
    dtype = numpy.dtype(c.dtype.char.upper())
    assert indc[-1] == '~'
    return empty_from_buf(c, c.shape[:-1], dtype, strides=c.strides[:-1])


def contraction(
        inda, a, indb, b, indc, c=None, alpha=True, beta=False,
        opa='IDENTITY', opb='IDENTITY', opc='IDENTITY', buf=None,
        alg=cutensorlib.ALGO_DEFAULT, ws_pref=cutensorlib.WORKSPACE_MIN,
        devices=None, ws_device=None, ws_host=None,
        issync=False, isforceMg=False, isskipc2r=False,
        defaulttype=MemoryTypeDevice):
    '''
    Contraction in form `alpha * opa(a) * opb(b) + beta * opc(c) -> c`.
    `inda`, `indb`, `indc` determine the indices of `a`, `b`, `c`.

    buf: The buffer to construct `c` id `c=None`.
    alg: The algorithm to use.
    ws_pref: The workspace preference.

    devices: cuTENSORMG only. The devices list/number to use.
    ws_device: cuTENSORMG only. The device buffer for workspace.
    ws_host: cuTENSORMG only. The host buffer for workspace. If not provied,
        it will be generated and the biggest one will be preserved for later
        used.
    issync: Whether to synchronize the devices after contraction.
    isforceMg: Whether to force use cuTENSORMG.
    isskipc2r: Whether to skip the complex2real tricks.
    defaulttype: The default memory type for case `c` is constructed.
    '''
    # try complex2real tricks
    if not isskipc2r and a.dtype != b.dtype \
            and a.dtype.char.lower() == b.dtype.char.lower() \
            and opa != 'CONJ' and opb != 'CONJ':
        # reparse the complex arr into a real one to get speed gain
        inda, a, indb, b, indc, carr = complex2real(inda, a, indb, b, indc, c)
        carr = contraction_try(inda, a, indb, b, indc, carr, alpha, beta,
                               opa, opb, opc, buf, alg, ws_pref,
                               devices, ws_device, ws_host,
                               isforceMg, defaulttype)
        if c is not None:
            out = c
        else:
            out = real2complex(indc, carr)
        return out

    out = contraction_try(inda, a, indb, b, indc, c, alpha, beta,
                          opa, opb, opc, buf, alg, ws_pref,
                          devices, ws_device, ws_host,
                          isforceMg, defaulttype)
    if issync:
        cupy.cuda.Device().synchronize()
    return out


def contraction_try(
        inda, a, indb, b, indc, c=None, alpha=True, beta=False,
        opa='IDENTITY', opb='IDENTITY', opc='IDENTITY', buf=None,
        alg=cutensorlib.ALGO_DEFAULT, ws_pref=cutensorlib.WORKSPACE_MIN,
        devices=None, ws_device=None, ws_host=None,
        isforceMg=False, defaulttype=MemoryTypeDevice):
    '''
    Try slice the tensors automatically when CUTENSOR_STATUS_NOT_SUPPORTED occurs when contracting.
    '''
    assert a.ndim == len(inda), 'a.ndim not same as len(inda)'
    assert b.ndim == len(indb), 'b.ndim not same as len(indb)'
    cshape = [a.shape[inda.index(
        c_t)] if c_t in inda else b.shape[indb.index(c_t)] for c_t in indc]
    if c is None:
        dtype = numpy.result_type(a, b)
        c = empty_from_buf(buf, cshape, dtype, type=defaulttype)
        beta = 0.0
    else:
        assert c.ndim == len(indc), 'c.ndim not same as len(indc)'

    if isinstance(a, numpy.ndarray) and a.nbytes < MG_NBYTES_THRESHOLD:
        a = cupy.asarray(a)
    if isinstance(b, numpy.ndarray) and b.nbytes < MG_NBYTES_THRESHOLD:
        b = cupy.asarray(b)

    opa = getop(opa, a)
    opb = getop(opb, b)
    opc = getop(opc, c)
    isMg = isforceMg or (arrayind(a, b, c) == 1).any()
    key = (inda, indb, indc, a.shape, b.shape, c.shape)
    if key in model.keys():
        char, n = model[key]
        contraction_slice(inda, a, indb, b, indc, c, alpha, beta, char, n,
                          opa, opb, opc, alg, ws_pref,
                          devices, ws_device, ws_host, isMg)
        return c
    else:
        try:
            _contraction(inda, a, indb, b, indc, c, alpha, beta,
                         opa, opb, opc, alg, ws_pref,
                         devices, ws_device, ws_host, isMg)
            return c
        except BaseException as err:
            if (isinstance(err, cutensorlib.CuTensorError)
                and str(err) == 'CUTENSOR_STATUS_NOT_SUPPORTED') \
                    or isinstance(err, cupy.cuda.memory.OutOfMemoryError):
                print('CUTENSOR_ERROR:%s!!! Try to slice...' % str(err))
                arrs = [a, b, c]
                perm = numpy.argsort([numpy.prod(x.shape)
                                      for x in arrs])[::-1]
                inds = [inda, indb, indc]
                inds = inds[perm[0]] + inds[perm[1]] + inds[perm[2]]
                shapes = numpy.hstack([arrs[x].shape for x in perm])
                iters = []
                for i, char in enumerate(inds):
                    item = (char, shapes[i])
                    if item not in iters:
                        iters.append(item)
                for char, shape in iters:
                    n = 2
                    while (shape + n - 1) // n >= 4:
                        try:
                            contraction_slice(
                                inda, a, indb, b, indc, c, alpha, beta,
                                char, n, opa, opb, opc, alg, ws_pref,
                                devices, ws_device, ws_host, isMg)
                            model[key] = (char, n)
                            print('Slice (%c, %d) successed and saved!' % (
                                char, n))
                            return c
                        except BaseException as e:
                            print('Slice (%c, %d) failed!:%s' % (
                                char, n, str(e)))
                            pass
                        n += 2
            raise err


def token(shape, i, n):
    a = shape // n
    return i * a, (i + 1) * a


def getslice(x, ind, char, p, n):
    s = []
    for i in range(len(ind)):
        if char == ind[i]:
            shape = x.shape[i]
            a = shape // n
            s.append(slice(p * a, (p + 1) * a))
        else:
            s.append(slice(None))
    return x[*s]


def get_slice(x, ind, n):
    shape = x.shape[ind]
    blksize = (shape + n - 1) // n
    return [x[*((slice(p0, p0 + blksize) if i == ind else slice(None))
            for i in range(x.ndim))] for p0 in range(0, shape, blksize)]


def get_slices(a, modea, b, modeb, beta, c, modec, char, n):
    if char in modea:
        a = get_slice(a, modea.index(char), n)
    else:
        a = [a] * n

    if char in modeb:
        b = get_slice(b, modeb.index(char), n)
    else:
        b = [b] * n

    if char in modec:
        beta = [beta] * n
        c = get_slice(c, modec.index(char), n)
    else:
        beta = [beta, *([True] * (n - 1))]
        c = [c] * n
    return zip(a, b, beta, c)


def contraction_slice(
        inda, a, indb, b, indc, c, alpha, beta, char, n, *args):
    for ai, bi, beta, ci in get_slices(
            a, inda, b, indb, beta, c, indc, char, n):
        _contraction(
            inda, ai, indb, bi, indc, ci, alpha, beta, *args)
    return c


def _contraction(inda, a, indb, b, indc, c, alpha, beta,
                 opa, opb, opc, alg, ws_pref,
                 devices, ws_device, ws_host, isMg):
    if isMg:
        assert opa == opb and opb == opc and opc == cutensorlib.OP_IDENTITY, \
            'cutensorMg not support operator'
        if devices is None:
            devices = Mg.gpus
        hostBufSize, deviceBuf = cutensor.contractionMgWorkspace(
            alpha, a, inda, b, indb, beta, c, indc,
            ws_pref=ws_pref, devices=devices)
        if ws_device is None:
            ws_device = []
            for size, gid in zip(deviceBuf, devices):
                with cupy.cuda.Device(gid):
                    r = cupy.empty((size,), numpy.int8)
                ws_device.append(r)
        if ws_host is None:
            ws_host, ws_lock = DEFAULT_WS_HOST(hostBufSize)
        else:
            ws_lock = None
        # model[('MG', inda, indb, indc, a.shape, b.shape, c.shape,
        #        a.strides, b.strides, c.strides)] = (
        #     str(type(a)), str(type(b)), str(type(c)))
        r = cutensor.contractionMg(
            alpha, a, inda, b, indb, beta, c, indc, hostBuf=ws_host,
            ws_pref=ws_pref, deviceBuf=ws_device, devices=devices)
        for d in devices:
            cupy.cuda.Device(d).synchronize()
        if ws_lock is not None:
            ws_lock.release()
        # del model[('MG', inda, indb, indc, a.shape, b.shape, c.shape,
        #            a.strides, b.strides, c.strides)]
        return r
    else:
        return cutensor.contraction(
            alpha, a, inda, b, indb, beta, c, indc, algo=alg,
            ws_pref=ws_pref, op_A=opa, op_B=opb, op_C=opc)


def elementwise_binary(inda, a, indc, c=None, out=None, alpha=True,
                       gamma=False, opa='IDENTITY', opc='IDENTITY',
                       opac='ADD', buf=None):
    """
        out = opac(alpha * opa(A), gamma * opc(C)). If out is None, c will be modified in-place. `buf` is used to construct the output array if both `c` and `out` equals `None`.
    """
    opa = getop(opa, a)
    opc = getop(opc, c)
    opac = getop(opac, c)
    if c is None:
        va = a.transpose(*tuple(indc.index(i) for i in inda))
        c = empty_from_buf(buf, va.shape, va.dtype)
        gamma = False
        out = c
    if out is None:
        out = c
    return cutensor.elementwise_binary(
        alpha, a, inda, gamma, c, indc, out, opa, opc, opac)


def elementwise_trinary(
        inda, a, indb, b, indc, c=None, out=None, alpha=True, beta=True,
        gamma=False, opa='IDENTITY', opb='IDENTITY', opc='IDENTITY',
        opab='ADD', opabc='ADD', buf=None):
    """
        out = opabc(opab(alpha * opa(A), beta * opb(B)), gamma * opc(C)).  If out is None, c will be modified in-place. `buf` is used to construct the output array if both `c` and `out` equals `None`.
    """
    opa = getop(opa, a)
    opb = getop(opb, a)
    opc = getop(opc, c)
    opab = getop(opab, c)
    opabc = getop(opabc, c)
    if c is None:
        va = a.transpose(*tuple(indc.index(i) for i in inda))
        c = empty_from_buf(buf, va.shape, va.dtype)
        gamma = False
        out = c
    if out is None:
        out = c
    return cutensor.elementwise_trinary(
        alpha, a, inda, beta, b, indb, gamma, c, indc, out,
        opa, opb, opc, opab, opabc)


def gemm(a, b, c=None, alpha=True, beta=False,
         buf=None, transa='N', transb='N'):
    '''
        c = alpha * a @ b + beta * c. `transa`, `transb` can be `N` for normal, `T` for transpose, `H` for conjugate transpose. `buf` is used to construct the output array if `c` equals `None`.
    '''
    assert a.ndim == b.ndim == 2, 'Only matrix is support for gemm'
    if isinstance(a, numpy.ndarray) or isinstance(b, numpy.ndarray) \
       or a.dtype != b.dtype or isinstance(c, numpy.ndarray) \
       or not (c is None or c.flags.c_contiguous or c.flags.f_contiguous):
        return contraction(
            'ab' if transa == 'N' else 'ba', a,
            'bc' if transb == 'N' else 'cb', b,
            'ac', c, alpha=alpha, beta=beta, buf=buf,
            opa='CONJ' if transa == 'H' else 'IDENTITY',
            opb='CONJ' if transb == 'H' else 'IDENTITY')
    _transa = _trans_to_cublas_op(transa)
    _transb = _trans_to_cublas_op(transb)
    if _transa == cublas.CUBLAS_OP_N:
        m, k = a.shape
    else:
        k, m = a.shape
    if _transb == cublas.CUBLAS_OP_N:
        n = b.shape[1]
        assert b.shape[0] == k, 'Invaid b shape'
    else:
        n = b.shape[0]
        assert b.shape[1] == k, 'Invaid b shape'
    lda, _transa = _decide_ld_and_trans(a, _transa)
    ldb, _transb = _decide_ld_and_trans(b, _transb)
    if lda is None or ldb is None:
        return contraction(
            'ab' if transa == 'N' else 'ba', a,
            'bc' if transb == 'N' else 'cb', b,
            'ac', c, alpha=alpha, beta=beta, buf=buf,
            opa='CONJ' if transa == 'H' else 'IDENTITY',
            opb='CONJ' if transb == 'H' else 'IDENTITY')

    dtype = a.dtype
    if c is None:
        c = empty_from_buf(buf, (m, n), dtype, type=MemoryTypeDevice)
        beta = 0.0
    else:
        assert c.dtype == dtype, 'Wrong datatype (%d) for c' % str(c.dtype)
    if any([m > INT32MAX, n > INT32MAX, k > INT32MAX]):
        suffix = '_64'
    else:
        suffix = ''
    func = getattr(nvcublas, blaschar(dtype) + 'gemm' + suffix)

    alpha, alpha_ptr = _get_scalar_ptr(alpha, dtype)
    beta, beta_ptr = _get_scalar_ptr(beta, dtype)
    handle = device.get_cublas_handle()
    orig_mode = cublas.getPointerMode(handle)
    if isinstance(alpha, cupy.ndarray) or isinstance(beta, cupy.ndarray):
        if not isinstance(alpha, cupy.ndarray):
            alpha = cupy.array(alpha)
            alpha_ptr = alpha.data.ptr
        if not isinstance(beta, cupy.ndarray):
            beta = cupy.array(beta)
            beta_ptr = beta.data.ptr
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
    else:
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    if c.flags.f_contiguous:
        try:
            func(
                handle, _transa, _transb, m, n, k, alpha_ptr, a.data.ptr,
                lda, b.data.ptr, ldb, beta_ptr, c.data.ptr, m)
        finally:
            cublas.setPointerMode(handle, orig_mode)
    else:
        # Computes out.T = alpha * b.T @ a.T + beta * out.T
        try:
            func(handle, 1 - _transb, 1 - _transa, n, m, k,
                 alpha_ptr, b.data.ptr, ldb, a.data.ptr, lda,
                 beta_ptr, c.data.ptr, n)
        finally:
            cublas.setPointerMode(handle, orig_mode)
    return c


def cudadatatype(dtype):
    if dtype == cupy.dtype('f'):
        return nvmath.CudaDataType.CUDA_R_32F
    if dtype == cupy.dtype('d'):
        return nvmath.CudaDataType.CUDA_R_64F
    if dtype == cupy.dtype('F'):
        return nvmath.CudaDataType.CUDA_C_32F
    if dtype == cupy.dtype('D'):
        return nvmath.CudaDataType.CUDA_C_64F
    raise TypeError("Invalid datatype %s" % str(dtype))


def svd(a, s=None, u=None, vt=None, jobu='S', jobvt='S', overwrite_a=False,
        ws_host=None, ws_device=None):
    '''
    Compute the singular value decomposition of a matrix. The input `s`, `u`, and `vt` will be used as buffer to construct the output. overwrite_a is used to indicate whether the input `a` will be overwritten. `ws_host` and `ws_device` are used as buffer to specify the workspace for the computation.
    `a` can be c-contiguous or f-contiguous.
    '''
    assert a.ndim == 2
    handle = device.get_cusolver_handle()
    istrans = False
    if not overwrite_a:
        a = cupy.copy(a, order='F')
    else:
        if a.flags.c_contiguous:
            istrans = True
            a = a.T
            jobu, jobvt = jobvt, jobu
        else:
            assert a.flags.f_contiguous, "`a` must be contiguous."
    m, n = a.shape
    mn = min(m, n)
    stype = cupy.dtype(a.dtype.char.lower())
    s = empty_from_buf(s, (mn), dtype=stype, type=MemoryTypeDevice)
    if jobu == 'N':
        uptr = 0
    else:
        if jobu == 'A':
            u = empty_from_buf(u, (m, m), dtype=a.dtype,
                               order='F', type=MemoryTypeDevice)
        elif jobu == 'S':
            u = empty_from_buf(u, (m, mn), dtype=a.dtype,
                               order='F', type=MemoryTypeDevice)
        elif jobu == 'O':
            u = a
        else:
            raise ValueError("Invalid `jobu` %s" % jobu)
        uptr = u.data.ptr

    if jobvt == 'N':
        vtptr = 0
    else:
        if jobvt == 'A':
            vt = empty_from_buf(vt, (m, m), dtype=a.dtype,
                                order='F', type=MemoryTypeDevice)
        elif jobvt == 'S':
            vt = empty_from_buf(vt, (m, mn), dtype=a.dtype,
                                order='F', type=MemoryTypeDevice)
        elif jobvt == 'O':
            vt = a
        else:
            raise ValueError("Invalid `jobvt` %s" % jobvt)
        vtptr = vt.data.ptr

    dtype = cudadatatype(a.dtype)
    stype = cudadatatype(stype)
    wsd, wsh = nvcusolverDn.xgesvd_buffer_size(
        handle, 0, ord(jobu),
        ord(jobvt),
        m, n, dtype, a.data.ptr, m, stype, s.data.ptr, dtype, uptr, m,
        dtype, vtptr, n, dtype)
    if wsd == 0:
        wsptr_device = 0
    else:
        ws_device = empty_from_buf(
            ws_device, (wsd,),
            dtype=numpy.int8, type=MemoryTypeDevice)
        wsptr_device = ws_device.data.ptr
    if wsh == 0:
        wsptr_host = 0
    else:
        ws_host = empty_from_buf(
            ws_host, (wsh,),
            dtype=numpy.int8, type=MemoryTypeHost)
        wsptr_host = ws_host.ctypes.data
    info = cupy.asarray(0)
    nvcusolverDn.xgesvd(handle, 0, ord(jobu),
                        ord(jobvt),
                        m, n, dtype, a.data.ptr, m, stype, s.data.ptr,
                        dtype, uptr, m, dtype, vtptr, n, dtype, wsptr_device,
                        wsd, wsptr_host, wsh, info.data.ptr)

    if istrans:
        u, vt = None if vt is None else vt.T, None if u is None else u.T
        jobu, jobvt = jobvt, jobu

    if jobu == 'N':
        if jobvt == 'N':
            return s
        else:
            return s, vt
    else:
        if jobvt == 'N':
            return u, s
        else:
            return u, s, vt


def solve_triangular(
        a, b, trans='N', lower=False, unit_diagonal=False, overwrite_b=False,
        left=True, check_finite=False, alpha=True):
    '''
    Solve the equation `op(a) x = alpha * b` for `x` if `left=True` and `x op(a) = alpha * b` if `left=False`.
    `a` is a triangular matrix and only the lower of `a` will be used if `lower=True`.
    `unit_diagonal` is used to indicate whether the diagonal of `a` will be treat as unit.
    `overwrite_b` is used to indicate whether the input `b` will be overwritten.
    `check_finite` is used to check whether the input `a` and `b` are finite.
    '''
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError('Expected square matrix')
    if len(a) != len(b):
        raise ValueError('Incompatible dimensions')
    dtype = a.dtype
    if dtype != b.dtype:
        raise ValueError('Incompatible dtypes')

    if check_finite:
        if a.dtype.kind == 'f' and not cupy.isfinite(a).all():
            raise ValueError(
                'Array must not contain infs or NaNs')
        if b.dtype.kind == 'f' and not cupy.isfinite(b).all():
            raise ValueError(
                'Array must not contain infs or NaNs')

    if not overwrite_b:
        b = b.copy()
    if b.ndim == 1:
        shape = (b.size, 1)
    else:
        shape = b.shape

    cublas_handle = device.get_cublas_handle()
    alpha = numpy.array(alpha, dtype=dtype)
    ntrans = 0
    lda = a.shape[0]
    if a.flags.c_contiguous:
        ntrans += 1
        lower = not lower
    if b.flags.c_contiguous:
        ntrans += 1
        left = not left
        n, m = shape
    else:
        m, n = shape
    ldb = m
    if ntrans == 1:
        assert trans != cublas.CUBLAS_OP_C, "Not support such operations"
        trans = 'N' if trans == 'T' else 'T'

    if trans == 'N':
        trans = cublas.CUBLAS_OP_N
    elif trans == 'T':
        trans = cublas.CUBLAS_OP_T
    elif trans == 'C':
        trans = cublas.CUBLAS_OP_C

    if lower:
        uplo = cublas.CUBLAS_FILL_MODE_LOWER
    else:
        uplo = cublas.CUBLAS_FILL_MODE_UPPER

    if unit_diagonal:
        diag = cublas.CUBLAS_DIAG_UNIT
    else:
        diag = cublas.CUBLAS_DIAG_NON_UNIT

    if left:
        side = cublas.CUBLAS_SIDE_LEFT
    else:
        side = cublas.CUBLAS_SIDE_RIGHT

    if dtype == 'f':
        trsm = cublas.strsm
    elif dtype == 'd':
        trsm = cublas.dtrsm
    elif dtype == 'F':
        trsm = cublas.ctrsm
    else:  # dtype == 'D'
        trsm = cublas.ztrsm
    trsm(cublas_handle, side, uplo, trans, diag, m, n, alpha.ctypes.data,
         a.data.ptr, lda, b.data.ptr, ldb)
    return b


def cholesky(a, overwrite=False):
    '''
    Compute the Cholesky decomposition of a matrix `a`. `overwrite` is used to indicate whether the input `a` will be overwritten.
    '''
    _util._assert_cupy_array(a)
    m, n = a.shape
    assert m == n, "`a` must be a square matrix"

    dtype, out_dtype = _util.linalg_common_type(a)
    if a.size == 0:
        return cupy.empty(a.shape, out_dtype)

    if not overwrite:
        a = cupy.copy(a, order='C')
    else:
        assert a.flags.c_contiguous or a.flags.f_contiguous, \
            '`a` must be contiguous.'
    handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)

    if dtype == 'f':
        potrf = cusolver.spotrf
        potrf_bufferSize = cusolver.spotrf_bufferSize
    elif dtype == 'd':
        potrf = cusolver.dpotrf
        potrf_bufferSize = cusolver.dpotrf_bufferSize
    elif dtype == 'F':
        potrf = cusolver.cpotrf
        potrf_bufferSize = cusolver.cpotrf_bufferSize
    else:  # dtype == 'D':
        potrf = cusolver.zpotrf
        potrf_bufferSize = cusolver.zpotrf_bufferSize

    if a.flags.c_contiguous:
        uplo = cublas.CUBLAS_FILL_MODE_UPPER
    else:
        uplo = cublas.CUBLAS_FILL_MODE_LOWER

    buffersize = potrf_bufferSize(handle, uplo, n, a.data.ptr, n)
    workspace = cupy.empty(buffersize, dtype=dtype)
    potrf(handle, uplo, n, a.data.ptr, n,
          workspace.data.ptr, buffersize, dev_info.data.ptr)
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrf, dev_info)
    if hasnan(a):
        raise cupy.linalg.LinAlgError(
            'Input matrix may not be positive definite, the result of the'
            ' cholesky decomposition includes nan.')
    _util._tril(a, k=0)
    return a


def check1d(x, y):
    assert len(x) == len(y), "The length of x, y must be same."
    assert x.dtype == y.dtype, "The dtype of x, y must be same."


def ravel(x):
    if x.flags.c_contiguous or x.flags.f_contiguous:
        x = x.ravel()
    else:
        assert x.ndim == 1, "Please ravel incontiguous array (need copy) first"
    return x


def blas_host(name, x, y=None, a=None):
    func = getattr(scipy.linalg.blas, blaschar(x.dtype) + name)
    n = len(x)
    kwgs = {'incx': x.strides[0] // x.dtype.itemsize,
            'offx': 0}
    if a is not None:
        kwgs['a'] = a
    if y is not None:
        kwgs['offy'] = 0
        kwgs['incy'] = y.strides[0] // y.dtype.itemsize
    for i in range(0, n, INT32MAX):
        s = slice(i, i + INT32MAX)
        _n = len(x[s])
        kwgs['n'] = _n
        kwgs['x'] = x[s]
        if y is not None:
            kwgs['y'] = y[s]
        func(**kwgs)


def blas_device(name, x, y=None, a=None):
    n = len(x)
    suffix = '_64' if n > INT32MAX else ''
    func = getattr(nvcublas, blaschar(x.dtype) + name + suffix)
    kwgs = {'n': n, 'x': x.data.ptr, 'incx': x.strides[0] // x.dtype.itemsize}
    if a is not None:
        na = numpy.asarray(a)
        kwgs['alpha'] = na.ctypes.data
    if y is not None:
        kwgs['y'] = y.data.ptr
        kwgs['incy'] = y.strides[0] // x.dtype.itemsize
    func(device.get_cublas_handle(), **kwgs)


def scal(a, x):
    '''
    Scale a cpu/gpu vector by a scalar in-place.
    '''
    x0 = ravel(x)
    if isinstance(x0, numpy.ndarray):
        blas_host('scal', x, a=a)
    elif isinstance(x, cupy.ndarray):
        blas_device('scal', x, a=a)
    return x


def blas2(name, x, y, a=None):
    x0 = ravel(x)
    y0 = ravel(y)
    check1d(x0, y0)
    arrind = arrayind(x0, y0)
    if (arrind == 1).all():
        blas_host(name, x0, y0, a)
    elif (arrind == 0).all():
        blas_device(name, x0, y0, a)
    else:
        raise ValueError('x, y are not the same type')


def copy(x, y):
    '''
    Copy x to y with blas library.
    '''
    blas2('copy', x, y)
    return y


def axpy(x, y, a=1):
    '''
    y = a*x + y for cpu/gpu array with blas library.
    '''
    blas2('axpy', x, y, a)
    return y


def swap(x, y):
    '''
    Swap x and y with blas library.
    '''
    blas2('swap', x, y)
    return x, y
