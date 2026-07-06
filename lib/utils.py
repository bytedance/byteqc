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

import cupy
import numpy
from byteqc.lib.array import empty_from_buf
from pyscf.lib import HERMITIAN


def is_pinned(x):
    '''
    Check if the memory is pinned.
    '''
    return cupy.cuda.pinned_memory.is_memory_pinned(x.ctypes.data)


def gpu_avail_bytes(ratio=0.9):
    '''
    Get the available memory multiplied by `ratio` in bytes on GPU.
    '''
    pool = cupy.get_default_memory_pool()
    pool.free_all_blocks()
    return int(ratio * (cupy.cuda.Device().mem_info[0]))


def gpu_used_bytes():
    '''
    Get the used memory in bytes on GPU. The free memory in cupy memory pool will be treated as free memory also.
    '''
    pool = cupy.get_default_memory_pool()
    mem = cupy.cuda.Device.mem_info
    return mem[1] - mem[0] - (pool.total_bytes() - pool.pool.used_bytes())


def free_all_blocks():
    '''
    Free all blocks in cupy memory pool.
    '''
    cupy.get_default_memory_pool().free_all_blocks()


def pool_status():
    '''
    Get the used memory and total memory in GB in cupy memory pool.
    '''
    pool = cupy.get_default_memory_pool()
    return pool.used_bytes() / 1024**3, pool.total_bytes() / 1024**3


def pack_tril(mat, axis=-1, out=None):
    '''
    Pack a cupy array into the triangle form. `axis` determines which to pack
    when rank of `mat` is 3. `out` is used as the buffer to store the packed
    array.
    '''
    if mat.size == 0:
        return cupy.zeros(mat.shape + (0, ), dtype=mat.dtype)

    if mat.ndim == 2:
        nd = mat.shape[0]
        shape = nd * (nd + 1) // 2
        if out is None:
            out = cupy.empty(shape, dtype=mat.dtype)
        else:
            out = cupy.ndarray(shape, dtype=mat.dtype, memptr=out.data)
        ker = cupy.ElementwiseKernel(
            'int64 m, raw T y', 'T out', '''
                int ind = floor(sqrt(0.25+2*i)-0.5);
                int res = i-ind*(ind+1)/2;
                out = y[ind*m+res];
            ''', 'pack_tril_2d')
        ker(nd, mat, out)
        return out

    if mat.ndim == 3:
        if axis == -1 or axis == 2:
            count, nd = mat.shape[:2]
            shape = (count, nd * (nd + 1) // 2)
            ker = cupy.ElementwiseKernel(
                'int64 m, int64 nd, raw T y', 'T out', '''
                    int strid = nd*(nd+1)/2;
                    int a = i / strid;
                    int c = i % strid;
                    int b = floor(sqrt(0.25+2*c)-0.5);
                    c = c-b*(b+1)/2;
                    out = y[a * nd * nd + b * nd + c];
                ''', 'pack_tril_3d')
        else:
            assert axis == 0, "Unsupported axis"
            nd, count = mat.shape[1:]
            shape = (nd * (nd + 1) // 2, count)
            ker = cupy.ElementwiseKernel(
                'int64 m, int64 nd, raw T y', 'T out', '''
                    int c = i % m;
                    int b = i / m;
                    int a = floor(sqrt(0.25+2*b)-0.5);
                    b = b - a*(a+1)/2;
                    out = y[a * nd * m + b * m + c];
                ''', 'pack_tril_3d_0')
        if out is None:
            out = cupy.empty(shape, dtype=mat.dtype)
        else:
            out = cupy.ndarray(shape, dtype=mat.dtype, memptr=out.data)
        ker(count, nd, mat, out)
        return out


def unpack_tril(tril, filltriu=HERMITIAN, axis=-1, out=None):
    '''
    Pack a cupy array into the triangle form.
    `filltriu` is used to fill the upper triangular part of the matrix.
    `filltriu` can be `pyscf.lib.HERMITIAN`, `pyscf.lib.ANTIHERMI.
    `axis` determines which to unpack when rank of `mat` is 2. `out` is used
    as the buffer to store the packed array.
    '''
    if filltriu == HERMITIAN and cupy.iscomplexobj(tril):
        assert False, "Not, implement"
    if tril.ndim == 1:
        n = int(numpy.sqrt(2 * tril.shape[0]))
        out = empty_from_buf(out, (n, n), tril.dtype)
        ker = cupy.ElementwiseKernel(
            'int64 m, raw T y, int64 fill', 'T out', '''
                int a = i / m;
                int b = i % m;
                if (a >= b)
                    out = y[a*(a+1)/2+b];
                else if (fill == 1)
                    out = y[b*(b+1)/2+a];
                else if (fill == 2)
                    out = -y[b*(b+1)/2+a];
            ''', 'unpack_tril_2d')
        ker(n, tril, filltriu, out)
    elif tril.ndim == 2:
        if axis == -1 or axis == 1:
            m = tril.shape[0]
            n = int(numpy.sqrt(2 * tril.shape[1]))
            out = empty_from_buf(out, (m, n, n), tril.dtype)
            ker = cupy.ElementwiseKernel(
                'int64 nd, raw T y, int64 fill', 'T out', '''
                    int c = i % nd;
                    int a = i / nd;
                    int b = a % nd;
                    a /= nd;
                    if (b >= c)
                        out = y[a * nd * (nd+1)/2 +  b*(b+1)/2+c];
                    else if (fill == 1)
                        out = y[a * nd * (nd+1)/2 + c*(c+1)/2+b];
                    else if (fill == 2)
                        out = -y[a * nd * (nd+1)/2 + c*(c+1)/2+b];
                ''', 'unpack_tril_3d')
            ker(n, tril, filltriu, out)
        else:
            m = tril.shape[1]
            n = int(numpy.sqrt(2 * tril.shape[0]))
            out = empty_from_buf(out, (n, n, m), tril.dtype)
            ker = cupy.ElementwiseKernel(
                'int64 m, int64 nd, raw T y, int64 fill', 'T out', '''
                    int c = i % m;
                    int a = i / m;
                    int b = a % nd;
                    a = a / nd;
                    if (a >= b)
                        out = y[(a*(a+1)/2+b)*m+c];
                    else if (fill == 1)
                        out = y[(b*(b+1)/2+a)*m+c];
                    else if (fill == 2)
                        out = -y[(b*(b+1)/2+a)*m+c];
                ''', 'unpack_tril_3d_1')
            ker(m, n, tril, filltriu, out)
    else:
        if axis == 0:
            cshape = tril.shape[1:]
            dim = numpy.prod(cshape)
            out = unpack_tril(tril.reshape(-1, dim), filltriu, 0, out)
            return out.reshape((*out.shape[:2], *cshape))
        elif axis == -1 or axis == tril.ndim - 1:
            cshape = tril.shape[:-1]
            dim = numpy.prod(cshape)
            out = unpack_tril(tril.reshape(dim, -1), filltriu, -1, out)
            return out.reshape((*cshape, *out.shape[-2:]))
        else:
            raise NotImplementedError("Unsupport ndims")

    return out


def hasnan(a):
    '''
    Check whether a cupy array contains nan in-place.
    '''
    ker = cupy.ReductionKernel(
        'T x',  # input params
        'bool out',  # output params
        'isnan(x) ? 1 : 0',  # map
        'a + b',  # reduce
        'out = a > 0',  # post-reduction map
        '0',  # identity value
        'hasnan'  # kernel name
    )
    return ker(a)
