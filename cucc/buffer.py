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
import h5py
import os
import tempfile
import bisect
import warnings
from pyscf.lib import param
from pyscf.lib import logger
from numbers import Number
import random
from byteqc.lib import empty, MemoryTypeDevice, MemoryTypeHost, \
    free_all_blocks, pool_status

log = logger.new_logger(None, 10)
# INFO (4) summary
# DEBUG (5) pull and push log
# DEBUG2 (7) enter and fix
# DEBUG4 (9) check


def getdev(val):
    if isinstance(val, cupy.ndarray):
        return 0
    elif isinstance(val, numpy.ndarray):
        return 1
    else:
        assert isinstance(
            val, h5py.Dataset), 'Unexpected type ' + str(type(val))
        return 2


def _init(obj, buf, dev, regist=True):
    obj.buf = buf
    obj.dev = dev
    obj.regist = regist
    if regist:
        buf.free[dev] -= obj.nbytes


class BufArr:
    '''
    BufArr is the base class for BufCupy and BufNumpy.
    It is used to trace the usage of array in BufPool.
    It is not intended to be used directly.
    '''

    def __del__(self):
        if self.regist:
            self.buf.free[self.dev] += self.nbytes

    def __array_finalize__(self, obj):
        if obj is None:
            return
        if getattr(obj, 'buf', None) is not None:
            self.buf = obj.buf
            self.dev = obj.dev
            self.regist = False

    def _op(self, op, val):
        dev = self.dev
        if dev == 2:
            arr = self.arr[:]
            op(arr, self.buf.forop(dev, val))
            self.arr[:] = arr
        else:
            op(self, self.buf.forop(dev, val))
        return self

    def enter(self, **kwgs):
        if self.dev == 0 and (
                self.flags.c_contiguous or self.flags.f_contiguous):
            return self
        else:
            self.tmp = self.ascupy(order='c', **kwgs)
            return self.tmp

    def __enter__(self):
        if getattr(self, 'enter_kwg', None) is None:
            return self.enter()
        else:
            r = self.enter(**self.enter_kwg)
            del self.enter_kwg
            return r

    def exit(self):
        if getattr(self, 'tmp', None) is not None:
            self.buf.copyto(self.tmp, self)
            cupy.cuda.get_current_stream().synchronize()
            del self.tmp

    def __exit__(self, type, valse, tb):
        return self.exit()

    def asnumpy(self, stream=None):
        if isinstance(self, cupy.ndarray):
            r = self.buf.empty(self.shape, self.dtype, type=MemoryTypeHost)
            self.get(out=r, stream=stream)
            if stream is not None:
                stream.synchronize()
            return r
        elif isinstance(self, numpy.ndarray):
            return self
        else:
            return self.arr[:]

    def ascupy(self, stream=None, order=None, copy=False, buf=None):
        if not copy and isinstance(self, cupy.ndarray):
            if order is None \
                    or (order in 'cC' and self.flags.c_contiguous) \
                    or (order in 'fF' and self.flags.f_contiguous):
                return self
        if buf is None:
            r = self.buf.empty(self.shape, self.dtype, order=order)
        else:
            r = cupy.ndarray(self.shape, self.dtype, order=order,
                             memptr=buf.data)
        if isinstance(self, cupy.ndarray):
            r[:] = self
        else:
            if isinstance(self, numpy.ndarray):
                arr = self
            else:
                arr = self.arr[:]
            r.set(arr, stream=stream)
            if stream is not None:
                stream.synchronize()
        return r

    def __broad_as__(self):
        r = self.ascupy()
        return r, r

    def __broad_new__(self):
        arr = cupy.empty(self.shape, dtype=self.dtype)
        wrapper = BufCupy(self.buf, arr, regist=False)
        return arr, wrapper


class BufCupy(cupy.ndarray, BufArr):
    '''
    BufCupy is a wrapper for cupy.ndarray with the usage of cupy.ndarray in BufPool traced.
    '''
    __array_finalize__ = BufArr.__array_finalize__

    def __new__(cls, buf, arr, regist=True):
        obj = arr.view(cls)
        _init(obj, buf, 0, regist)
        return obj

    def __setitem__(self, key, val):
        if isinstance(val, Number):
            cupy.ndarray.__setitem__(self, key, val)
        else:
            self.buf.copyto(val, self, key)

    def __iadd__(self, val):
        return self._op(cupy.ndarray.__iadd__, val)

    def __isub__(self, val):
        return self._op(cupy.ndarray.__isub__, val)

    def __imul__(self, val):
        return self._op(cupy.ndarray.__imul__, val)

    def __itruediv__(self, val):
        return self._op(cupy.ndarray.__itruediv__, val)

    def dot(self, x, out=None):
        return cupy.ndarray.dot(self, self.buf.forop(self.dev, x), out=out)

    def __broad_as__(self):
        return self, self

    def __broad_new__(self):
        arr = cupy.empty(self.shape, dtype=self.dtype)
        wrapper = BufCupy(self.buf, arr, regist=False)
        return arr, wrapper


class BufNumpy(numpy.ndarray, BufArr):
    '''
    BufNumpy is a wrapper for numpy.ndarray with the usage of numpy.ndarray in BufPool traced.
    '''
    __array_finalize__ = BufArr.__array_finalize__

    def __new__(cls, buf, arr, regist=True):
        obj = arr.view(cls)
        _init(obj, buf, 1, regist)
        return obj

    def __setitem__(self, key, val):
        if isinstance(val, Number):
            numpy.ndarray.__setitem__(self, key, val)
        else:
            self.buf.copyto(val, self, key)

    def __iadd__(self, val):
        return self._op(numpy.ndarray.__iadd__, val)

    def __isub__(self, val):
        return self._op(numpy.ndarray.__isub__, val)

    def __imul__(self, val):
        return self._op(numpy.ndarray.__imul__, val)

    def __itruediv__(self, val):
        return self._op(numpy.ndarray.__itruediv__, val)

    def dot(self, x, out=None):
        return numpy.ndarray.dot(self, self.buf.forop(self.dev, x), out=out)


class BufFile(h5py.Dataset, BufArr):
    '''
    BufFile is a wrapper for h5py.Dataset with the usage of h5py.Dataset in BufPool traced.
    '''

    def __init__(self, buf, arr, regist=False, cb=lambda: None):
        _init(self, buf, 2, regist)
        self.arr = arr
        self.cb = cb

    def reshape(self, *args):
        arr = self.arr[:].reshape(*args)
        if isinstance(self.arr, h5py.Dataset):
            return BufFile(self.buf, arr,
                           cb=lambda: self.arr.__setitem__(numpy.s_[:], arr))
        else:
            return BufFile(self.buf, arr, cb=self.cb)

    def arrback(self, arr):
        if isinstance(self.arr, h5py.Dataset):
            self.arr[:] = arr
        else:
            self.cb()

    def readcheck(self):
        assert isinstance(self.arr, h5py.Dataset), \
            "Slice of BufFile is readonly!"

    def __getattr__(self, k):
        return getattr(self.arr, k)

    def __getitem__(self, key):
        arr = self.arr.__getitem__(key)
        return BufFile(self.buf, arr,
                       cb=lambda: self.arr.__setitem__(key, arr))

    def __setitem__(self, key, val):
        self.readcheck()
        if isinstance(val, Number):
            self.arr.__setitem__(key, val)
        else:
            self.buf.copyto(val, self.arr, key)
        self.cb()

    def __iadd__(self, val):
        self.readcheck()
        r = self._op(numpy.ndarray.__iadd__, val)
        self.cb()
        return r

    def __isub__(self, val):
        self.readcheck()
        r = self._op(numpy.ndarray.__isub__, val)
        self.cb()
        return r

    def __imul__(self, val):
        self.readcheck()
        r = self._op(numpy.ndarray.__imul__, val)
        self.cb()
        return r

    def __itruediv__(self, val):
        self.readcheck()
        r = self._op(numpy.ndarray.__itruediv__, val)
        self.cb()
        return r

    def enter(self, *arg, **kwgs):
        self.readcheck()
        BufArr.enter(self, *arg, **kwgs)

    def exit(self):
        if getattr(self, 'tmp', None) is not None:
            self.buf.copyto(self.tmp, self)
            cupy.cuda.get_current_stream().synchronize()
            del self.tmp
        self.cb()


class BufferPool():
    '''
    A pool to trace all arrays allocated.
    '''

    def __init__(self, gpulim, cpulim, path=param.TMPDIR,
                 verbose=10, mode='w', *args, **kwargs):
        '''
        gpulim: the maximum memory size of GPU in bytes.
        cpulim: the maximum memory size of CPU in bytes.
        path: the path to save the h5py file which serves as the file backend.
        mode: the mode of h5py.File.
        *args: the args of h5py.File.
        **kwargs: the kwargs of h5py.File.
        '''
        if os.path.isdir(path):
            path = tempfile.NamedTemporaryFile(dir=param.TMPDIR).name
        self.file = h5py.File(path, mode, *args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        maxgpulim = cupy.cuda.Device().mem_info[1]
        if cpulim is None:
            cpulim = 3 * maxgpulim
        if gpulim is None:
            gpulim = int(maxgpulim * 0.75)
        else:
            assert gpulim < maxgpulim, \
                'gpulim(%f) is larger than max gpu memory(%f)' \
                % (gpulim / 1e9, maxgpulim / 1e9)

        self.lim = [gpulim, cpulim, 100 * 1024 * 1024 * 1024 * 1024]
        self.free = [gpulim, cpulim, 100 * 1024 * 1024 * 1024 * 1024]
        self.stream = cupy.cuda.Stream(non_blocking=True)

        def alloc_gpu(shape, dtype, order):
            return BufCupy(
                self, empty(shape, dtype, order=order, type=MemoryTypeDevice))

        def alloc_cpu(shape, dtype, order):
            return BufNumpy(
                self, empty(shape, dtype, order=order, type=MemoryTypeHost))

        def alloc_file(shape, dtype, order):
            assert order in 'cC', 'only "C" order is supported'
            name = 'data'
            while name in self.file.keys():
                name = 'data' + '_' + str(random.randint(0, 9999))
            return BufFile(self, self.file.create_dataset(
                name, shape, dtype, *self.args, **self.kwargs))

        self.allocator = [alloc_gpu, alloc_cpu, alloc_file]
        log.verbose = verbose

    @property
    def free_memory(self):
        '''
        Return the free memory of GPU in bytes.
        '''
        return self.free[0]

    def alloc(self, dev, shape, dtype='f8', order='C'):
        '''
        Allocate a array on backend `dev`. `dev=0` means GPU, `dev=1` means CPU, `dev=2` means file.
        '''
        return self.allocator[dev](shape, dtype, order)

    def new(self, name, shape, dtype='f8', order='C', pin=0):
        '''
        Create a new array with name `name`. The backends are decided by
        `self.status`. If the `name` is not in `self.status`, the backend is
        tryied from `pin` to the slower backends.
        '''
        if name in self.status:
            return self.allocator[self.status[name]](shape, dtype, order)
        # unregist value
        r = None
        dev = pin
        nbytes = numpy.prod(shape) * numpy.dtype(dtype).itemsize
        while r is None and dev < 3:
            if nbytes <= self.free[dev]:
                try:
                    r = self.allocator[dev](shape, dtype, order)
                except cupy.cuda.memory.OutOfMemoryError:
                    pass
            dev += 1
        assert r is not None, "Cannot allocate %s" % name
        return r

    def add(self, name, data):
        '''
        Add and trace a array into buffer pool. No allocation is performed.
        '''
        if not isinstance(data, BufArr):
            if isinstance(data, cupy.ndarray):
                return BufCupy(self, data)
            elif isinstance(data, numpy.ndarray):
                return BufNumpy(self, data)
            else:
                assert isinstance(data, h5py.Dataset), \
                    'Unexpected type ' + str(type(data))
                return BufFile(self, data)
        else:
            return data

    def empty(self, shape, dtype='f8', order='C', type=MemoryTypeDevice):
        '''
        Keep the same interface as ByteQC.lib.empty.
        '''
        if type == MemoryTypeDevice:
            return self.alloc(0, shape, dtype, order)
        if type == MemoryTypeHost:
            return self.alloc(1, shape, dtype, order)
        else:
            return self.alloc(2, shape, dtype, order)

    def forop(self, dev, val, stream=None):
        if isinstance(val, Number):
            return val
        if isinstance(val, cupy.ndarray):
            if dev != 0:
                r = self.empty(val.shape, val.dtype, type=MemoryTypeHost)
                val.get(out=r, stream=stream)
                return r
            else:
                return val
        if dev == 0:
            r = self.empty(val.shape, val.dtype, type=MemoryTypeDevice)
            r.set(val[:], stream=stream)
            return r
        return val[:]

    def copyto(self, src, dest, key=slice(None), stream=None):
        i = getdev(src) * 3 + getdev(dest)
        if i == 0:  # gpu tp gpu
            if stream is not None:
                with stream:
                    cupy.ndarray.__setitem__(dest, key, src[:])
            else:
                cupy.ndarray.__setitem__(dest, key, src[:])
        elif i == 1:  # gpu to cpu
            if dest[key].flags.c_contiguous or dest[key].flags.f_contiguous:
                src.get(out=dest[key], stream=stream)
            else:
                r = self.empty(src.shape, src.dtype, type=MemoryTypeHost)
                src.get(out=r, stream=stream)
                if stream is not None:
                    stream.synchronize()
                numpy.ndarray.__setitem__(dest, key, r)
        elif i == 2:  # gpu to file
            r = self.empty(src.shape, src.dtype, type=MemoryTypeHost)
            src.get(out=r, stream=stream)
            if stream is not None:
                stream.synchronize()
            dest[key] = r
        elif i == 3 or i == 6:  # cpu/file to gpu
            if dest[key].flags.c_contiguous or dest[key].flags.f_contiguous:
                dest[key].set(src[:], stream=stream)
            else:
                tmp = self.empty(src.shape, src.dtype, type=MemoryTypeDevice)
                tmp.set(src[:], stream=stream)
                cupy.ndarray.__setitem__(dest, key, tmp)
        elif i == 4 or i == 5:  # cpu to cpu/file
            dest[key] = src
        elif i == 7:  # file to cpu
            numpy.ndarray.__setitem__(dest, key, src[:])
        else:  # file to file
            h5py.Dataset.__setitem__(dest, key, src[:])

    def asarray(self, arr):
        '''
        A cupy array will be allocated and returned. The data is copied from `arr`.
        '''
        r = self.alloc(0, arr.shape, arr.dtype)
        self.copyto(arr, r)
        return r

    def setverbose(self, verb):
        log.verbose = verb

    def memory_status(self, nocc, nvir, naux=None, mem_ratio=0.7,
                      which=0):
        '''
        Calculate the backends of the pre-registed tensors according to the current memory usage.
        '''
        free_all_blocks()
        r = pool_status()
        if r[0] != r[1]:
            warnings.warn(
                'There are about %.2fGB memory fragment before CCSD running. '
                'May cause out of memoey error!' % (r[1] - r[0]))
        gpu = self.free[0] / 8 * mem_ratio
        cpu = self.free[1] / 8
        if which == 0:  # ccsd
            gpu -= nocc * nvir * 2
            cpu -= - nocc**2 * nvir**2 * 2
            status = {'t1': 0, 't1new': 0}
            name = ['Loo', 'Lov', 'Lvv', 't2new', 't2', 'woooo', 'wVOov',
                    'wVooV', 'oooo', 'ovoo', 'ovov', 'oovv', 'ovvv']
            if naux is None:
                mem = [0, 0, 0] + [nocc**2 * nvir**2] * 2 + [nocc**4] \
                    + [nocc**2 * nvir**2] * 2 + [nocc**4, nocc**3 * nvir] \
                    + [nocc**2 * nvir**2] * 2 + [nvir**3 * nocc]
            else:
                mem = [nocc**2 * naux, nocc * nvir * naux, nvir**2 * naux] \
                    + [nocc**2 * nvir**2] * 2 + [nocc**4] \
                    + [nocc**2 * nvir**2] * 2 + [nocc**4, nocc**3 * nvir] \
                    + [nocc**2 * nvir**2] * 2 + [nvir**3 * nocc]
        elif which == 1:  # lambda
            gpu -= nocc * nvir * 2
            cpu -= - nocc**2 * nvir**2 * 2
            status = {'l1': 0, 'l1new': 0}
            name = ['Loo', 'Lov', 'Lvv', 'l2new', 'l2', 'woooo',
                    'wvooo', 'wVOov', 'wvOOv', 'wvvov', 'oooo', 'ovoo',
                    'ovov', 'oovv', 'ovvv']
            if naux is None:
                mem = [0, 0, 0] + [nocc**2 * nvir**2] * 2 \
                    + [nocc**4, nocc**3 * nvir] + [nocc**2 * nvir**2] * 2 \
                    + [nvir**3 * nocc, nocc**4, nocc**3 * nvir] \
                    + [nocc**2 * nvir**2] * 2 + [nvir**3 * nocc]
            else:
                mem = [nocc**2 * naux, nocc * nvir * naux, nvir**2 * naux] \
                    + [nocc**2 * nvir**2] * 2 + [nocc**4, nocc**3 * nvir] \
                    + [nocc**2 * nvir**2] * 2 \
                    + [nvir**3 * nocc, nocc**4, nocc**3 * nvir] \
                    + [nocc**2 * nvir**2] * 2 + [nvir**3 * nocc]
        elif which == 2:  # 1rdm
            status = {}
            name = ['theta']
            mem = [nocc**2 * nvir**2]
        else:  # 2rdm
            cpu -= - nocc**2 * nvir**2 * 2
            assert which == 3
            status = {}
            name = ['dvvvv', 'dovvo', 'mvOOv', 'mvoOV', 'doooo', 'dooov',
                    'doovv', 'dovov']
            mem = [0] * 8

        assert gpu >= 0, 'GPU memory free mem %d is too small for ' \
            'nocc,nvir:(%d,%d)' % (self.free[0], nocc, nvir)
        assert cpu >= 0, 'CPU memory free mem %d is too small for ' \
            'nocc,nvir:(%d,%d)' % (self.free[1], nocc, nvir)
        ng = bisect.bisect_right(numpy.cumsum(mem), gpu)
        for k in name[:ng]:
            status[k] = 0
        nc = bisect.bisect_right(numpy.cumsum(mem[ng:]), cpu) + ng
        for k in name[ng:nc]:
            status[k] = 1
        for k in name[nc:]:
            status[k] = 2
        if getattr(self, 'status', None) is None:
            self.status = status
        else:
            self.status.update(status)
        log.info('Memory status: %s', status)
        return status
