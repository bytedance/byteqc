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

import numpy
import cupy
import cupyx
import ctypes
from cupy_backends.cuda.libs import curand
from numbers import Integral


MemoryTypeUnregistered = 0  # pageable
MemoryTypeHost = 1  # pinned
MemoryTypeDevice = 2  # device
MemoryTypeManagedNumpy = 3  # managed and prefetched to cpu
MemoryTypeManagedCupy = 4  # managed and prefetched to gpu

CpuDeviceId = -1  # using device id -1 to represent cpu

# see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#querying-data-usage-attributes-on-managed-memory  # noqa: E501
AdviseSetReadMostly = 1
AdviseUnsetReadMostly = 2
AdviseSetPreferredLocation = 3
AdviseUnsetPreferredLocation = 4
AdviseSetAccessedBy = 5
AdviseUnsetAccessedBy = 6


def prefetch(a, device=None, stream=None):
    '''
    Prefetch the managed memeory. See CUDA documatation for more details.
    '''
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    if device is None:
        device = cupy.cuda.Device()
    if isinstance(device, cupy.cuda.Device):
        device = device.id
    cupy.cuda.runtime.memPrefetchAsync(a.ptr, a.size, device, stream.ptr)


def advise(a, advise, device=None, size=None, offset=0):
    '''
    Advise the behavior of managed memeory. See CUDA documatation for more details.
    '''
    if not isinstance(a, cupy.cuda.ManagedMemory):
        a = a.mem
    if size is None:
        size = a.size - offset
    if device is None:
        device = cupy.cuda.Device()
    if isinstance(device, cupy.cuda.Device):
        device = device.id
    cupy.cuda.runtime.memAdvise(a.ptr + offset, size, advise, device)


def __array_finalize__(self, obj):
    if obj is None:
        return
    self.mem = getattr(obj, 'mem', None)
    self.offset = getattr(obj, 'offset', 0)


class ManagedNumpy(numpy.ndarray):
    '''
    A subclass of `numpy.ndarray` with the managed memory as the backend
    '''
    __array_finalize__ = __array_finalize__

    def __new__(subtype, shape, dtype='f8', mem=None, offset=0,
                strides=None, order=None, isprefetch=True):
        if mem is None:
            size = numpy.prod(shape) * numpy.dtype(dtype).itemsize
            mem = cupy.cuda.ManagedMemory(size=size)
            advise(mem, AdviseSetAccessedBy, CpuDeviceId)
            advise(mem, AdviseSetPreferredLocation, CpuDeviceId)
            if isprefetch:
                prefetch(mem, CpuDeviceId)
        else:
            assert isinstance(
                mem, cupy.cuda.ManagedMemory), "mem should be a ManagedMemory"
        ptr = ctypes.cast(mem.ptr, ctypes.POINTER(
            numpy.ctypeslib.as_ctypes_type(dtype)))
        arr = numpy.ctypeslib.as_array(ptr, shape=shape)
        obj = super().__new__(subtype, shape, dtype,
                              arr.data, offset, strides, order)
        obj.mem = mem
        obj.offset = offset
        return obj

    def tocupy(self):
        '''
        Convert to `ManagedCupy` array.
        '''
        return ManagedCupy(
            self.shape, self.dtype, self.mem, self.offset, self.strides)


class ManagedCupy(cupy.ndarray):
    '''
    A subclass of `cupy.ndarray` with the managed memory as the backend.
    '''
    __array_finalize__ = __array_finalize__

    def __new__(cls, shape, dtype='f8', mem=None, offset=0, strides=None,
                order='C', isprefetch=None):
        if mem is None:
            size = numpy.prod(shape) * numpy.dtype(dtype).itemsize
            mem = cupy.cuda.ManagedMemory(size=size)
            advise(mem, AdviseSetAccessedBy)
            advise(mem, AdviseSetPreferredLocation)
            if isprefetch is None:
                isprefetch = size <= cupy.cuda.Device().mem_info[0]
            if isprefetch:
                prefetch(mem)
        else:
            assert isinstance(
                mem, cupy.cuda.ManagedMemory), "mem should be a ManagedMemory"
        obj = super().__new__(
            cls, shape, dtype, cupy.cuda.MemoryPointer(mem, offset),
            strides, order)
        obj.mem = mem
        obj.offset = offset
        return obj

    def tonumpy(self):
        '''
        Convert to `ManagedNumpy` array.
        '''
        return ManagedNumpy(
            self.shape, self.dtype, self.mem, self.offset, self.strides)


def fillrand(x):
    '''
    Fill a cupy array `x` with random numbers with uniformed distributed.
    '''
    rs = cupy.random._generator.get_random_state()
    if x.dtype.char == 'f':
        func = curand.generateUniform
        c = 1
    elif x.dtype.char == 'd':
        func = curand.generateUniformDouble
        c = 1
    elif x.dtype.char == 'F':
        func = curand.generateUniform
        c = 2
    elif x.dtype.char == 'D':
        func = curand.generateUniformDouble
        c = 2
    else:
        raise TypeError("Unsupported type %s" % str(x.dtype))
    func(rs._generator, x.data.ptr, x.size * c)
    return x


def empty_pinned(shape, dtype='f8', order='C'):
    '''
    Create a pinned numpy array without the pinned_memory pool of `cupy`.
    '''
    nbytes = numpy.prod(shape) * numpy.dtype(dtype).itemsize
    nbytes = (nbytes + 1023) // 1024 * 1024
    mem = cupy.cuda.pinned_memory._malloc(nbytes)
    return numpy.ndarray(shape, dtype=dtype, buffer=mem, order=order)


def empty(shape, dtype='f8', order='C', type=MemoryTypeDevice,
          pool=False):
    '''
    A unitfy function to create a array.

    type:
        MemoryTypeDevice: create a cupy array
        MemoryTypeHost: create a pinned numpy array
        MemoryTypeUnregistered: create a pageable numpy array
        MemoryTypeManagedNumpy: create a ManagedNumpy array
        MemoryTypeManagedCupy: create a ManagedCupy array
    pool: decide wether to use the default pinned memory pool when `type=MemoryTypeHost`.
    '''
    if type == MemoryTypeUnregistered:
        return numpy.empty(shape, dtype, order=order)
    if type == MemoryTypeHost:
        if not pool:
            return empty_pinned(shape, dtype, order=order)
        return cupyx.empty_pinned(shape, dtype, order=order)
    if type == MemoryTypeDevice:
        return cupy.empty(shape, dtype, order=order)
    if type == MemoryTypeManagedNumpy:
        return ManagedNumpy(shape, dtype=dtype, order=order)
    if type == MemoryTypeManagedCupy:
        return ManagedCupy(shape, dtype=dtype, order=order)
    raise ValueError("Unknown type(%d)" % type)


def empty_from_buf(x, shape, dtype=None, order='C', strides=None,
                   type=MemoryTypeDevice, pool=False):
    '''
    Create a array with the memory of `x`. Fallback to `empty` if `x` is None.
    '''
    if x is None:
        return empty(shape, dtype, order, type)
    if dtype is None:
        dtype = x.dtype
    assert x.nbytes >= numpy.prod(shape) * \
        numpy.dtype(dtype).itemsize, "buf is not enough"
    if isinstance(x, numpy.ndarray):
        return numpy.ndarray(shape, dtype=dtype, order=order,
                             strides=strides, buffer=x.data)
    if isinstance(x, cupy.ndarray):
        return cupy.ndarray(shape, dtype=dtype, order=order,
                            strides=strides, memptr=x.data)
    raise ValueError("%s is not support" % (str(x.__class__)))


class ArrayBuffer:
    '''
    A buffer object to allocate small arrays from a large array.
    '''

    def __init__(self, mem=0, type=MemoryTypeDevice, pool=False, nblk=10):
        '''
        mem: can be a array used as buffer or a integer of the size of the
             buffer in bytes.
        type: when `mem` is a integer, determine the type of the buffer.
        pool: determine wether pinned memory pool is used when
              `type=MemoryTypeHost`.
        nblk: the max number of blocks to allocate. the memory actually
              allocated is (mem + 1023) // 1024 * 1024 + nblk * 1024.
        '''
        if isinstance(mem, Integral):
            mem = (mem + 1023) // 1024 * 1024 + nblk * 1024
            arr = empty((mem,), type=type, dtype=numpy.int8, pool=pool)
        elif isinstance(mem, cupy.ndarray):
            arr = mem.view(dtype=cupy.int8)
            type = MemoryTypeDevice
        elif isinstance(mem, numpy.ndarray):
            arr = mem.view(dtype=numpy.int8)
            from byteqc.lib.utils import is_pinned
            if is_pinned(mem):
                type = MemoryTypeHost
            else:
                type = MemoryTypeUnregistered
        else:
            raise ValueError("Unsupport inpput `mem`")
        self.type = type
        self.arr = arr
        self.offset = 0
        self.bufsize = arr.nbytes
        self.tags = dict()

    def empty(self, shape, dtype='f8'):
        '''
        Allocate a array from the buffer.
        '''
        size = numpy.prod(shape) * numpy.dtype(dtype).itemsize
        size = (size + 1023) // 1024 * 1024
        self.check(size)
        if self.type < MemoryTypeDevice:
            r = numpy.ndarray(shape, dtype=dtype,
                              buffer=self.arr, offset=self.offset)
        else:
            r = cupy.ndarray(shape, dtype=dtype,
                             memptr=self.arr.data + self.offset)
        self.offset = self.offset + size
        self.bufsize -= size
        return r

    def asarray(self, arr):
        '''
        Allocate a array from the buffer and copy the data from `arr`. Only avaliable when `type=MemoryTypeDevice`.
        '''
        if self.type != MemoryTypeDevice:
            raise AttributeError(
                "CPU array do not have this attribute (asarray)")
        r = self.empty(arr.shape, arr.dtype)
        r.set(arr)
        return r

    @property
    def nbytes(self):
        '''
        The number of total bytes in the buffer.
        '''
        return self.arr.nbytes

    @property
    def data(self):
        '''
        The current data pointer of the buffer. A `MemoryPointer` object is
        returned for cuda array and a `memoryview` object is returned for cpu
        array.
        '''
        if self.type == MemoryTypeDevice:
            return self.arr.data + self.offset
        else:
            return self.arr.data[self.offset:].data

    def left(self):
        '''
        Return a array formed by all left bytes in the buffer.
        '''
        return self.empty(((self.bufsize - 1024) // 8,))

    def check(self, nbytes):
        '''
        Check if the buffer has enough space to allocate a array of `nbytes`.
        '''
        assert self.bufsize >= nbytes, "Buffer size %d not enough for %d!!!" \
            % (self.bufsize, nbytes)

    def tag(self, tag='tmp'):
        '''
        Tag the current position of the buffer.
        '''
        self.tags[tag] = self.offset, self.bufsize

    def untag(self, tag='tmp'):
        '''
        Load and delete the current position of the buffer.
        '''
        self.offset, self.bufsize = self.tags[tag]
        del self.tags[tag]

    def loadtag(self, tag='tmp'):
        '''
        Load the current position of the buffer.
        '''
        self.offset, self.bufsize = self.tags[tag]

    def __broad_as__(self):
        assert self.offset == 0 and not self.tags, "Cannot broadcast a " \
            "ArrayBuffer which has already be used!!"
        arr = cupy.asarray(self.arr)
        return arr, ArrayBuffer(arr, type=MemoryTypeDevice)

    def __broad_new__(self):
        arr = cupy.empty(self.arr.shape, dtype=self.arr.dtype)
        wrapper = ArrayBuffer(arr)
        return arr, wrapper


def arr2ptr(arr):
    '''
    Convert a numpy.ndarray to a tuple of MemoryPointer, shape, dtype and strides. Used for multiprocessing.
    '''
    assert isinstance(arr, numpy.ndarray), "Only numpy.ndarray is supported"
    return arr.ctypes.data, arr.shape, arr.dtype, arr.strides


def ptr2arr(para):
    '''
    Reconstruct a numpy.ndarray from a tuple of MemoryPointer, shape, dtype and strides. Used for multiprocessing.
    '''
    data, shape, dtype, strides = para
    ptr = ctypes.cast(data, ctypes.POINTER(
        numpy.ctypeslib.as_ctypes_type(dtype) * numpy.prod(shape)))
    return numpy.ndarray(shape, dtype, strides=strides, buffer=ptr.contents)
