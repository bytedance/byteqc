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

import h5py
import numpy
import os
import pickle
import shutil
import warnings
from multiprocessing import Pool
from numbers import Integral
from byteqc.lib.array import empty_from_buf, MemoryTypeHost, arr2ptr, ptr2arr
from byteqc.lib.utils import is_pinned
from itertools import product

# The number of threads to use when doing multiprocess I/O.
NumFileProcess = 8

# The function to determined the blksizes when not specificated.
default_blksizes = lambda filemp, name, shape, dtype, data, **kwds: None


def _wait(obj):
    if obj.waits is not None:
        for r in obj.waits:
            r.wait()
        obj.waits = None


def set_num_threads(num):
    '''Set the number of threads to use when doing multiprocess I/O.'''
    global NumFileProcess
    NumFileProcess = num


def readTask(path, para, src, dest, dataset='DatasetMp'):
    arr = ptr2arr(para)
    with h5py.File(path, 'r') as file:
        file[dataset].read_direct(arr, src, dest)


def writeTask(path, para, src, dest, dataset='DatasetMp'):
    arr = ptr2arr(para)
    with h5py.File(path, 'r+') as file:
        file[dataset].write_direct(arr, src, dest)


def errcb(e):
    '''Error callback for multiprocess I/O.'''
    print("IOMp error:", e)


def ioMp(pool, iotype, path, arr, src, dest, dataset='DatasetMp'):
    '''Multiprocess I/O.'''
    if iotype == 'read':
        r = pool.apply_async(
            readTask, (path, arr2ptr(arr), src, dest, dataset),
            error_callback=errcb)
    elif iotype == 'write':
        r = pool.apply_async(
            writeTask, (path, arr2ptr(arr), src, dest, dataset),
            error_callback=errcb)
    else:
        raise ValueError("Unknown iotype (%s)" % iotype)
    return r


class FutureNumpy(numpy.ndarray):
    '''
    A numpy array that waits for the multiprocess I/O to finish. Waitting operation will be triggered if get attributes which not in ['T', 'data', 'dtype', 'flags', 'size', 'itemsize', 'nbytes', 'ndim', 'shape', 'strides', 'ctypes', 'ravel', 'view', 'wait', 'waits'].
    '''
    wait = _wait

    def __new__(cls, arr, waits):
        obj = arr.view(cls)
        obj.waits = waits
        return obj

    def __repr__(self):
        self.wait()
        return super().__repr__()

    def __str__(self):
        self.wait()
        return super().__str__()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs_ = []
        for i in inputs:
            if isinstance(i, FutureNumpy):
                i.wait()
                inputs_.append(i.view(numpy.ndarray))
            else:
                inputs_.append(i)
        return super().__array_ufunc__(ufunc, method, *inputs_, **kwargs)

    def __getattribute__(self, name):
        if name not in ('T', 'data', 'dtype', 'flags', 'size', 'itemsize',
                        'nbytes', 'ndim', 'shape', 'strides', 'ctypes',
                        'ravel', 'view', 'wait', 'waits'):
            self.wait()
        return super().__getattribute__(name)

    def __getitem__(self, k):
        self.wait()
        return super().__getitem__(k)

    def __setitem__(self, k, v):
        self.wait()
        return super().__setitem__(k, v)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.waits = getattr(obj, 'waits', None)


def inttrunc(i, extent):
    return max(0, min(i if i >= 0 else i + extent, extent))


def indparse(keys, shape):
    r = [slice(0, s, 1) for s in shape]
    if not isinstance(keys, tuple):
        keys = (keys,)
    for i in range(len(keys)):
        k = keys[i]
        if isinstance(k, Integral):
            if k < 0:
                k += shape[i]
            assert k >= 0 and k < shape[i], 'Invalid indices!!!'
            r[i] = k
        elif isinstance(k, slice):
            if k.step is None:
                step = 1
            else:
                step = k.step
            if k.start is None:
                start = 0
            else:
                start = inttrunc(k.start, shape[i])
            if k.stop is None:
                stop = shape[i]
            else:
                stop = inttrunc(k.stop, shape[i])
            r[i] = slice(start, stop, step)
        else:
            assert len(set(k)) == len(k), "Duplicates indices not supported"
            r[i] = numpy.asarray([inttrunc(ik, shape[i]) for ik in k])
    return r


def inddiv(inds, blksizes):
    return [_inddiv(x, n) for (x, n) in zip(inds, blksizes)]


def _inddiv(x, n):
    if isinstance(x, Integral):
        return x // n
    elif isinstance(x, slice):
        if x.start is None:
            start = None
        else:
            start = x.start // n

        if x.stop is None:
            stop = None
        else:
            stop = (x.stop + n - 1) // n
        return slice(start, stop, x.step)
    else:
        return numpy.unique([i // n for i in x])


def indfmt(ind):
    return '/' + (('%d-' * len(ind)) % tuple(ind))[:-1]


def indshape(inds):
    r = [_indshape(i) for i in inds]
    real_shape = [i for i in r if i is not None]
    shape = [i if i is not None else 1 for i in r]
    return real_shape, shape


def _indshape(ind):
    if isinstance(ind, Integral):
        return None
    elif isinstance(ind, slice):
        step = abs(ind.step)
        return (abs(ind.stop - ind.start) + step - 1) // step
    else:
        return len(ind)


def indtrans(ranges, keys):
    r = [_indtrans(*r, k) for (r, k) in zip(ranges, keys)]
    return tuple(ir[0] for ir in r), tuple(ir[1] for ir in r)


def _indtrans(start0, stop, k):
    if isinstance(k, Integral):
        s_file = k - start0
        s_cpu = 0
    elif isinstance(k, slice):
        if k.step > 0:
            start = max(start0, k.start)
            stop = min(stop, k.stop)
            start += (start - k.start) % k.step
        else:
            start = min(start0, k.start)
            stop = max(stop, k.stop)
            start -= (start - k.start) % k.step
        s_file = slice(start - start0, stop - start0, k.step)
        extent = _indshape(s_file)
        offset = _indshape(slice(k.start, start, k.step))
        s_cpu = slice(offset, offset + extent, 1)
    else:
        s_file = k[(k >= start0) * (k < stop)] - start0
        k = list(k)
        s_cpu = numpy.asarray([k.index(i) for i in s_file])
    return s_file, s_cpu


def dispatchDataset(d):
    '''
    Dispatch dataset to DatasetMp or return the original dataset.
    '''
    if d.dtype.char == 'O' and d.shape == (
    ) and d[()][:15] == b'#!*DatasetMp*!#':
        return DatasetMp(d)
    else:
        return d


class DatasetMp():
    '''
    A wrapper for a dataset that is stored in multiple files. The interface
    is the same as `h5py.dataset`.
    '''
    wait = _wait

    def __init__(self, dataset, info=None):
        self.dataset = dataset
        self.dir = str(dataset[()][15:], 'utf-8')
        if info is None:
            with open(self.dir + '/meta.dat', 'rb') as f:
                info = pickle.load(f)
        self.info = info
        self.waits = None

    @property
    def dtype(self):
        return self.info[0]

    @property
    def super_shape(self):
        '''
        The shape of the tensor of dataset slices. 2x3x4 means the dataset is split into 2x3x4 slices.
        '''
        return self.info[1]

    @property
    def shape(self):
        return self.info[2]

    @property
    def shapes(self):
        return self.info[3]

    @property
    def inds(self):
        return self.info[4]

    @property
    def delims(self):
        return self.info[5]

    @property
    def size(self):
        return self.info[6]

    @property
    def nbytes(self):
        return self.dtype.itemsize * numpy.prod(self.shape)

    def __getattr__(self, k):
        return getattr(self.dataset, k)

    def __getitem__(self, k):
        return self.getitem(k)

    def getitem(self, k, pool=None, buf=None, iswrap=False):
        self.wait()
        ndim = len(self.shape)
        keys = indparse(k, self.shape)
        out_shape, ndimshape = indshape(keys)
        if buf is not None:
            assert is_pinned(buf), "buf must be pinned!"
        out = empty_from_buf(buf, out_shape, self.dtype, type=MemoryTypeHost)
        procs = []
        if pool is None:
            pool = Pool(processes=NumFileProcess)
        else:
            assert buf is not None, 'When `pool` is not None, ' \
                'the parameter `buf` must be specified! The user is ' \
                'responsible for ensuring that `buf` is pinned memory ' \
                'and that it is generated prior to initializing `pool`'
        super_keys = (numpy.unique(self.inds[i][keys[i]]) for i in range(ndim))
        for ind in product(*super_keys):
            r = ((self.delims[i][ind[i]:ind[i] + 2]) for i in range(ndim))
            src, dest = indtrans(r, keys)
            if iswrap:
                p = ioMp(pool, 'read', self.file.filename,
                         out.reshape(ndimshape), dest, dest, self.name)
            else:
                p = ioMp(pool, 'read', self.dir + indfmt(ind) + '.dat',
                         out.reshape(ndimshape), src, dest)
            procs.append(p)
        return FutureNumpy(out, procs)

    def __setitem__(self, k, v):
        waits = self.setitem(k, v)
        if self.waits is None:
            self.waits = waits
        else:
            self.waits.extend(waits)

    def setitem(self, k, v, pool=None):
        ndim = len(self.shape)
        keys = indparse(k, self.shape)
        v = numpy.asarray(v)
        intmask = list(not isinstance(ik, Integral) for ik in keys)
        nvdim = sum(intmask)
        if v.ndim < nvdim:
            vrealshape = list((*([1] * (nvdim - v.ndim)), *v.shape))
        else:
            vrealshape = list(v.shape)
        vshape = numpy.ones(ndim, dtype=numpy.int64)
        vshape[intmask] = vrealshape
        v = v.reshape(vshape)
        super_keys = (numpy.unique(self.inds[i][keys[i]]) for i in range(ndim))
        waits = []
        if pool is None:
            pool = Pool(processes=NumFileProcess)
        for ind in product(*super_keys):
            r = ((self.delims[i][ind[i]:ind[i] + 2]) for i in range(ndim))
            dest, src = indtrans(r, keys)
            # src = tuple(0 if v.shape[i] == 1 else src[i] for i in range(ndim))
            p = ioMp(pool, 'write', self.dir + indfmt(ind) + '.dat',
                    v, src, dest)
            waits.append(p)
        return waits


class DatasetMpWrapper(DatasetMp):
    '''
    A wrapper for a dataset that is stored in multiple files. The interface
    is the same as `h5py.dataset`.
    '''

    getitem = lambda *args: DatasetMp.getitem(*args, iswrap=True)

    def setitem(self, k, v):
        self.dataset[k] = v

    def __init__(self, dataset, blksizes=None):
        self.dataset = dataset
        self.blksizes = blksizes
        self.waits = None

        shape = dataset.shape
        if not isinstance(blksizes, tuple):
            blksizes = (blksizes, )
        blksizes = (*blksizes, *shape[len(blksizes):])
        inds, shapes, delims = parseblk(blksizes, shape)
        super_shape = tuple(len(i) - 1 for i in delims)
        self.info = (dataset.dtype, super_shape, shape, shapes, inds, delims,
                      dataset.nbytes)


class GroupMp(h5py.Group):
    '''
    A wrapper for a group that is stored in multiple files. The interface
    is the same as `h5py.group`.
    '''

    def __init__(self, filemp, group):
        self.filemp = filemp
        self.group = group

    def __getattr__(self, k):
        return getattr(self.group, k)

    def __getitem__(self, name):
        r = super().__getitem__(name)
        return dispatchDataset(r)

    def __setitem__(self, name, val):
        if name in self.group.keys():
            warnings.warn('Setting a dataset(%s) already exist!' % name)
            del self[name]
        return self.filemp.create_dataset(self.name + '/' + name, data=val)

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        return self.filemp.create_dataset(
            self.name + '/' + name, shape, dtype, data, **kwds)


def _parseblk(blk, ext):
    if isinstance(blk, Integral):
        n = (ext + blk - 1) // blk
        blk = numpy.full(n, blk)
        if n * blk[-1] != ext:
            blk[-1] = ext - (n - 1) * blk[-1]
    else:
        blk = numpy.asarray(blk)
    n = len(blk)
    ind = numpy.repeat(numpy.arange(n), blk)
    delim = numpy.empty(n + 1, dtype=numpy.int64)
    delim[0] = 0
    numpy.cumsum(blk, out=delim[1:])
    return ind, blk, delim


def parseblk(blks, exts):
    inds = []
    shapes = []
    delims = []
    for blk, ext in zip(blks, exts):
        r = _parseblk(blk, ext)
        inds.append(r[0])
        shapes.append(r[1])
        delims.append(r[2])
    return inds, shapes, delims


class FileMp(h5py.File):
    '''
    A wrapper for a file that is stored in multiple files. The interface
    is the same as `h5py.File`.
    '''

    def __getitem__(self, name):
        r = super().__getitem__(name)
        if isinstance(r, h5py.Group):
            return GroupMp(self, r)
        else:
            return dispatchDataset(r)

    def __delitem__(self, name):
        d = self[name]
        if isinstance(d, DatasetMp):
            d.wait()
            shutil.rmtree(d.dir)
            d = None
            super().__delitem__(name)
        else:
            super().__delitem__(name)

    def __setitem__(self, name, val):
        if name in self.keys():
            warnings.warn('Setting a dataset(%s) already exist!' % name)
            del self[name]
        return self.create_dataset(name, data=val)

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        blksizes = kwds.get('blksizes', default_blksizes(
            self, name, shape, dtype, data, **kwds))
        if blksizes is not None:
            if 'blksizes' in kwds:
                del kwds['blksizes']
            if data is not None:
                shape = data.shape
                dtype = data.dtype
            else:
                if dtype is None:
                    dtype = numpy.dtype('f8')
                else:
                    dtype = numpy.dtype(dtype)
            ndim = len(shape)
            size = numpy.prod(shape)
            if not isinstance(blksizes, tuple):
                blksizes = (blksizes, )
            blksizes = (*blksizes, *shape[len(blksizes):])
            inds, shapes, delims = parseblk(blksizes, shape)
            super_shape = tuple(len(i) - 1 for i in delims)

            root = self._id.name.decode()
            root = root.rsplit('/', 1)
            if len(root) == 2:
                root = root[0] + '/' + root[1].rsplit('.', 1)[0] + '_Mp/'
            else:
                root = root[0].rsplit('.', 1)[0] + '_Mp/'
            dir = root + name
            strdata = '#!*DatasetMp*!#' + dir
            dataset = h5py.Group.create_dataset(self, name, data=strdata)
            info = (dtype, super_shape, shape, shapes, inds, delims, size)
            if not os.path.isdir(dir):
                os.makedirs(dir)
            with open(dir + '/meta.dat', 'wb') as f:
                pickle.dump(info, f)

            if data is None:
                for i in numpy.ndindex(super_shape):
                    ishape = (shapes[j][i[j]] for j in range(ndim))
                    f = h5py.File(dir + indfmt(i) + '.dat', 'w')
                    f.create_dataset('DatasetMp', shape=ishape,
                                     dtype=dtype, **kwds)
                    f.close()
            else:
                for i in numpy.ndindex(super_shape):
                    idata = data[*
                                 (slice(
                                     delims[j][i[j]],
                                     delims[j][i[j] + 1]) for j in
                                  range(ndim))]
                    f = h5py.File(dir + indfmt(i) + '.dat', 'w')
                    f.create_dataset('DatasetMp', data=idata, ** kwds)
                    f.close()

            return DatasetMp(dataset)
        else:
            return h5py.Group.create_dataset(
                self, name, shape, dtype, data, **kwds)
