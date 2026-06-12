# `lib` subpackage

This subpackage contains some utilities for GPU coding. All interfaces are exported in the `__init__.py` file, thus can be called directly from `lib` subpackage, e.g. `lib.empty`.
Some functions are adapted from the <https://github.com/cupy/cupy> to extend its features.

## Array related - `array.py`

### Array allocation

* `empty` is the uniform interface of `numpy.empty` and `cupy.empty`, and the returned type is determined by the `type` parameters valued as:

```python
MemoryTypeUnregistered = 0  # pageable
MemoryTypeHost = 1  # pinned
MemoryTypeDevice = 2  # device
MemoryTypeManagedNumpy = 3  # managed and prefetched to cpu
MemoryTypeManagedCupy = 4  # managed and prefetched to gpu
```

* `emptyfrombuf` is similar as `empty`. But has an additional parameter as the first parameter `x`, which can be `None` and falls back to `empty`. If it is a `ndarray` it will use this array to create a new array without allocations. If `x` is not big enough, an error will be raised.

* `fillrand` is used to fill an existing `cupy` array with random values. It is similar to the `cupy.random.rand`, but without allocation and support complex numbers.

* `ArrayBuffer` class is designed to preallocate a large `numpy`/`cupy` array as the buffer. `ArrayBuffer.empty` for allocation and `ArrayBuffer.asarray` for allocation of a `cupy` array and copy a `numpy` array to it. `ArrayBuffer.tag` and `ArrayBuffer.loadtag` can save/load the current pointer of the buffer. `ArrayBuffer.untag` will delete the saved tag after load it.

### Managed memory related

Two classes `ManagedNumpy` and `ManagedCupy` are defined, they are subclass of `numpy.ndarray` and `cupy.ndarray` respectively but with managed memory as the backends. These two classes can be transformed to each other without change the backend memory: `ManagedNumpy.tocupy` and `ManagedCupy.tonumpy`.

Functions `prefetch` and `advise` are used to change the behavior of the managed memory, more information can be found in the [CUDA documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-programming).

## File related - `file.py`

This is a multiprocess extension to the `h5py` class. To enable it, user should first open a file as `FileMp` instead of `h5py`. When creating a database, using the `blksizes` parameter to specify the block size and thus enable the multiprocess. Each block is stored as individual files and their read/write is done in parallel. If `blksizes` is not specified, it falls back to the original `h5py.create_dataset`. For example:

```python
import ByteQC
f = ByteQC.lib.FileMp('filename.dat', 'w')
a = f.create_dataset('name', (20,20,20), blksizes=(10, 15))  # a is sliced into 2*2*1=4 slices.
a[:10,:15] = 1  # this will not trigger multiprocess writing because it only involved one block
print(a[:10])  # this will trigger multiprocess reading because it involves two blocks
```

Function `set_num_threads` is used to set the number of threads used in the multiprocess. It only affects the operations after it is called.

Slicing `h5py.DataSet` will return a `numpy` array, while here a `FutureNumpy` is returned and the i/o operation may be not done when it is returned, that is, slicing operation is asynchronized. `FutureNumpy` is subclass of `numpy.ndarray`, and all attributes that do not rely on the data itself will behave the same as `numpy.ndarray`, while others will first wait the i/o operation to be complete. Thus there is no need to change the code for this feature, but will affect the timing logic.

## Linalg related - `linalg.py`

This file exports several linear algebra functions:

* `contraction` will automatically check whether `numpy.ndarray` is passed in. If `numpy.ndarray`s' size is less than `MG_NBYTES_THRESHOLD` bytes (default `1GB`) for `a` and `b`, they are transferred to GPU automatically. After that, if any tensor is still on cpu, the `cutensorMg` is used (one can also set `isforceMg=True` to force using `cutensorMg`). Otherwise, the `cutensor` is used. For contraction between real numbers and complex numbers, there is a trick to enhance the performance: reparse the complex array as a real one with additional axes with extent `2`. This trick is enabled by default and can be closed by setting `isskipc2r=True`. `DEFAULT_WS_HOST` is the default value of parameter `ws_host`.

* `elementwise_binary` calculates the `out = opac(alpha * opa(A), gamma * opc(C))` using cuTENSOR library.

* `elementwise_trinary` calculates the `out = opabc(opab(alpha * opa(A), beta * opb(B)), gamma * opc(C))` using cuTENSOR library.

* `gemm` will choose the 64-bit or 32-bit backends automatically for different input size. It calls `cublas` by normal case, but will fall back to `contraction` if `cublas` not support such multiplication. Such case include:

  * `numpy` array is passed in;
  * arrays passed in are not contiguous;
  * arrays passed in have different types. (This by default will trigger the complex-to-real trick mentioned above.)

* `svd` calls the 64-bit `cusolver.xgesvd` backend, and supports all the features of CUDA level functions. No cpu array is allowed.

* `solve_triangle` solves the equation `op(a) x = alpha * b` for `x` if `left=True` and `x op(a) = alpha * b` if `left=False`.

* `cholesky` performs the Cholesky decomposition and can modify input array in-place with `overwrite=True`.

* `scal` scales an array by a scalar in-place. For cpu arrays it is much faster than the naive implementation `arr *= c`. For gpu arrays the performance is similar.

* `copy` copies array `x` to array `y`.  For cpu arrays it is much faster than the naive implementation `y[:] = x`. For gpu arrays the performance is similar.

* `axpy`: in-place modify `yarr` as `a*xarr+yarr`.  It is much faster than the naive implementation `yarr[:] += a*xarr`. No extra memory is needed for `axpy` while naive implementation demanding an extra memory of size of `xarr`

* `swap` in-place swaps `xarr` and `yarr`. It changes the memory while `x, y = y, x` only swaps the pointers.

## Multigpu related - `multigpu.py`

The only bindings exported by this file are the `Mg` object. The backend communicator is initialized at first used.

`Mg.set_gpu(gpus)` will (re)init the `Mg`. `gpus` can be `None` for all gpus, an int `n` for first `n` gpus, and a list of integers specifying which gpu to use.

`Mg.getgid()` will return the current gpu index in the `Mg.gpus` instead of the gpu id. Because `Mg.gpus` can be incontiguous or not in ascending order.

`Mg.sum` and `Mg.reduce` both sum over the arrays in different gpus. The difference is all arrays are equal to the summation for `Mg.sum` while only the `root`th gpu in `Mg.gpus` equals to the summation for `Mg.reduce`. The former can use the ring-broadcast and has better performance. But if arrays not in the `root`th gpu should not be modified, `Mg.reduce` should be used.

The function used to `sum` or `reduce` must be in the following form:

```python
def task(r, ...):
  if r is None:
    r = cupy.zeros((n, n))
  ...
  return r
```

`Mg.map` map a function into various parameters and return a list, which contain the results for all parameters. Which parameter calculated on which gpu is random. The parameter is automatically broadcast unless the parameter length cannot be determined (in such case `length` parameter should be passed in). e.g.

```python
Mg.map(f, range(10), 3)  # return [f(0, 3), f(1, 3),... f(9, 3)]
Mg.map(f, range(10), 3, length=2)  # return [f(0, 3), f(1, 3)]
```

`Mg.broadcast` broadcasts a cpu/gpu array into all gpus.

`Mg.mapgpu` is similar to `Mg.map`, but the `length` is fixed to the number of gpus and `i`th parameter is calculated on `i`th gpu in `Mg.gpus`.

## Other utils - `utils.py`

Here exports `is_pinned`, `gpu_avail_bytes`, `gpu_used_bytes`, `free_all_blocks`, `pool_status`, `pack_tril`, `unpack_tril`, and `hasnan` with the detailed explanation in the doc strings.
