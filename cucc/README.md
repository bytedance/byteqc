# `cucc` subpackage

This package follows the interfaces in PySCF packages, and some codes are adapted from it to keep the same interface. The examples are in the `byteqc/cucc/test` folder.

The `buffer.py` and `culib.py` modules enable automatic backend determination. The arrays in `cucc` can be stored on the GPU, CPU, or disk, all with the same interface. Users and developers can work with these arrays without having to consider the underlying backend.
