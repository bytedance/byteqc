# `cump2` subpackage

The cump2 subpackage provides GPU-accelerated MP2 calculations. It exports two functions: `DFMP2` and `DFKMP2`, which support both open boundary conditions (OBC) and periodic boundary conditions (PBC). For PBC, only gamma-point calculations are currently supported.
The cump2 subpackage also provides the gradient of OBC DFMP2 as the function `DFMP2_grad`. But for now `DFMP2_grad` is treated as an advanced feature which need `gpu4pyscf`. We recommend users to install `gpu4pyscf` by using the following command.
```shell
pip install --no-deps gpu4pyscf-cuda12x==1.3.1 pyscf==2.8.0 pyscf-dispersion==1.3.0 geometric==1.1 gpu4pyscf-libxc-cuda12x==0.6 networkx==3.4.2
```
Note that, the compatibility between `gpu4pyscf` and `byteqc` is not fully tested. If users find the functions do not work in `byteqc`. We recommend you to reset your environment by following the main document installation guide.

Examples can be found in the following files:
* `byteqc/cump2/tests/mp2.py``
* `byteqc/cump2/tests/mp2_gamma_point.py``
* `byteqc/cump2/tests/mp2_grad.py``
