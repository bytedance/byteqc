# ByteDance Quantum Chemistry: ByteQC

ByteQC is a high-performance, GPU-accelerated quantum chemistry package designed for large-scale quantum chemistry simulations. It currently supports a range of methods, including mean-field calculations for both open and periodic boundary conditions, MP2 simulations, CCSD, and CCSD(T) calculations. All of these methods are optimized to support multiple GPUs, enabling efficient parallelization.
Additionally, ByteQC includes an integrated functionality for systematically improvable embedding (SIE), which allows for scalable simulations of complex systems. The package also exports several useful tools for the development of GPU-based quantum chemistry applications.

## Exteranl dependencies

This package incorporates parts of its code adapted from several external open-source projects:

* https://github.com/bytedance/gpu4pyscf
* https://github.com/hongzhouye/pyscf/tree/rsdf_direct
* https://github.com/pyscf/pyscf
* https://github.com/sunqm/libcint
* https://github.com/cupy/cupy
* https://github.com/BoothGroup/Vayesta
* https://github.com/gkclab/libdmet_preview

## Installation

Requirement:

- cupy (master branch in github)
- nvmath-python
- h5py
- mpi4py
- numpy
- scipy
- pyscf = 2.5.0

Build dependencied:

- libcutensor (installation from conda is also supported)
- libcublas

Build the package by run command `python byteqc/setup.py`.

## Package structure

```plaintext
ByteQC
├── cucc          # CCSD and CCSD(T) solver
├── cump2         # MP2 solver
├── cupbc         # mean-field for PBC
├── cuobc         # mean-field for OBC
├── lib           # GPU utilities
├── embyte        # systematically improvable embedding
├── __init__.py
├── LICENSE
├── README.md
└── setup.py      # installation script
```

The `lib` subpackage is exported directly when importing ByteQC. The usage of each module is described in the README.md file in the corresponding directory.

## Citations

```latex
@Misc{Guo2025,
  author       = {Guo, Zhen and Huang, Zigeng and Chen, Qiaorui and Shao, Jiang and Liu, Guangcheng and Pham, Hung and Cao, Changsu and Chen, Ji and Lv, Dingshun},
  title        = {{ByteQC}: {GPU}-accelerated quantum chemistry package for large-scale systems},
  year         = {2025},
  doi          = {10.48550/ARXIV.2502.17963},
  publisher    = {arXiv},
}
```

If the SIE feature is used, please also cite:

```latex
@Misc{Huang2024,
  author       = {Huang, Zigeng and Guo, Zhen and Cao, Changsu and Pham, Hung Q. and Wen, Xuelan and Booth, George H. and Chen, Ji and Lv, Dingshun},
  title        = {Advancing surface chemistry with large-scale ab-initio quantum many-body simulations},
  year         = {2024},
  doi          = {10.48550/ARXIV.2412.18553},
  publisher    = {arXiv},
}
```
