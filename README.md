# ByteDance Quantum Chemistry: ByteQC

ByteQC is a high-performance, GPU-accelerated quantum chemistry package designed for large-scale quantum chemistry simulations. It currently supports a range of methods, including mean-field calculations for both open and periodic boundary conditions, MP2 simulations, CCSD, and CCSD(T) calculations. All of these methods are optimized to support multiple GPUs, enabling efficient parallelization.
Additionally, ByteQC includes an integrated functionality for systematically improvable embedding (SIE), which allows for scalable simulations of complex systems. The package also exports several useful tools for the development of GPU-based quantum chemistry applications.

## External dependencies

This package incorporates parts of its code adapted from several external open-source projects:

* https://github.com/bytedance/gpu4pyscf
* https://github.com/hongzhouye/pyscf/tree/rsdf_direct
* https://github.com/pyscf/pyscf
* https://github.com/sunqm/libcint
* https://github.com/cupy/cupy
* https://github.com/BoothGroup/Vayesta

## Installation

Requirement:

GPU requirement: test on NVIDIA V100, A100, H100, and 4070Ti.

- python >= 3.11
- cupy (master branch in github or >=14.0)
- nvmath-python
- h5py
- mpi4py
- numpy
- scipy
- pyscf =2.5.0

Build dependencies:

- libcutensor >=2.1.0.9 (installation from conda is also supported)
- libcublas

Build the package by running the command `python byteqc/setup.py`.

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
@Article{Guo2025,
  author       = {Guo, Zhen and Huang, Zigeng and Chen, Qiaorui and Shao, Jiang and Liu, Guangcheng and Pham, Hung Q. and Huang, Yifei and Cao, Changsu and Chen, Ji and Lv, Dingshun},
  journal      = {{WIREs} Computational Molecular Science},
  title        = {{ByteQC}: {GPU}-Accelerated quantum chemistry package for large-scale systems},
  year         = {2025},
  note         = {e70034 CMS-1169.R1},
  number       = {3},
  pages        = {e70034},
  volume       = {15},
  doi          = {10.1002/wcms.70034},
  keywords     = {electronic structure, GPU-accelerated, large-scale, quantum chemistry simulation, quantum embedding},
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
