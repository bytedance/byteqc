# EmByte
`EmByte` is a GPU-accelerated quantum chemistry software developed based on the [systematically improvable quantum embedding algorithm](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.12.011046), which is termed as SIE, combined with the high-performance quantum chemistry toolkit from ByteQC. In testing, CCSD(T)-level quantum chemistry calculations for systems with over 10,000 orbitals could be done on the machines with A100 GPUs with 80 GB of GPU memory.

## Feature
### Functions Supported
Basically now EmByte supported

1. Partition wavefunction form results, including the linear energy functional form and the reduced density matrix (RDM) form, are all supported, where the RDM form is more accurate due to the introduction of the inter-cluster information;
2. MP2/CCSD could be used as the high-level solver, and the special-designed SIE-suitable in-situ perturbative (T) calculation is also accessible if the solver is CCSD and finally push the calculation accuracy to the level of CCSD(T);
3. Supported flexible BNO threshold setting, which allows set different BNO thresholds for clusters or set more than 1 threshold for the testing purpose.

See the Quick Start in `./embyte/example/1_Quick_start_water_dimmer_test` for more details.

### PBC&OBC Supported

EmByte supports calculations on systems using open boundary conditions (OBC) or periodic boundary conditions (PBC). For OBC systems, all computations are performed on-the-fly. This means that aside from necessary checkpoint saving or data required for subsequent calculations (which are typically very small), all other variables are generated as needed and discarded after use. This approach greatly alleviates the pressure caused by I/O operations.
For PBC systems, however, Cholesky Decomposition Electron Repulsion Integrals (CDERI) must be precomputed and stored on disk. It will be repeatedly read during SIE calculation. While some asynchronous operations are utilized to prevent excessive I/O consumption, this overhead remains significant in extremely large systems. Efforts will be made in future versions to address this issue.

See the OBC example in `./embyte/example/1_Quick_start_water_dimmer_test`.

See the PBC example in `./embyte/example/2_PBC_example_water@BN`.

### MPI Parallel Acceleration
EmByte supports MPI parallelization based on `mpi4py`, which can be used to accelerate the calculation of large systems. Basically, after MPI being set, like the node information or others, the MPI parallel could be done like
```
mpirun -n $NODE python SIE_script.py
```
If you get multi GPUs in a node, you can directly use `mpirun` like
```
mpirun -np 2 \ # 2 is the number of the GPUs you want to use
    --bind-to none \
    -x CUDA_VISIBLE_DEVICES \
    bash -c 'export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK}; python SIE_script.py'
```
Note that for now, only the MPI with `openmpi` as the backend is supported.
### Checkpoint System
`EmByte` will by default record checkpoint to prevent losses from unexpected computation interruptions. If the program is accidentally halted, you can directly rerun the script without changing the script and logfile. The system will automatically resume the calculation from the most recent checkpoint prior to the interruption.

## Cite
If you use `EmByte` in your research, in addition to the `ByteQC` paper, please cite the following paper:

[1] Nusspickel M, Booth G H. Systematic improvability in quantum embedding for real materials[J]. *Physical Review X*, 2022, 12(1): 011046.

[2] Nusspickel M, Ibrahim B, Booth G H. Effective Reconstruction of Expectation Values from Ab Initio Quantum Embedding[J]. *Journal of Chemical Theory and Computation*, 2023, 19(10): 2769-2791.

[3] Huang Z, Guo Z, Cao C, et al. Advancing Surface Chemistry with Large-Scale Ab-Initio Quantum Many-Body Simulations[J]. *arXiv preprint* arXiv:2412.18553, 2024.