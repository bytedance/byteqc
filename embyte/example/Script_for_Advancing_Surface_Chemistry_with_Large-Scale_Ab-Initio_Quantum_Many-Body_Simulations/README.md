# Guide for installation

This guide provides a comprehensive walkthrough starting from environment configuration to properly run all script files in this directory for the scriptes of the paper "Advancing Surface Chemistry with Large-Scale Ab-Initio Quantum Many-Body Simulations". Readers are encouraged to patiently follow this guide step-by-step to complete all required setups.

## System information
The package of ByteQC is running on Linux type system `Debian 12.9 bookworm`, which we used for coding and testing with the information as following.
```
Distributor ID: Debian
Description:    Debian GNU/Linux 12 (bookworm)
Release:        12
Codename:       bookworm
```
The code for GPU part is running on NVIDIA GPUs with the hardware information as following.
```
GPU:            NVIDIA A100-SXM4-80GB
CPU:            Intel(R) Xeon(R) Platinum 8336C CPU @ 2.30GHz, 14 cores
Memory:         245 GB
```

Please note that the testing environment requires CUDA 12.6. According to NVIDIA's official documentation, your NVIDIA driver version must be greater than 535. You can verify this by using the command `nvidia-smi`. The result on our machine is as follows. If your NVIDIA driver version is lower than 535, please manually upgrade the NVIDIA driver. Learn more about the NVIDIA driver upgrading in Linux from this [link](https://www.nvidia.com/en-in/drivers/), and learn more about the CUDA compatibility in this [link](https://docs.nvidia.com/deploy/cuda-compatibility/).
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.08             Driver Version: 535.161.08   CUDA Version: 12.6     |
|-----------------------------------------+----------------------+----------------------+
```

The code for CPU part, like some Hartree-Fock calculation. is running on the hardware with the information as following.
```
CPU:            Intel(R) Xeon(R) Platinum 8336C CPU @ 2.30GHz, 96 cores
Memory:         980 GB
```
**Please Note:**

1. The configurations listed above are highly recommended. If your system does not meet these specifications, the program can still run, but you may encounter **out-of-memory (OOM)** errors in large systems.
2. About the disk, running script for the largest system at the largest basis will consume over 5 TB disk space and 4 TB of the space is taken for saving ERI into one file. Therefore, a disk with 8 TB is recommended if you want to fully rerun the code. If you just want to run the code for the small system, the requirement for the disk will decrease to around 1 TB.

## Environment configuration

### Basic environment setting

This part will take about 10 minutes but the time cost will highly depends on the internet downloading speed. at the begainning, we need to set some environment variables to make sure all installation goes smoothly. Please add the following lines to the end of your `~/.bashrc` file.
```bash
# For CUDA
export CUDA_HOME="/usr/local/cuda"
export CUDA_PATH="/usr/local/cuda"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH}"
# For conda and CUDA
export PATH="${HOME}/miniconda3/bin:${CUDA_HOME}/bin:${PATH}"
# For cupy
export CUDA_CACHE_MAXSIZE=4294967296
```

Then reopen your terminal or running `source ~/.bashrc` to refresh.

Some basic package we recommand to install as following.
```bash
sudo apt-get update
sudo apt-get -y install \
    wget \
    gnupg2 \
    curl \
    python3-requests \
    ca-certificates \
    apt-transport-https \
    libncursesw5 \
    libtinfo5 \
    openssh-client \
    openssh-server \
    python3-requests \
    htop \
    build-essential \
    ssh \
    cmake \
    psmisc \
    nvtop \
    openmpi-bin
```

### Install CUDA

The whole installation CUDA will take about 20 minutes but the time cost will highly depends on the internet downloading speed. Then we can install CUDA with the following command.
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -yq --no-install-recommends --fix-missing \
    cuda-cudart-12-6=12.6.77-1 \
    cuda-compat-12-6 \
    cuda-libraries-12-6=12.6.2-1 \
    cuda-libraries-dev-12-6=12.6.2-1 \
    cuda-nvtx-12-6=12.6.77-1 \
    cuda-nvml-dev-12-6=12.6.77-1 \
    cuda-command-line-tools-12-6=12.6.2-1 \
    cuda-minimal-build-12-6=12.6.2-1 \
    libcublas-12-6=12.6.3.3-1 \
    libcublas-dev-12-6=12.6.3.3-1 \
    libcusparse-12-6=12.5.4.2-1 \
    libcusparse-dev-12-6=12.5.4.2-1 \
    libnccl2=2.23.4-1+cuda12.6 \
    libnccl-dev=2.23.4-1+cuda12.6 \
    libcutensor-dev=2.1.0.9-1
```

After that, some file should be instead to make sure  the CUDA 12.6 is compatible with your system.
```bash
sudo ln -s /usr/local/cuda-12.6 /usr/local/cuda
sudo ln -sf /usr/local/cuda/compat/libcuda.so /lib/x86_64-linux-gnu/
sudo ln -sf /usr/local/cuda/compat/libcuda.so.1 /lib/x86_64-linux-gnu/
```

### Install `conda`
We highly recommand user to set the environment with the help of `conda`. Therefore, we will install `miniconda` by following the [official guide](https://docs.anaconda.com/miniconda/install/). The whole installation process will take about 20 minutes.
```bash
mkdir -p $HOME/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-py312_25.1.1-2-Linux-x86_64.sh -O $HOME/miniconda3/miniconda.sh
bash $HOME/miniconda3/miniconda.sh -b -u -p $HOME/miniconda3
source ~/.bashrc # or reopen the terminal
```

Basically the `conda` environment is ready. Using the following commands to check if the `conda` environment is working properly and initialize `conda`.
```bash
which conda
# expected output: $HOME/miniconda3/bin/conda
conda init --all
```

Some common package will installed in the default virtual environment named `base` by following commands.
```bash
conda install -y -c conda-forge \
        python \
        "libblas=*=*mkl" \
        openmpi \
        flake8 \
        autopep8 \
        mpi4py \
        pynvml \
        numpy=1.26.4 \
        h5py \
        psutil \
        pyscf=2.5.0 \
        scipy \
        fastrlock \
        nvmath-python \
        pandas \
        addict \
        git
```

### Install `cupy` with version 14.0.0a1
The `cupy` will be installed from source to support `cutensormg` and achieve the best performance. And whole process will take about 10-20 minutes.
```bash
mkdir $HOME/cupy
cd $HOME/cupy
git clone --depth=1 -b main https://github.com/cupy/cupy.git
cd cupy
git submodule update --init
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}"
PIP_INDEX_URL=https://pypi.org/simple/ $HOME/miniconda3/bin/pip install .
```

Reopen the terminal and run the following command to make sure the `cupy` is installed successfully.
```bash
python -c "import cupy;print(cupy.__version__);print(cupy.__file__)"
# The expect output:
# 14.0.0a1
# $HOME/miniconda3/lib/python3.12/site-packages/cupy/__init__.py
```

### Install `byteqc`
The `byteqc` will be installed from source. And whole process will take about 10-30 minutes.
```bash
mkdir $HOME/byteqc
cd $HOME/byteqc
git clone https://github.com/bytedance/byteqc.git
cd byteqc
python setup.py
```
Open the ~/.bashrc file and add the following lines to the end of the file to make sure the `byteqc` could be import in python:
```bash
export PYTHONPATH="${HOME}/byteqc/:${PYTHONPATH}"
```
Then reopen the terminal and run following command to make sure the `byteqc` is installed successfully.
```bash
python -c "import byteqc;print(byteqc.__file__)"
# expect output: $HOME/byteqc/byteqc/__init__.py
```

## Simple test code

Using python to test this simple example to make sure the installation is correct.
```python
from byteqc.cuobc import scf as gscf
from pyscf import gto
from pyscf import scf as cscf

mol = gto.Mole()
mol.atom = [
    ['O', [0.00000000, 0.00000000, 0.58589194]],
    ['H', [0.00000000, 0.75668992, 0.00000000]],
    ['H', [0.00000000, -0.75668992, 0.00000000]],
]
mol.basis = 'cc-pVDZ'
mol.build()
mf_GPU = gscf.RHF(mol)
mf_GPU.kernel()
mf_CPU = cscf.RHF(mol)
mf_CPU.kernel()
print(abs(mf_GPU.e_tot - mf_CPU.e_tot))
# expect output to be ~1e-12
```

# The demo case: water@graphene

## Structure information
The demo case is the energy calcualtion for water@graphene with the scripts in `byteqc/embyte/example/Script_for_Advancing_Surface_Chemistry_with_Large-Scale_Ab-Initio_Quantum_Many-Body_Simulations/water@graphene/1_E_int_calculation`. The test systems under OBC and PBC information as the following list:

| | OBC | PBC |
| :---: | :---:|  :---:| 
| Water Configuration| 2-leg | 2-leg |
| Substrate | PAH(2) | Graphene(4x4x1) |
| Basis | ccECP-cc-pVDZ | ccECP-cc-pVDZ |
| Structure type | Full | Full |

where the structure type "Full" means there is no ghost atoms exsits in structure. The scripts are split into several files for different methods, as the following table:
| | OBC | PBC |
| :---: | :---:|  :---:|
| HF | `1-1_HF_OBC.py` | `1-2_HF_PBC.py`  |
| Canonical MP2 | `2-1_MP2_OBC.py` | `2-2_MP2_PBC.py`  |
| SIE+MP2/CCSD(T) | `3-1_SIE_OBC.py` | `3-2_SIE_PBC.py`  |

## How to run the scripts

1. All scripts can be executed directly using the Python command, and the results will be generated in the folder where the script files are located. 
2. Please note that the calculations for HF, MP2, and SIE should be performed in sequence, as the HF calculation will generate the necessary files for the MP2 and SIE calculations. For example, for OBC calculations, you should execute `1-1_HF_OBC.py`, `2-1_MP2_OBC.py`, and `3-1_SIE_OBC.py` in order, and the same applies to PBC calculations. 
3. To switch between SIE+MP2 and SIE+CCSD(T) calculations, you need to modify the `if_MP2` variable in the SIE script. For instance, in OBC, this variable is located at line 244 in `3-1_SIE_OBC.py`, while in PBC, it is at line 199 in `3-2_SIE_PBC.py`. When `if_MP2 = False`, the SIE+CCSD(T) calculation will be performed, and when `if_MP2 = True`, the SIE+MP2 calculation will be executed.

## Time cost
All the time cost is test based on the hardware we list in "System information" at very begainning. 

Note that, the PBC HF calculation is excuted on CPU, and the other calculation is excuted on GPU without statement. The use of CPU for PBC HF calculations here is not because ByteQC does not support GPU-based PBC HF calculations. Instead, it is because ByteQC employs a specialized density fitting method to accelerate integral computation, which may cause the final result to slightly deviate from the CPU HF result we obtained in the early stage of data collection. To ensure consistency in the results, we have opted to use the CPU version of PySCF for all PBC HF calculations.

| System | HF | MP2 | SIE+MP2 | SIE+CCSD(T) |
| :---: | :---:| :---: | :---: | :---: |
| OBC | 4m8s | 0m14s | 1m45s | 34m32s |
| PBC | 2m56s (on CPU) | 0m9s | 2m7s | 421m33s |

## Expect output

The total energy for the demo cases are list in the following table. All this result could be found in their logfile.

| Method | Logfile name | Total energy for OBC | Total energy for PBC |
| :---: | :---:| :---: | :---: |
| HF | `HF.log` | -156.08350596751 | -187.20912914967 |
| MP2 | `MP2.log` | -159.1784528440146 | -191.56621326446023 |
| SIE+MP2 | `SIE+MP2/main_logger.log` | -159.17136519012413 | -191.55348114327566 |
| SIE+CCSD(T) | `SIE+CCSD(T)/main_logger.log` | -159.40743567077087 | -191.464739648656 |

# Fully reproduce all results
Fully reproducing the results is highly challenging due to the involvement of large-scale computations on high-performance clusters. If you wish to attempt replication, it is recommended that you have the following computational resources:
1. GPU Node Requirements: 8 * A100-SXM4-80GB GPUs, 120 CPU cores, 2 TB of memory, and 8 TB of SSD disk space.
2. CPU Node Requirements: 96 CPU cores, 980 of memory, and 8 TB of SSD disk space.

In the following section, I will put describes how to fully reproduce the results step by step by following information.
1. **Scripts path**: shows the folder path for the scripts of different examples.
2. **Excute order**: shows the excute order for the scripts in the example folder.
3. **Settings**: shows the variables which need to be looped in several values. Without explicit instructions, these variables can be directly found in all the current example Python scripts and need to be manually modified by the reader. Additionally, all possible combinations of values for these variables should be explored unless otherwise specified.
4. **Some explaination**: provides some explaination for the variables shown in the **Settings**.
5. **Note**: highlights something in the current examples that require special attention or additional specific operations to be performed.

## Water@graphene

**Scripts path:**

```
byteqc/embyte/example/Script_for_Advancing_Surface_Chemistry_with_Large-Scale_Ab-Initio_Quantum_Many-Body_Simulations/water@graphene
```

### Interacting energy calculation

**Scripts path:**

```
water@graphene/1_E_int_calculation
```
**Excute order:**

OBC: `1-1_HF_OBC.py -> 2-1_MP2_OBC.py -> 3-1_SIE_OBC.py`

PBC: `1-2_HF_PBC.py -> 2-2_MP2_PBC.py -> 3-2_SIE_PBC.py`

**Settings:**
| Variable | Value for loop | 
| :---: | :---:|
| `basis` | `'ccecp-cc-pVDZ', 'ccecp-cc-pVTZ'` |
| `mol_type` | `0, 1, 2` |
| `n` for OBC | `2, 4, 6, 8` but `10` only for `'ccecp-cc-pVDZ'`|
| `k` for PBC | `4, 8, 10, 14` but `16` only for `'ccecp-cc-pVDZ'`|
| `ad_type` | `'0-leg', '2-leg'` |
| `if_MP2` only for SIE |`True, False`|

**Some explaination**

1. `mol_type`: controls the structure type. `2` for full water@graphene system, `1` for ghost-water@graphene , and `0` for water@ghost-graphene. This variable is set for CP correction BSSE.
2. `n` and `k`: controls the size of substrate. `n` is PAH($n$) under OBC, and `k` is the k-mesh of graphene supercell under PBC.
3. `ad_type`: The water monomer configuration. `'0-leg'` for the 0-leg adsorption, and `'2-leg'` for the 2-leg adsorption.
4. `if_MP2`: controls the calculation of SIE+MP2 (set `True`) or SIE+CCSD(T) (set `False`).

### Water rotation and water-graphene distance optimization

**Scripts path:**

```
water@graphene/2_Water_rotation_and_distance_opt
```

**Excute order:**

`1_HF.py -> 2_MP2.py -> 3_SIE.py`

**Settings:**
| Variable | Value for loop | 
| :---: | :---:|
| `ad_type` | `0, 30, 60, 90, 120, 150, 180` |
| `shift`|`numpy.arange(-0.5, 0.51, 0.1)`|
| `if_MP2` only for SIE|`True, False`|

**Some explaination**
1. `ad_type`: The water monomer rotate angle from 2-leg configuration.
2. `shift`: The shift of water-graphene distance from the rotate center.
3. `if_MP2`: controls the calculation of SIE+MP2 (set `True`) or SIE+CCSD (set `False`).

Note that a variable named `equilibrium_position_shift` is provided in the scripts to offer the equilibrium shift distance at different rotate angle.


### Geometry relaxation

**Scripts path:**

```
water@graphene/3_Geometry_relaxation
```

**Note**

Before excuting the scripts, `gpu4pyscf` should be installed in a new virtual miniconda environment by using following command:
```bash
conda create -n -y gpu4pyscf python=3.12
conda activate gpu4pyscf
pip3 install gpu4pyscf-cuda12x==1.1.0
```
And after excution, the `gpu4pyscf` environment should be deactivated and remove by using following command:
```bash
conda deactivate
conda remove --name gpu4pyscf --all -y
```

**Excute order:**

`1_DFT_geometry_opt.py -> 2_E_int_calculation.py`

**Settings:**

No variable need to be modified manully.


## Organic molecules@coronene

**Scripts path:**

```
byteqc/embyte/example/Script_for_Advancing_Surface_Chemistry_with_Large-Scale_Ab-Initio_Quantum_Many-Body_Simulations/Organic_mol@coronene
```

**Excute order:**

`1_DFT_geometry_opt.py -> 2_HF.py -> 3_MP2.py -> 4_SIE.py`

**Note**

No variable need to be modified manully for the script `1_DFT_geometry_opt.py`. And the `gpu4pyscf` environment is also need for running this script.

**Settings:**

| Variable | Value for loop | 
| :---: | :---:|
| `basis` | `cc-pVDZ, cc-pVTZ` |
| `mol_type` | `0, 1, 2` |
| `adsorbate`|`'acetone', 'acetonitrile', 'dichloromethane',`<br>`'ethanol', 'ethylacetate', 'toluene'`|
| `if_MP2` only for SIE|`True, False`|

**Some explaination**

1. `mol_type`: controls the structure type. `2` for full adsorbate@coronene system, `1` for ghost-adsorbate@coronene , and `0` for adsorbate@ghost-coronene. This variable is set for CP correction BSSE.
2. `adsorbate`: changes the adsorbate on coronene.
3. `if_MP2`: controls the calculation of SIE+MP2 (set `True`) or SIE+CCSD(T) (set `False`).

## CO@MgO

**Scripts path:**


```
byteqc/embyte/example/Script_for_Advancing_Surface_Chemistry_with_Large-Scale_Ab-Initio_Quantum_Many-Body_Simulations/CO@MgO
```

**Excute order:**

`1_HF_PBC.py -> 2_MP2_PBC.py -> 3_SIE_PBC.py`

**Note**

As the reason mentioned in "The demo case: water@graphene" part, the PBC HF calculation is excuted on CPU.

**Settings:**

| Variable | Value for loop | 
| :---: | :---:|
| `basis` | `aug-cc-pVDZ, aug-cc-pVTZ` and `aug-cc-pVQZ` only for HF |
| `mol_type` | `0, 1, 2` |
| `if_MP2` only for SIE | `True, False`|

**Some explaination**

1. `mol_type`: controls the structure type. `2` for full CO@MgO system, `1` for ghost-CO@MgO , and `0` for CO@ghost-MgO. This variable is set for CP correction BSSE.
2. `if_MP2`: controls the calculation of SIE+MP2 (set `True`) or SIE+CCSD(T) (set `False`).

## CO/CO2@CPO-27-Mg

**Scripts path:**


```
byteqc/embyte/example/Script_for_Advancing_Surface_Chemistry_with_Large-Scale_Ab-Initio_Quantum_Many-Body_Simulations/CO@CPO-27-Mg

byteqc/embyte/example/Script_for_Advancing_Surface_Chemistry_with_Large-Scale_Ab-Initio_Quantum_Many-Body_Simulations/CO2@CPO-27-Mg
```

**Excute order:**

`1_HF.py -> 2_MP2.py -> 3_SIE.py`

**Settings:**

| Variable | Value for loop | 
| :---: | :---:|
| `basis` | `aug-cc-pVDZ, aug-cc-pVTZ` |
| `mol_type` | `0, 1, 2` |
| `if_MP2` only for SIE | `True, False`|

**Some explaination**

1. `mol_type`: controls the structure type. `2` for full CO/CO2@CPO-27-Mg system, `1` for ghost-CO/CO2@CPO-27-Mg, and `0` for CO/CO2@ghost-CPO-27-Mg This variable is set for CP correction BSSE.
2. `if_MP2`: controls the calculation of SIE+MP2 (set `True`) or SIE+CCSD(T) (set `False`).


