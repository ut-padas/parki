# (Par)allel (Par)ticle (K)ernel (I)nteractions library: ParKI
The ParKI library provides a python API, ParkiPy, supporting a `CellList` class for local-particle interactions, the `ewald` module for computing Ewald summations of the Stokes and Laplace kernels in arbitrary periodicities, and the `distributed.ewald` module for computing Ewald summation in a slab distributed box. 

## Supported Kernels

|            | Stokes single layer | Stokes single + double layer | Laplace |
|------------|---------------------|------------------------------|---------|
| 0-periodic | ❌                  | ❌                           | ❌      |
| 1-periodic | ❌                  | ✅                           | ❌      |
| 2-periodic | ❌                  | ❌                           | ❌      |
| 3-periodic | ❌                  | ✅                           | ✅      |


# Repository layout
... 

# Installing ParkiPy

We provide the installation script `install.sh` builds a conda environment 
and performs a native install of PyKokkos based off specified enviornment
variables, listed below:
 
| flag           | default | description                              |
|----------------|---------|------------------------------------------|
| ENV_NAME       | parki   | Name of the conda environment            |
| PYTHON_VERSION | 3.13    | Python version of the conda environment  |
| ENABLE_OPENMP  | ON      | Enable the Kokkos OpenMP execution space |
| ENABLE_CUDA    | OFF     | Enable the Kokkos CUDA execution space   |
| ENABLE_HIP     | OFF     | Enable the Kokkos HIP execution space    |

Once the environment variables are set, install with `bash install.sh` 
