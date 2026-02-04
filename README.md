# (Par)allel (Par)ticle (K)ernel (I)nteractions library: ParKI
The ParKI library provides a python API, ParkiPy, supporting a `CellList` class for local-particle interactions, the `ewald` module for computing Ewald summations of the Stokes and Laplace kernels in arbitrary periodicities, and the `distributed.ewald` module for computing Ewald summation in a slab distributed box. 

## Supported Kernels

|            | Stokes single layer | Stokes single + double layer | Laplace |
|------------|---------------------|------------------------------|---------|
| 0-periodic | ❌                  | ❌                           | ❌      |
| 1-periodic | ✅                  | ✅                           | ❌      |
| 2-periodic | ❌                  | ❌                           | ❌      |
| 3-periodic | ✅                  | ✅                           | ✅      |

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


# Repository Structure
```bash
.
├── analysis
│   ├── cycle_counts
│   ├── distributed
│   │   └── data
│   ├── erf-approximation
│   └── ewald
│       ├── data
│       └── plots
├── doc
│   └── source
│       ├── reference
│       │   └── generated
│       ├── _static
│       ├── _templates
│       │   └── autosummary
│       └── user
├── examples
│   ├── distributed
│   │   └── ewald
│   └── ewald
├── external
├── parkipy
│   ├── distributed
│   │   └── ewald
│   ├── ewald
│   │   └── _pk_kernels
│   │       └── templates
│   └── _pk_kernels
│       └── templates
└── tests
```
The Parki repository contains 6 subdirectories:
* **analysis**: performance analysis scripts for package methods.
* **doc**: rst files used by sphinx to generate package documentation.
* **examples**: common use cases for different APIs.
* **external**: a dummy repository for an external install of pykokkos via the `install.sh` script.
* **parkipy**: python source code; defines the `parkipy` namespace as well as the `parkipy.ewald` and `parkipy.distributed` submodules. 
	+ PyKokkos kernels are defined in the `_pk_kernels` subdirectories. The kernels are computed just-in-time and cached in an auto-generated `pk_cpp/` repository. 
* **tests**: python unit tests; run with `pytest tests`.

# Reproducing/Generating Performance Results
...
