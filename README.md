# ParKI
```
         ____                 __  __   ______
        /\  _`\              /\ \/\ \ /\__  _\
        \ \ \L\ \ __     _ __\ \ \/'/'\/_/\ \/
         \ \ ,__/'__`\  /\`'__\ \ , <    \ \ \
          \ \ \/\ \L\.\_\ \ \/ \ \ \\`\   \_\ \__
           \ \_\ \__/.\_\\ \_\  \ \_\ \_\ /\_____\
            \/_/\/__/\/_/ \/_/   \/_/\/_/ \/_____/

          Parallel  ·  Particle  ·  Kernel  ·  Interactions

   .+------+     +------+     +------+     +------+     +------+.
 .' |    .'|    /|     /|     |      |     |\     |\    |`.    | `.
+---+--+'  |   +-+----+ |     +------+     | +----+-+   |  `+--+---+
|   |  |   |   | |    | |     |      |     | |    | |   |   |  |   |
|  ,+--+---+   | +----+-+     +------+     +-+----+ |   +---+--+   |
|.'    | .'    |/     |/      |      |      \|     \|    `. |   `. |
+------+'      +------+       +------+       +------+      `+------+

```
The ParKI library provides a python API, ParkiPy, supporting a `CellList` class for local-particle interactions, the `ewald` module for computing Ewald summations of the Stokes and Laplace kernels in arbitrary periodicities, and the `distributed.ewald` module for computing Ewald summations in a slab-distributed box.


## Supported Kernels

|            | Stokes single layer | Stokes single + double layer | Laplace |
|------------|---------------------|------------------------------|---------|
| 0-periodic | ❌                  | ❌                           | ❌      |
| 1-periodic | ✅                  | ✅                           | ❌      |
| 2-periodic | ❌                  | ❌                           | ❌      |
| 3-periodic | ✅                  | ✅                           | ✅      |

# Installing ParkiPy

We provide the installation script `install.sh` which builds a conda environment 
and performs a native install of PyKokkos based on specified environment
variables, listed below:
 
| flag           | default | description                              |
|----------------|---------|------------------------------------------|
| ENV_NAME       | parki   | Name of the conda environment            |
| PYTHON_VERSION | 3.13    | Python version of the conda environment  |
| ENABLE_OPENMP  | ON      | Enable the Kokkos OpenMP execution space |
| ENABLE_CUDA    | OFF     | Enable the Kokkos CUDA execution space   |
| ENABLE_HIP     | OFF     | Enable the Kokkos HIP execution space    |

Once the environment variables are set, install with `bash install.sh` 

## Multi-GPU installation
Our tests use `hpcx` and `nvhpc-hpcx-cuda12/25.5`
for CUDA and MPI libraries.

ParkiPy relies on `mpi4py` for internode communication
and `nvmath-python v8.0` for distributed FFTs.

We recommend installing these packages with pip:
```
CC=gcc CXX=g++ CFLAGS="" CXXFLAGS="" pip install mpi4py --no-cache-dir --no-binary :all:
pip install nvmath-python
```

# Example
Consider the discretized Stokes single-layer potential
```math
u(x_i) = \sum_{j=1}^{N_s} \left( \frac{I}{\|x_i-y_j\|} + \frac{(x_i-y_j) \otimes (x_i-y_j) }{\|x_i-y_j\|^3} \right) f(y_j)
```
with fully periodic boundary conditions.

Computing $u(x_i)$ is easy with ParKI:

```python
import cupy as cp     # numpy also supported
import parkipy

rng = cp.random.default_rng(123)

# generate particles and densities
x = rng.random(size=(3, 312))
y = rng.random(size=(3, 773))
f = rng.random(size=(3, 773))

# declare Ewald sum options
options = parkipy.ewald.EwaldOptions(
    periodicity=3,
    box=[1,1,1],
    tolerance=1e-8,
    cell_size=23,
    execution_space="CUDA",
)

# compute the potential
u = parkipy.ewald.stokes_sl(x, y, f, options)
```

# Repository Structure
```bash
.
├── analysis                # Performance analysis scripts
│   ├── cycle_counts
│   ├── distributed
│   ├── erf-approximation
│   └── ewald
├── doc                     # Documentation
├── examples                # Common uses
│   ├── distributed
│   │   └── ewald
│   └── ewald
├── external                # Location for third-party libraries
├── parkipy                 # parkipy module
│   ├── distributed         # parkipy.distributed module
│   │   └── ewald           # parkipy.distributed.ewald module
│   ├── ewald               # parkipy.ewald module
│   │   └── _pk_kernels     # parkipy.ewald PyKokkos kernels
│   │       └── templates
│   └── _pk_kernels         # parkipy PyKokkos kernels
│       └── templates
└── tests                   # Unit tests
```
The ParKI repository contains 6 subdirectories:
* **analysis**: performance analysis scripts for package methods.
* **doc**: rst files used by sphinx to generate package documentation.
* **examples**: common use cases for different APIs.
* **external**: a placeholder directory for an external install of pykokkos via the `install.sh` script.
* **parkipy**: python source code; defines the `parkipy` namespace as well as the `parkipy.ewald` and `parkipy.distributed` submodules. 
	+ PyKokkos kernels are defined in the `_pk_kernels` subdirectories. The kernels are compiled just-in-time and cached in an auto-generated `pk_cpp/` repository. 
* **tests**: python unit tests; run with `pytest tests`.

# Reproducing/Generating Performance Results
## Tables
* **Table 1** (millicycles): follow steps in `analysis/cycle_counts/README.md`
* **Table 3** (P2P models): run `analysis/ewald/analyze_p2p_performance_models.py` 
* **Table 5** (P2G/G2P models): run `analysis/ewald/analyze_p2g_performance_models.py` and `analysis/ewald/analyze_g2p_performance_models.py` 
* **Table 6** (P2P methods): run `analysis/ewald/analyze_p2p_methods.py` 
* **Table 7** (P2G methods): run `analysis/ewald/analyze_p2g_methods.py`
* **Table 8** (G2P methods): run `analysis/ewald/analyze_g2p_methods.py`
* **Table 9** (non-uniformity): run `analysis/ewald/analyze_particle_distributions.py`
* **Table 10** (float precision): run `analysis/ewald/analyze_dtypes.py`
* **Table 11** (multi-GPU Ewald): run `analysis/distributed/analyze_ewald_mpi.py`

## Figures
* **Figure 4** (erf(x)/x): follow steps in `analysis/cycle_counts/README.md`
* **Figure 5** (roofline): run `analysis/ewald/analyze_roofline_model.py` 
* **Figure 6** (P2P workloads): run `analysis/ewald/analyze_p2p_workloads.py`
* **Figure 7** (P2P portability): run `analysis/ewald/analyze_p2p_portability.py`
* **Figure 8** (P2G portability): run `analysis/ewald/analyze_p2g_portability.py`
* **Figure 9** (Ewald portability): run `analysis/ewald/analyze_ewald_portability.py`

# Building the Docs

To build the documentation locally, first install Sphinx and the required Furo theme:
```bash
conda install -c conda-forge sphinx furo
```

Then, navigate to the `doc` directory and build the HTML documentation:
```bash
cd doc
make html
```

Once the build completes, you can view the documentation by opening `doc/build/html/index.html` in your web browser.
