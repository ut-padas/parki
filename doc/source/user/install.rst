.. _install:

************
Installation
************

.. note::
   Prebuilt binary wheels for common architectures are planned for a future
   release. Until then, please follow the native install guide below.


Prerequisites
-------------

Before installing ParkiPy, ensure the following are available on your system:

- `Conda <https://docs.conda.io/en/latest/>`_ (Miniconda or Anaconda)
- A C++17-capable compiler (e.g. GCC ≥ 9)
- **For GPU builds**: CUDA ≥ 12 (NVIDIA) or ROCm (AMD)
- **For multi-GPU builds**: an MPI library such as HPC-X or OpenMPI, and
  ``nvmath-python`` ≥ 8.0


Native Install
--------------

The repository ships an ``install.sh`` script that creates a Conda environment
and performs a native build of PyKokkos. The build is controlled by the
environment variables listed below.

.. list-table:: ``install.sh`` configuration flags
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Default
     - Description
   * - ``ENV_NAME``
     - ``parki``
     - Name of the Conda environment to create.
   * - ``PYTHON_VERSION``
     - ``3.13``
     - Python version used in the Conda environment.
   * - ``ENABLE_OPENMP``
     - ``ON``
     - Enable the Kokkos OpenMP execution space (CPU parallelism).
   * - ``ENABLE_CUDA``
     - ``OFF``
     - Enable the Kokkos CUDA execution space (NVIDIA GPU support).
   * - ``ENABLE_HIP``
     - ``OFF``
     - Enable the Kokkos HIP execution space (AMD GPU support).

Set the desired variables and run the script:

.. code-block:: bash

   # CPU-only build (OpenMP)
   bash install.sh

   # NVIDIA GPU build
   ENABLE_CUDA=ON bash install.sh

   # AMD GPU build
   ENABLE_HIP=ON bash install.sh

   # Custom environment name and Python version
   ENV_NAME=myenv PYTHON_VERSION=3.11 ENABLE_CUDA=ON bash install.sh

Once the script completes, activate the environment:

.. code-block:: bash

   conda activate parki   # or whatever ENV_NAME was set to


Verifying the Install
---------------------

After activating the environment, confirm ParkiPy is importable:

.. code-block:: python

   import parkipy
   print(parkipy.__version__)

You can also run the full test suite from the repository root:

.. code-block:: bash

   pytest tests


Multi-GPU Installation
----------------------

Multi-GPU support requires two additional packages: ``mpi4py`` for
inter-node communication and ``nvmath-python`` for distributed FFTs.
The tests use the ``hpcx`` and ``nvhpc-hpcx-cuda12/25.5`` modules for
CUDA and MPI libraries.

Install both packages with ``pip`` after activating your Conda environment.
``mpi4py`` must be built from source so it links against your system MPI:

.. code-block:: bash

   CC=gcc CXX=g++ CFLAGS="" CXXFLAGS="" \
       pip install mpi4py --no-cache-dir --no-binary :all:

   pip install nvmath-python

.. warning::
   Do **not** install ``mpi4py`` from a pre-built binary (e.g. via Conda or
   the default pip wheel) when using a custom HPC MPI stack such as HPC-X.
   Building from source ensures the correct MPI library is linked.

Once both packages are installed, the :mod:`parkipy.distributed.ewald` module
will be available.


Building the Documentation
--------------------------

The documentation is built with `Sphinx <https://www.sphinx-doc.org/>`_ using
the `Furo <https://pradyunsg.me/furo/>`_ theme. Install both into your Conda
environment:

.. code-block:: bash

   conda install -c conda-forge sphinx furo

Then build from the ``doc/`` directory:

.. code-block:: bash

   cd doc
   make html

Open ``doc/build/html/index.html`` in a browser to view the result.
