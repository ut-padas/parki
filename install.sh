#!/usr/bin/env bash
set -euo pipefail

############################
# User-configurable options
############################

ENV_NAME=${ENV_NAME:-parki}
PYTHON_VERSION=${PYTHON_VERSION:-3.13}

# Execution spaces
ENABLE_OPENMP=${ENABLE_OPENMP:-ON}
ENABLE_CUDA=${ENABLE_CUDA:-OFF}
ENABLE_HIP=${ENABLE_HIP:-OFF}

############################
# Sanity checks
############################

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Please install Miniconda or Anaconda."
  exit 1
fi

if [[ "$ENABLE_CUDA" == "ON" ]] && ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: ENABLE_CUDA=ON but nvcc not found"
  exit 1
fi

############################
# Create conda environment
############################

echo "Creating conda environment: $ENV_NAME"

conda create -y \
  -n "$ENV_NAME" \
  python="$PYTHON_VERSION"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
conda info --envs

conda install -y -c conda-forge pybind11 patchelf pandas matplotlib scipy black

if [[ "$ENABLE_CUDA" == "ON" ]]; then
  conda install -y -c conda-forge cupy
elif [[ "$ENABLE_HIP" == "ON" ]]; then
  conda install -y -c conda-forge cupy-hip
fi

############################
# Install pykokkos
############################

echo "Installing pykokkos"

if [[ -d external ]]; then
  rm -rf external
fi
mkdir external; cd external

git clone https://github.com/kokkos/pykokkos.git
cd pykokkos

conda env update -n $ENV_NAME -f base/environment.yml
python install_base.py install -- \
	-DENABLE_VIEW_RANKS=4 \
	-DENABLE_MEMORY_TRAITS=OFF \
	-DENABLE_THREADS=OFF \
	-DENABLE_LAYOUTS=ON \
	-DENABLE_CUDA=$ENABLE_CUDA \
	-DENABLE_HIP=$ENABLE_HIP \
	-DENABLE_OPENMP=$ENABLE_OPENMP

pip install --user -e .

cd ../..

############################
# Install ParkiPy
############################

echo "Installing ParkiPy"
pip install -e .

############################
# Validation
############################

python - <<EOF
import parkipy
import pykokkos as pk
print("ParkiPy installed successfully")
print("Execution spaces:", pk.ExecutionSpace)
EOF

echo "Installation complete."
