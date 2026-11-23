#!/bin/bash
#SBATCH -J parki-adaptive-1p
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o adaptive-1p-%j.out
#SBATCH -e adaptive-1p-%j.err

set -eo pipefail
source "$WORK/setup_sphcaps.sh"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
printf 'Commit: %s\n' "$(git rev-parse HEAD)"
python -m pytest -q tests/test_ewald_plan.py -k 'adaptive_against_global and GPU'
python -u analysis/ewald/time_adaptive_1p.py --sizes 10000 30000 100000 300000 --repeats 3 --tolerance 0.0001
