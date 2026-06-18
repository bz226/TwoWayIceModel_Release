#!/bin/bash
#SBATCH --job-name=euler1_venv
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12.1}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$PROJECT_DIR/requirements_euler1_benchmark.txt}"
RECREATE_VENV="${RECREATE_VENV:-0}"

cd "$PROJECT_DIR"

if command -v ml >/dev/null 2>&1; then
    ml "$PYTHON_MODULE"
elif command -v module >/dev/null 2>&1; then
    module load "$PYTHON_MODULE"
else
    echo "Warning: no module command found; using current python3." >&2
fi

if [[ "$RECREATE_VENV" == "1" && -d "$VENV_DIR" ]]; then
    rm -rf "$VENV_DIR"
fi

python3 --version
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python --version
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r "$REQUIREMENTS_FILE"

python - <<'PY'
import h5py
import matplotlib
import numpy
import torch

print("Euler-1 benchmark venv check")
print("numpy", numpy.__version__)
print("h5py", h5py.__version__)
print("matplotlib", matplotlib.__version__)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY
