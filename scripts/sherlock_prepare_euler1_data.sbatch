#!/bin/bash
#SBATCH --job-name=euler1_prepare
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
RAW_DIR="${RAW_DIR:-Archive (1)}"
OUT_DIR="${OUT_DIR:-data/euler1_clean}"
RESULT_DIR="${RESULT_DIR:-results/euler1_comparison}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-twoway-euler1-benchmark}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_DIR/.venv/bin/activate}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12.1}"

load_python_module() {
    if command -v ml >/dev/null 2>&1; then
        ml "$PYTHON_MODULE"
    elif command -v module >/dev/null 2>&1; then
        module load "$PYTHON_MODULE"
    fi
}

activate_python_env() {
    if [[ -n "${PY_ENV_ACTIVATE:-}" ]]; then
        source "$PY_ENV_ACTIVATE"
        return
    fi

    if [[ -f "$VENV_ACTIVATE" ]]; then
        source "$VENV_ACTIVATE"
        return
    fi

    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV_NAME"
        return
    fi

    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV_NAME"
        return
    fi

    echo "No Python environment found. Run scripts/sherlock_setup_euler1_venv.sbatch first or set PY_ENV_ACTIVATE=/path/to/venv/bin/activate." >&2
    exit 1
}

cd "$PROJECT_DIR"
load_python_module
activate_python_env

export PYTHONUNBUFFERED=True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python -u scripts/inspect_euler1_h5.py \
    --h5 "$RAW_DIR/euler1_train_data.h5" \
         "$RAW_DIR/euler1_valid_data.h5" \
         "$RAW_DIR/euler1_test_data.h5" \
    --out "$RESULT_DIR/data_report_raw.json"

python -u scripts/prepare_clean_euler1_splits.py \
    --input_h5 "$RAW_DIR/euler1_train_data.h5" \
               "$RAW_DIR/euler1_valid_data.h5" \
               "$RAW_DIR/euler1_test_data.h5" \
    --out_dir "$OUT_DIR" \
    --seed 12345 \
    --train_ratio 0.8 \
    --valid_ratio 0.1 \
    --test_ratio 0.1 \
    --stratify_by logS

python -u scripts/inspect_euler1_h5.py \
    --h5 "$OUT_DIR/euler1_train_clean.h5" \
         "$OUT_DIR/euler1_valid_clean.h5" \
         "$OUT_DIR/euler1_test_clean.h5" \
    --out "$RESULT_DIR/data_report_clean.json"
