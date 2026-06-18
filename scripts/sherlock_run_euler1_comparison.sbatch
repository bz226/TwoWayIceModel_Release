#!/bin/bash
#SBATCH --job-name=euler1_compare
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
CLEAN_DIR="${CLEAN_DIR:-data/euler1_clean}"
OUTPUT_DIR="${OUTPUT_DIR:-results/euler1_comparison}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-twoway-euler1-benchmark}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_DIR/.venv/bin/activate}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12.1}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
SEEDS="${SEEDS:-12345 23456 34567}"
MODELS="${MODELS:-persistence,resnet_cnn,fno}"

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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-euler1}"
mkdir -p "$MPLCONFIGDIR"

python -u scripts/compare_euler1_surrogates.py \
    --train_h5 "$CLEAN_DIR/euler1_train_clean.h5" \
    --valid_h5 "$CLEAN_DIR/euler1_valid_clean.h5" \
    --test_h5 "$CLEAN_DIR/euler1_test_clean.h5" \
    --models "$MODELS" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --seeds $SEEDS \
    --output_dir "$OUTPUT_DIR" \
    --preload
