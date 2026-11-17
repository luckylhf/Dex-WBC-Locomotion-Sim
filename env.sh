#!/usr/bin/env bash
v7_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
dex_python="${DEX_V7_PYTHON:-$v7_root/.venv/bin/python}"
export DEX_MIMIC_ROOT="$v7_root/baseline" PYTHONPATH="$v7_root/baseline/src:$v7_root"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export DEX_EVALUATION_PROFILE=jump_tracking_v2
