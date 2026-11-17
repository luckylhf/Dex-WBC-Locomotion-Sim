#!/usr/bin/env bash
set -e
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
runtime="${DEX_V7_RUNTIME_ROOT:-$here/..}"
sim_root="${DEX_ISAAC_SIM_ROOT:?Set DEX_ISAAC_SIM_ROOT to the Isaac Sim 5.1 installation}"
py="$(command -v "${DEX_ISAAC_PYTHON:-python}")" || {
    echo "Use the Isaac Lab Python 3.11 environment or set DEX_ISAAC_PYTHON." >&2
    exit 2
}
cd "$runtime"
runtime="$(pwd -P)"
missing() {
    printf 'Missing %s. See %s/README.md for setup.\n' "$1" "$here" >&2
    exit 2
}
for file in baseline/bundle/{contract.json,policy.onnx} realtime/omni.py; do
    [[ -r "$file" ]] || missing "$runtime/$file"
done
[[ -r "$sim_root/setup_conda_env.sh" ]] || missing "$sim_root/setup_conda_env.sh"
asset="${DEX_ISAAC_ROBOT_ASSETS:-$runtime/baseline/assets/dex_evt}"
[[ -r "$asset/urdf/dex_29.urdf" ]] || missing "$asset/urdf/dex_29.urdf"
models=(model)
for arg in "$@"; do
    if [[ "$arg" == --carry ]]; then
        models+=(empty braced_ten_kg)
        break
    fi
done
source "$sim_root/setup_conda_env.sh"
export PYTHONPATH="${DEX_ISAAC_PYTHON_OVERLAY:+$DEX_ISAAC_PYTHON_OVERLAY:}$runtime/baseline/src:$runtime:${PYTHONPATH:-}"
export DEX_ISAAC_ROBOT_ASSETS="$asset"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$py" "$here/../tools/compile_models.py" --root "$runtime" "${models[@]}"
exec "$py" "$here/run_empty.py" "$@"
