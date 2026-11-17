#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
bootstrap="${DEX_V7_BOOTSTRAP_PYTHON:-python3.11}"
"$bootstrap" -c 'import platform,sys; sys.exit(0 if sys.version_info[:2] == (3,11) and platform.system() == "Linux" and platform.machine() == "x86_64" else "Linux x86_64 and Python 3.11 are required for the bundled wheels")'
if ! compgen -G 'wheels/*.whl' >/dev/null; then
  echo 'Extract Dex-WBC-Locomotion-Sim-Linux-x86_64-Py311-wheels.tar.zst from packages/ in the repository root first.' >&2
  exit 2
fi
if [[ ! -d .venv ]]; then
  "$bootstrap" -m venv .venv
fi
.venv/bin/python -m pip install --no-index --find-links "$PWD/wheels" -r requirements.txt
.venv/bin/python -m pip check
