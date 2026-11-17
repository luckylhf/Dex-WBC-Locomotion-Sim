#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
usage() {
  printf '%s\n' 'Usage: ./run_v7.sh verify | compile | {live|empty|load-1kg|load-10kg-stand|load-10kg-forward|load-10kg-omni} NEW_OUTPUT_DIRECTORY [--video]'
}
mode="${1:-help}"
case "$mode" in
  help|-h|--help) usage; exit 0 ;;
  verify) [[ $# -eq 1 ]] || { usage >&2; exit 2; }
          cd "$root"; sha256sum --quiet -c SHA256SUMS; echo 'Package checksums verified'; exit 0 ;;
  compile) [[ $# -eq 1 ]] || { usage >&2; exit 2; }
           source "$root/env.sh"
           if [[ ! -x "$dex_python" ]]; then echo 'Set DEX_V7_PYTHON to the installed Python.' >&2; exit 2; fi
           exec "$dex_python" "$root/tools/compile_models.py" all ;;
  live|empty|load-1kg|load-10kg-stand|load-10kg-forward|load-10kg-omni) ;;
  *) usage >&2; exit 2 ;;
esac
if [[ $# -lt 2 || $# -gt 3 || -z "$2" || ( $# -eq 3 && "$3" != --video ) ]]; then
  usage >&2; exit 2
fi
out="$2"
[[ "$out" == /* ]] || out="$PWD/$out"
if [[ -e "$out" || -L "$out" ]]; then
  printf 'Choose a new output directory: %s\n' "$out" >&2; exit 2
fi
source "$root/env.sh"
if [[ ! -x "$dex_python" ]]; then echo 'Install requirements.txt in a virtual environment or set DEX_V7_PYTHON.' >&2; exit 2; fi
if [[ "${3:-}" == --video ]] && ! command -v ffmpeg >/dev/null; then
  echo 'Install FFmpeg to generate video, or omit --video.' >&2; exit 2
fi
cd "$root"
models=(model)
case "$mode" in
  empty) models+=(empty) ;;
  load-1kg) models+=(one_kg) ;;
  load-10kg-*) models+=(braced_ten_kg) ;;
esac
"$dex_python" "$root/tools/compile_models.py" "${models[@]}"
report=carry_omni_report
video=carry_video
case "$mode" in
  live) "$dex_python" -m realtime.sim --demo --out "$out"; report=report; video=video ;;
  empty) "$dex_python" -m realtime.carry_omni --out "$out" ;;
  load-1kg) "$dex_python" -m realtime.carry_payload --added-mass 1 --out "$out" ;;
  load-10kg-stand) "$dex_python" -m realtime.carry_load_transfer --stand-seconds 60 --out "$out"; report=carry_load_report ;;
  load-10kg-forward) "$dex_python" -m realtime.carry_load_transfer --walk --out "$out"; report=carry_loaded_walk_report ;;
  load-10kg-omni) "$dex_python" -m realtime.carry_load_transfer --omni --out "$out"; report=carry_loaded_walk_report ;;
esac
"$dex_python" -m "realtime.$report" "$out"
"$dex_python" - "$out" "$report" <<'PY'
import json, sys
from pathlib import Path

out, report = Path(sys.argv[1]), sys.argv[2]
name = 'settle_metrics.json' if report == 'carry_load_report' else 'metrics.json'
metrics = json.loads((out / name).read_text())
if report == 'carry_omni_report':
    trials = metrics['trials']
    passed = bool(trials and all(t.get('result') and t['completed'] and
                                 all(v for v in t['result'].values() if isinstance(v, bool)) for t in trials))
elif report == 'report':
    segments = metrics['speed_segments']
    stops = metrics['stops']
    passed = (len(segments) == 6 and len(stops) == 1 and metrics['alternating']
              and 1.9 <= (metrics['walking_cadence_hz'] or 0.) <= 2.1
              and not any(metrics[k] for k in ('crossing_frames', 'rub_substeps', 'nonsupport_ground_substeps'))
              and all(abs(x['actual_mean_mps'] - x['command_mps']) <= .025 for x in segments)
              and all(x['final_ankle_gap_heading_m'] <= .025 and x['settled_root_drift_m'] <= .025 for x in stops))
else:
    passed = metrics.get('passed', False)
if not passed or not metrics['completed'] or metrics['physics_resets'] != 0:
    sys.exit(f'Acceptance failed. Results were kept in {out / name}')
PY
if [[ "${3:-}" == --video ]]; then
  MUJOCO_GL="${MUJOCO_GL:-egl}" "$dex_python" -m "realtime.$video" "$out" "$out/front_side.mp4"
fi
printf 'Results: %s\n' "$out"
