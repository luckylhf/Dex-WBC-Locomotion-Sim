#!/usr/bin/env bash
set -e
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${DEX_V7_RUNTIME_ROOT:-$here/..}"
source env.sh
out="$1"
MUJOCO_GL=egl "$dex_python" -m realtime.omni_video "$out" "$out/front_side_unlabeled.mp4"
ffmpeg -nostdin -y -loglevel error -i "$out/front_side_unlabeled.mp4" \
    -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='Isaac Sim PhysX trajectory  |  MuJoCo visualization':fontsize=16:fontcolor=white:x=w-tw-16:y=15" \
    -an -c:v libx264 -crf 19 -preset fast -threads 3 -movflags +faststart "$out/front_side.mp4"
rm "$out/front_side_unlabeled.mp4"
