#!/usr/bin/env bash
set -eo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
workspace="$root/out/ros_ws"
ros_setup="${DEX_ROS_SETUP:-/opt/ros/humble/setup.bash}"
ros_python="${DEX_ROS_PYTHON:-/usr/bin/python3}"
if [[ ! -r "$ros_setup" ]]; then
  printf 'ROS 2 environment not found at %s. Install ROS 2 Humble or set DEX_ROS_SETUP.\n' "$ros_setup" >&2
  exit 1
fi
source "$ros_setup"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
if [[ "${1:-}" == build ]]; then
  if ! command -v colcon >/dev/null; then
    printf 'colcon not found. Install python3-colcon-common-extensions.\n' >&2
    exit 1
  fi
  exec colcon --log-base "$workspace/log" build --base-paths "$root/ros_ws/src" \
    --build-base "$workspace/build" --install-base "$workspace/install" \
    --packages-select bodyctrl_msgs --cmake-force-configure \
    --cmake-args -DBUILD_TESTING=OFF "-DPython3_EXECUTABLE=$ros_python"
fi
if [[ ! -r "$workspace/install/bodyctrl_msgs/share/bodyctrl_msgs/package.xml" ]]; then
  printf 'The local ROS messages are not built. Run ./ros_sim/run.sh build first.\n' >&2
  exit 1
fi
source "$workspace/install/local_setup.bash"
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-147}"
if [[ "${1:-}" == teleop ]]; then
  shift
  exec "$ros_python" "$root/ros_sim/teleop_gui.py" "$@"
fi
if [[ "${1:-}" == joy-relay ]]; then
  shift
  cd "$root"
  exec "$ros_python" -m ros_sim.joy_relay "$@"
fi
wbc_python="${DEX_V7_PYTHON:-$root/.venv/bin/python}"
(
  cd "$root"
  source "$root/env.sh"
  "$dex_python" "$root/tools/compile_models.py" model
)
exec "$ros_python" "$root/ros_sim/bridge.py" --wbc-python "$wbc_python" "$@"
