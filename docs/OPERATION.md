# 运行与问题排查

## 安装和启动

先按 [README](../README.md#克隆后离线安装) 解压仓库中的离线 wheels。`./install.sh` 使用 `--no-index` 将 `wheels/` 中的固定版本依赖安装到本目录 `.venv`。可用 `DEX_V7_BOOTSTRAP_PYTHON` 指向现有的 Python 3.11。安装固定依赖需要本机支持该架构的 wheel 与 Python `venv`。使用现有隔离解释器时，可将 `DEX_V7_PYTHON` 指向环境中 Python 的绝对路径。

```bash
DEX_V7_BOOTSTRAP_PYTHON=/path/to/python3.11 ./install.sh
./run_v7.sh --help
./run_v7.sh load-10kg-forward ./out/forward --video
```

MuJoCo 入口首次运行编译所选模型，生成文件的 SHA-256 与冻结值一致后才进入仿真。`./run_v7.sh compile` 可一次生成四个缓存。[Isaac Sim 离线准备](../isaacsim/README.md#linux-x86_64-离线安装与运行)

仿真场景含 0.65 kg 空箱、腕端新增 1 kg 和腕端新增 10 kg 的标定模型。自定义箱体参数属于建模流程。保留完整的输出目录可复算接触、载荷、并步和姿态指标。

## 实时命令

本地无箱模拟支持 UDP 前进速度指令。第一个终端执行以下命令，在另一个终端发送速度。指令为 m/s，支持 0 或 0.10–0.40。最后一次有效指令持续生效，发送 0 触发减速停车。

```bash
./run_v7.sh compile
source env.sh
"$dex_python" -m realtime.sim --pace --port 18070 --duration 90 --out ./out/udp-test
```

```bash
source env.sh
"$dex_python" -c 'import socket,sys; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.sendto(sys.argv[1].encode(),("127.0.0.1",18070))' 0.20
# 发送 0 可触发停车
```

三轴空手遥操作见 [ROS 2 遥操作](ROS_MUJOCO_TELEOP.md)。空手和抱箱也可用 [原生 MuJoCo 窗口](../realtime/README.md#可交互-viewer) 本机操作。预设抱箱入口运行固定指令序列。

## 输出和排查

| 文件或表现 | 内容或处理 |
| --- | --- |
| `run.json` | 模型哈希、控制选项、物理时钟和命令阶段 |
| `metrics.json` / `settle_metrics.json` | 速度、停稳、姿态、载荷及足部判据 |
| `rollout.npz` 或 `rollout.trajectory.npz` | 50 Hz 状态和过程数据 |
| `telemetry.npy`、`clearance.npy` | 腕力与足底承重、脚间净空等诊断 |
| 输出目录已存在 | 指定新目录。原输出可用于继续复算 |
| 校验失败 | 用 `./run_v7.sh verify` 检查解压后的文件 |
| EGL、GL 或 FFmpeg 错误 | 可以先运行纯仿真并检查指标，再配置视频依赖重录 |
| 验收未过 | 入口保留原输出及指标，并返回非零退出码 |

对已生成的带载输出，可重新评估与渲染。

```bash
source env.sh
"$dex_python" -m realtime.carry_loaded_walk_report ./out/load-10kg-omni
MUJOCO_GL=egl "$dex_python" -m realtime.carry_video ./out/load-10kg-omni ./out/omni-rerender.mp4
```
