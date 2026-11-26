# Isaac Sim 空手与抱箱仿真

运行入口使用 Isaac Sim 5.1 / Isaac Lab 2.3.0 的 CPU PhysX。空手为 200 Hz 物理步进，带载示例为 400 Hz，WBC 均为 50 Hz。地面默认厚 2 m，顶面保持 z=0。带载入口以抓持接口反力估重，配重位于两侧腕端。 URDF 导入配置为 `collision_from_visuals=False`、`replace_cylinders_with_capsules=True`，双脚共 14 根胶囊。

## Linux x86_64 离线安装与运行

先准备可运行的 Isaac Sim 5.1、Isaac Lab 2.3.0 和其 Python 3.11 环境。Isaac Sim 与 Isaac Lab 本体单独安装。按 [主 README](../README.md#克隆后离线安装) 在克隆后的仓库根目录解压离线 wheels，并用 `install.sh` 备好 MuJoCo 的评分与渲染环境。

从仓库根目录，把两个 wheel 安装到 Isaac 专用模块目录。使用 `--no-deps` 保留 Isaac 环境已有的 NumPy、Torch 等依赖版本。

```bash
repo="$PWD"
export DEX_ISAAC_SIM_ROOT="/path/to/isaac-sim-5.1"
export DEX_ISAAC_PYTHON="/path/to/isaaclab-env/bin/python"
export DEX_ISAAC_PYTHON_OVERLAY="$repo/.isaac-python"
"$DEX_ISAAC_PYTHON" -m pip install --no-index --no-deps --find-links "$repo/wheels" \
  --target "$DEX_ISAAC_PYTHON_OVERLAY" mujoco==3.11.0 onnxruntime==1.31.1

mkdir -p "$repo/out"
bash isaacsim/run.sh --out "$repo/out/isaac-empty" --disable-unmatched-colliders --filter-self
bash isaacsim/run.sh --out "$repo/out/isaac-carry" --carry --disable-unmatched-colliders \
  --filter-self --record-contact-failures --grip-k 19000 --grip-d 53 --grip-rot-k 40 \
  --grip-rot-d 1.2 --substeps 8
```

首次启动会按工况编译所需的 MuJoCo 模型，再用同包 URDF 和网格生成本次运行的 USD。

要在 Isaac Sim 原生窗口观看，在图形桌面运行以下命令。空手路线按墙钟节拍呈现，结束后窗口停在最后一帧，关窗退出。

```bash
bash isaacsim/run.sh --viewer --out "$repo/out/isaac-empty-view" --disable-unmatched-colliders --filter-self
```

## ROS 2 空手遥操作

按 [ROS 2 安装和首次构建](../docs/ROS_MUJOCO_TELEOP.md#安装和首次构建)准备 ROS，在仓库根目录运行。Isaac Sim 与 Teleop 在两个终端使用同一 ROS 域。ROS 传递 Joy 指令，WBC 和 PhysX 分别以 50 Hz 和 200 Hz 运行。

```bash
export ROS_DOMAIN_ID=148
bash isaacsim/run.sh --viewer --teleop --out ./out/isaac-teleop --disable-unmatched-colliders --filter-self
```

```bash
export ROS_DOMAIN_ID=148
./ros_sim/run.sh teleop --isaac
```

开窗静站，按 D → 取消 D → 按 A 后启用三轴控制。前进 0.10–0.40 m/s，后退和横移到 0.30 m/s，转向到 0.45 rad/s，每档 0.05。勾选 C 收步并退出行走使能。勾选 F 归零，取消后恢复当前滑杆指令。Joy 断流 0.30 秒时收步，再起步需要重新按 D → A。关闭 Isaac Sim 窗口可保存记录并退出。可加 `--seconds` 限制运行时长。

[Isaac Sim 遥操作窗口截图](../docs/videos/Isaac-Sim原生窗口.png)

## 评分与视频

评分与双视角视频使用主 README 的离线安装步骤生成的 `.venv`。轨迹由 Isaac Sim 生成，视频由 MuJoCo 渲染。

```bash
repo="$PWD"
source env.sh
"$dex_python" "$repo/isaacsim/score_empty.py" "$repo/out/isaac-empty"
"$dex_python" -m realtime.carry_loaded_walk_report "$repo/out/isaac-carry"
"$dex_python" "$repo/isaacsim/summarize_carry.py" "$repo/out/isaac-carry"
bash "$repo/isaacsim/render.sh" "$repo/out/isaac-empty"
bash "$repo/isaacsim/render_carry.sh" "$repo/out/isaac-carry"
```

[验证结果与已知限制](../docs/ACCEPTANCE.md#isaac-sim-核心结果)
