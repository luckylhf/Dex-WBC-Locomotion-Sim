# ROS 2 与 MuJoCo 空手遥操作

## 安装和首次构建

在 Ubuntu 22.04 x86_64 上，先准备已安装本包依赖的 Python 3.11 环境（默认 `.venv/`）。ROS 接入使用 ROS 2 Humble 和系统 Python 3.10。若系统尚未安装以下依赖，在配置好 ROS 2 Humble 的 APT 源后执行。

```bash
sudo apt install build-essential cmake python3-colcon-common-extensions python3-tk \
  ros-humble-ros-base ros-humble-rosidl-default-generators ros-humble-rosidl-default-runtime
```

离线机器先从同版本 Ubuntu 的 APT 镜像或离线 DEB 集合安装上述系统依赖。仓库的 wheels 包提供 MuJoCo 和 WBC 的 Python 3.11 依赖。系统依赖就绪后，下面的 ROS 构建与启动均可离线执行。

从本包根目录构建一次。

```bash
./ros_sim/run.sh build
```

消息和服务定义随包提供，位于 [ros_ws/src/bodyctrl_msgs](../ros_ws/src/bodyctrl_msgs)。构建、安装和日志输出写入 `out/ros_ws/`。更新消息定义或更换 ROS 环境后再运行上述构建命令。

[ros_sim/dex_rl](../ros_sim/dex_rl) 随包携带 Dex-RL 的 Joy 发布器和 29 轴电机 ID 映射。构建和运行使用本目录的源码、构建产物及系统 ROS。

## 启动

在本包根目录分别开启这两个终端，两边使用相同的 ROS 域。

```bash
export ROS_DOMAIN_ID=147
./ros_sim/run.sh --viewer --out ./out/teleop-01
```

```bash
export ROS_DOMAIN_ID=147
./ros_sim/run.sh teleop
```

无显示时可省去 `--viewer`。每次仿真选择新的输出目录。命令、轨迹、足间净空和墙钟下发时间随该目录保存。`--seconds` 可以限制交互仿真时长。

`DEX_ROS_SETUP` 可以指定 ROS 的 `setup.bash`，`DEX_ROS_PYTHON` 可以指定与该 ROS 配套的 Python，`DEX_V7_PYTHON` 可以指定本包的 MuJoCo/WBC Python。ROS 和 WBC 两个进程各用自己的 Python。

开窗后保持静站。依次勾选 **D**、取消 **D**、勾选 **A**，滑杆开始控制运动。**C** 让机器人收步并退出行走使能。再起步时，先松开按键，再重复启动顺序。

| 滑杆 | 有效速度档位 |
| --- | --- |
| Y1 前后 | 后退 −0.30 至 0、前进 0.10 至 1.20 m/s，每档 0.05 m/s。前进从 0 直接到 0.10 m/s。 |
| X1 横移 | −0.30 至 +0.30 m/s，每档 0.05 m/s，正值向左。 |
| X2 转向 | −0.45 至 +0.45 rad/s，每档 0.05 rad/s，正值向左。 |

三轴可同时使用，正值为前进、左移、左转。「滑杆回零」、勾选 **F** 或 Joy 断流 0.30 s 均触发收步。F 保留行走使能，取消勾选后会恢复当前滑杆指令。

## 闭环与时钟

仿真节点把 29 轴位置、速度和根 IMU 经 ROS 消息发给控制节点。控制节点使用该消息构造 WBC 观测。WBC 每 20 ms 推理一次，最近一次串联关节目标连续用两拍，每拍通过腿、腰、臂的 `CmdMotorCtrl` 下发。仿真节点收到全套电机消息后推进两个 5 ms 物理子步。整条链使用冻结模型、增益和 record12 ONNX。RC 输入、仿真电机消息和反馈通过 DDS 传输，进程间管道用于隔离 ROS 与冻结仿真的 Python 环境。

电机话题使用 `/dex_wbc_sim` 前缀，随启动限制到本机 ROS 域。该入口的终点是冻结 MuJoCo 模型，电机消息中的目标仍在串联关节空间。真机踝串并联转换沿用单独的部署方案。

查看器以独立 60 Hz 节拍呈现最新仿真位姿。在 MuJoCo 窗口拖动可转视角，滚轮缩放。

[启动、走停与时钟复验数据](../verification/ros_mujoco_teleop.json)
