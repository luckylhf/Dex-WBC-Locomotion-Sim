# Dex-WBC-Locomotion-Sim 版本说明

## 1.0.0 源文件交付

提供 Dex 站走停与抱箱负载补偿仿真。参考参数经世界模型辅助标定。

- GitHub 仓库直接提供冻结 ONNX、MJCF、URDF、共享网格、运行和验收源码。
- MuJoCo 的四种模型工况与 Isaac Sim 的空手、双腕新增 10 kg 工况共用一套显示 STL 和一套凸包 STL。MuJoCo 首跑按需编译并校验生成的 `.mjb`，Isaac Sim 从相同网格导入 USD。
- 仓库 `packages/` 中的离线依赖包为 `Dex-WBC-Locomotion-Sim-Linux-x86_64-Py311-wheels.tar.zst`。`install.sh` 通过本地 wheels 安装，运行环境为 Linux x86_64、Python 3.11。Isaac Sim 5.1 / Isaac Lab 2.3.0 需单独准备。
- 空手 ROS 2 遥操作附带 `bodyctrl_msgs` 定义、Joy 发布器和电机 ID 映射。安装 ROS 2 Humble 系统依赖后，从当前仓库执行 `./ros_sim/run.sh build` 生成消息接口。[构建和启动说明](ROS_MUJOCO_TELEOP.md)
- MuJoCo 四种工况的缓存可按冻结哈希复建。MuJoCo、Isaac Sim 和 ROS 遥操作的验收范围及结果见 [仿真验收](ACCEPTANCE.md)。

安装步骤见 [README](../README.md#克隆后离线安装)。资产许可见 [Dex 资产许可证](../licenses/Dex-asset-LICENSE.txt)，Python 依赖许可随各 wheel 的元数据保留。
