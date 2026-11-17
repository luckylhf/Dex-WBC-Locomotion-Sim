# Dex-WBC-Locomotion-Sim 版本说明

## 1.0.0 源文件交付

提供 Dex 站走停与抱箱负载补偿仿真。参考参数经世界模型辅助标定。

- GitHub 仓库直接提供冻结 ONNX、MJCF、URDF、共享网格、运行和验收源码。
- MuJoCo 的四种模型工况与 Isaac Sim 的空手、双腕新增 10 kg 工况共用一套显示 STL 和一套凸包 STL。MuJoCo 首跑按需编译并校验生成的 `.mjb`，Isaac Sim 从相同网格导入 USD。
- 仓库 `packages/` 中的离线依赖包为 `Dex-WBC-Locomotion-Sim-Linux-x86_64-Py311-wheels.tar.zst`。`install.sh` 通过本地 wheels 安装，运行环境为 Linux x86_64、Python 3.11。Isaac Sim 5.1 / Isaac Lab 2.3.0 需单独准备。
- 用户可以使用已有 1.0.0 运行环境，运行入口、冻结策略与原验收口径一致。四个 .mjb 现场重编译后逐个通过原版 SHA-256，四个 MuJoCo 工况均完整复跑通过。
- Isaac Sim 的源文件版空手 3 秒及带载接重 10 秒与存档逐值一致。已存档的完整 Isaac 路线上，空手 4/6 档通过，双腕新增 10 kg 的带载路线 2/16 档通过。详见 [Isaac Sim 验证结果](../isaacsim/README.md)。

安装步骤见 [README](../README.md#克隆后离线安装)。资产许可见 [Dex 资产许可证](../licenses/Dex-asset-LICENSE.txt)，Python 依赖许可随各 wheel 的元数据保留。
