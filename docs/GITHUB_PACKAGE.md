# Dex-WBC-Locomotion-Sim 仓库交付

Git 仓库包含源码、网格、ONNX、标定、文档和验收指标。离线依赖位于 `packages/`。

| 用途 | 文件 |
| --- | --- |
| Linux x86_64 / Python 3.11 离线 wheels | `packages/Dex-WBC-Locomotion-Sim-Linux-x86_64-Py311-wheels.tar.zst` |
| 离线 wheels SHA-256 | `packages/Dex-WBC-Locomotion-Sim-Linux-x86_64-Py311-wheels.tar.zst.sha256` |

通过分支或标签定位同一版本的源码和 wheels。完整克隆后可直接按 [README](../README.md#克隆后离线安装) 校验、解压 wheels 和安装。仓库根目录的 `SHA256SUMS` 校验源码、模型资产和文档，离线 wheels 由同目录的校验文件核验。

MuJoCo 的 .mjb 与 Isaac Sim 的 USD 由运行入口按需生成。生成缓存、解压的 wheels、虚拟环境、视频和运行输出均由 Git 忽略。源文件压缩备份保存在本地备份目录。
