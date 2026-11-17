# v7 Isaac Sim 5.1 验证补充

2026-11-17 在 luckylhf 上用 Isaac Lab 2.3.0 与 CPU PhysX 跑了原版 record12 EMA ONNX 和 v7 参考规划。WBC 频率 50 Hz，空手 200 Hz，带载复验 400 Hz。运行入口和评分脚本放在本目录，冻结模型由同包 MJCF 与共享网格现场编译。

## 地面接触修复

原场景使用 100 × 100 × 0.10 m 的实体地板，过大的长厚比会触发足底圆柱接触异常。8 秒对照中，保持 0.10 m 厚度，将水平尺寸缩至 10 × 10 m 或改成原生平面均未复现厘米级穿地。改用另一种 USD 缩放方式表达同尺寸，结果逐帧一致。实际验收入口采用已完整复跑的 **2 m** 厚地板，保持顶面 z=0。复现旧配置可传入 `--ground-thickness 0.10`。[地面几何对照](evidence/ground_geometry_probe/results.json)

同一条 75 秒空手六方向路线中，最大足底下沉由 **98.40 mm** 降至 **3.67 mm**，非足底触地由 892 个子步降为 0。4/6 档通过。后退 −0.30 m/s 实测 −0.272 m/s，左横移停车双踝前后差 2.67 cm。[空手逐档门槛](evidence/r15_thick_ground/gate.json) · [旧地板足底数据](evidence/r13/floor_audit.json) · [修正后足底数据](evidence/r15_thick_ground/floor_audit.json)

## 双腕新增 10 kg

0.65 kg 箱壳，两侧抓持点各有 5 kg 配重。5.00–5.24 s 撤除配重托力。箱体是自由 PhysX 刚体，两处弹簧阻尼抓持接口产生双腕虚拟六维力，供手臂、腰和腿部补偿。箱重真值进入物理加载与离线评分。开始接重后 **1.62 s** 收稳，左踝最远移动 **6.02 mm**，超出 5 mm 限值。静站末秒虚拟腕力估计的箱体总重为 **10.652 kg**。首档指令之后严格估重窗口仅 **28.4%** 有效，有效样本平均绝对误差 **0.013 kg**。

前进四档、左右横移六档、左右转向六档全部执行，202 秒且无中途重置。前进 0.20 m/s 和右横移 0.20 m/s 两档通过全部单档门槛。12 档停车平移或航向漂移超限，6 档并步脚差超限。全程无脚间擦碰、交叉、非足底触地或箱体碰地。前进 0.40 m/s 停车窗内根部净移 44.11 cm，右转 0.30 rad/s 稳走箱体横滚峰峰值 12.69°。[逐档门槛与估重](evidence/carry_r02/summary.json) · [完整评测](evidence/carry_r02/metrics.json) · [生效参数](evidence/carry_r02/run.json)

双视角视频可按文末命令生成。动作与测量来自 Isaac Sim，画面由 MuJoCo 离线渲染。

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

评分与离线双视角视频使用主 README 的离线安装步骤生成的 `.venv`。

```bash
repo="$PWD"
source env.sh
"$dex_python" "$repo/isaacsim/score_empty.py" "$repo/out/isaac-empty"
"$dex_python" -m realtime.carry_loaded_walk_report "$repo/out/isaac-carry"
"$dex_python" "$repo/isaacsim/summarize_carry.py" "$repo/out/isaac-carry"
bash "$repo/isaacsim/render.sh" "$repo/out/isaac-empty"
bash "$repo/isaacsim/render_carry.sh" "$repo/out/isaac-carry"
```

