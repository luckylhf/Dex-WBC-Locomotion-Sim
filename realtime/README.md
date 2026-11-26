# MuJoCo 本机实时控制

本机 UDP 输入支持前向 0.1–0.4 m/s，发送 0 触发减速停车。MuJoCo 进程在每个 50 Hz 控制周期处理最新指令。正常走时每 0.50 秒安排一次交替落脚，按指令调整一步的前进量。在线更新保留 WBC 已经看到的参考帧和已进入的支撑阶段，从后续步的边界更新落点。停车用两步缩短步幅，最后再安排后脚跟上。重新收到非零指令时，从当前站姿移重后起步。同一 MuJoCo 状态、WBC 历史和物理时钟贯穿运行。

## 本机输入

在仓库根目录准备空手模型并启动仿真。

```bash
source env.sh
"$dex_python" tools/compile_models.py model
"$dex_python" -m realtime.sim --pace --port 18070 --duration 90 --out ./out/dex-v7-live
```

从同一台主机的另一个终端发送一次速度。连续发送可随时更改目标，最后一次有效指令保持到下一次有效指令。

```bash
source env.sh
"$dex_python" -c 'import socket, sys; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.sendto(sys.argv[1].encode(),("127.0.0.1",18070))' 0.2
```

空手模型就绪后，可用内置路线复跑 UDP 指令，随后评估和渲染。

```bash
source env.sh
"$dex_python" -m realtime.sim --demo --out ./out/dex-v7-live-demo
"$dex_python" -m realtime.report ./out/dex-v7-live-demo
MUJOCO_GL=egl "$dex_python" -m realtime.video ./out/dex-v7-live-demo ./out/dex-v7-live-demo/front_side.mp4
```

[UDP 路线实测指标](release/metrics.json) · [指令和步态时间线](release/run.json)

## 可交互 viewer

两个窗口共用按键和记录格式。控制频率为 50 Hz，物理频率为 200 Hz，默认渲染为 30 Hz，关窗或到时写入输出目录。需要可用的 GLFW 图形会话。

```bash
source env.sh
"$dex_python" -m realtime.carry_viewer --out ./out/dex-carry-viewer                 # 0.65 kg 空箱
"$dex_python" -m realtime.carry_viewer --added-mass 10 --out ./out/dex-carry-10kg   # 腕端各 5 kg
"$dex_python" -m realtime.carry_viewer --content-mass 10 --out ./out/dex-carry-box10  # 箱内物品 10 kg
"$dex_python" -m realtime.hand_viewer --out ./out/dex-hand-viewer                    # 空手，无箱
```

`--mass` 是空箱壳体质量，`--added-mass` 是两手抓持点各加一半的配重，`--content-mass` 是躺在箱底的物品质量（约 6 cm 高的一块，重心在箱底上方 3 cm，无额外碰撞形状）。三者相加即总重（空箱 0.65 kg；`--content-mass 10` 为 10.65 kg）。窗口内不落盘，退出时写 `telemetry.npy`、`clearance.npy`、`rollout.npz`、`run.json`，空手入口另写 `rollout.trajectory.npz` 与 `commands.npz`；指定 `--out` 时抱箱入口另存 `box_model.mjb`。产物字段与批处理入口一致，`realtime.carry_report` 可直接读取，但交互路线不按验收时序走停，报告的分段与速度检查不代表验收结论。

带载（`--content-mass` 或 `--added-mass` 配 `--brace`）会自动打开 `realtime.carry_load_transfer` 里的 `BracedStanding`：用腕部实测力驱动腰后仰、腿部移重和手臂阻尼，并在 `--transfer-start`（默认 5 s）把外部承重托架撤掉，复现"搬起箱子"的接重过程。`--no-brace` 可关掉这层补偿（10 kg 下会明显下沉）。箱内物品的重心在箱子中间，比腕端配重点前移 0.157 m，因此走 `content_profile` 的混合标定：前进用空箱档位乘 1.27，横移/转向沿用带载档位；`--lean-gain`/`--com-gain` 默认取 0.30/1.25（腕端配重仍为 0.20/1.0）。

按键：`W`/`S`（或 `↑`/`↓`）调 vx，`A`/`D`（或 `←`/`→`）调 vy，`Q`/`E` 调转向角速度 wz，`SPACE`、`C` 或 `0` 全部归零，`1`–`4` 只把 vx 设为 0.10、0.20、0.30、0.40，`G` 开关抓持（空手入口无箱，按下只提示），`R` 重置到起始站姿，`P` 打印估重（空手入口打印骨盆高度与胸腔姿态），`F` 存盘，`ESC` 退出。各轴按标定档位步进：vx 为 0、±0.10、±0.15…0.40（正向到 0.40，后退到 −0.30），vy 为 0、±0.05…±0.30，wz 为 0、±0.15、±0.30、±0.45。改速仍受规划前瞻限制，指令约 0.25 s 后才进入参考。`R` 重置前会把上一段写到 `segment_N/`。窗口按键需先点击窗口取得焦点；若某个键没反应，用 `--log-keys` 启动可看到窗口实际送达的 GLFW 键码。

三轴走的是 `realtime.carry_omni` 的 `CarryOmniPreview` 与 `carry_omni_calibration.json`（空手入口用 `release/calibration.json` 加 `realtime/omni_calibration.json` 的 `OmniPreview`），即 19 档六方向试验的那套标定；前进档与各自批处理入口的验证结果一致。同时给两个轴（例如边走边转）属于标定外的组合，可跑但不在已验证范围内。

无窗口复现同一条闭环：`"$dex_python" -m realtime.carry_viewer --headless --script "5:0.1,15:0.2,25:0" --duration 30 --out ./out/dex-carry-headless`，空手入口同样支持 `--headless`/`--script`。脚本项可写 `t:vx` 或 `t:vx/vy/wz`，例如 `--script "3:0,8:0/0.2,13:0,18:0/0/0.3,23:0"` 依次跑横移和原地转向。
