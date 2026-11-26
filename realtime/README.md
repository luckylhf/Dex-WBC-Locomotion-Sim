# v7 实时速度指令

本机 UDP 输入支持前向 0.1–0.4 m/s，发送 0 触发减速停车。MuJoCo 进程在每个 50 Hz 控制周期处理最新指令。正常走时每 0.50 秒安排一次交替落脚，按指令调整一步的前进量。在线更新保留 WBC 已经看到的参考帧和已进入的支撑阶段，从后续步的边界更新落点。停车用两步缩短步幅，最后再安排后脚跟上。重新收到非零指令时，从当前站姿移重后起步。同一 MuJoCo 状态、WBC 历史和物理时钟贯穿运行。

## 本机输入

在 luckylhf 的 v7 目录运行仿真。

```bash
source env.sh
"$dex_python" -m realtime.sim --pace --port 18070 --duration 90 --out /tmp/dex-v7-live
```

从同一台主机的另一个终端发送一次速度。连续发送可随时更改目标，最后一次有效指令保持到下一次有效指令。

```bash
source env.sh
"$dex_python" -c 'import socket, sys; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.sendto(sys.argv[1].encode(),("127.0.0.1",18070))' 0.2
```

内置复现按仿真时刻通过同一个 UDP 接口发送验收序列。以下命令使用与视频相同的仿真、评测和渲染入口。

```bash
source env.sh
"$dex_python" -m realtime.sim --demo --out /tmp/dex-v7-live-demo
"$dex_python" -m realtime.report /tmp/dex-v7-live-demo
MUJOCO_GL=egl "$dex_python" -m realtime.video /tmp/dex-v7-live-demo /tmp/dex-v7-live-demo/front_side.mp4
```

## 视频对应的运行

视频对应 `experiments/live_external`。一个单独的进程在墙钟 3、9、15、21、27、33、39 秒逐次发送 0.1、0.2、0.3、0.4、0.2、0、0.1 m/s。仿真以 50 Hz 节拍运行，在下一控制周期接收指令。四次非零换速的首个新步计划在指令到达约 0.50 秒后生效。全程 46 秒、9,200 次 200 Hz 物理子步、0 次中途重置。规划、推理和步进的同周期耗时 p95 为 5.63 ms，最大 7.90 ms。

| 速度指令 m/s | 稳态实测均速 m/s | 有效单步前进量 cm |
| --- | --- | --- |
| 0.10 | 0.101 | 5.09 |
| 0.20 | 0.202 | 10.08 |
| 0.30 | 0.298 | 14.90 |
| 0.40 | 0.401 | 20.04 |
| 0.20 | 0.201 | 10.04 |
| 再起步 0.10 | 0.101 | 5.07 |

以实际竖直反力识别落脚，连续行走段平均为 2.001 Hz，换速期间左右脚继续交替。有效单步前进量按各段的实速与实际落脚频率计算。33.02 秒接到停车指令，经过两步减速后，约 34.52 秒进入右脚跟步。右脚移动 2.13 cm，停车后按机身朝向投影的双踝前后差为 0.44 cm。39.02 秒收到再起步指令，并走到结束。全程无非足部触地、双脚交叉、脚与对侧脚或小腿的穿透接触。

[连续正侧面视频](release/front_side.mp4) · [实测指标](release/metrics.json) · [指令和步态时间线](release/run.json)

以上为这一路线的 CPU MuJoCo 验证。运行使用 v7 的冻结 WBC 和已校准参数，新增实时生成的运动学参考。独立的 0–1.2 m/s 连续路线见 [扩速试验](../experiments/speed_ladder/RESULTS.md)。真机留待后续。

## 可交互 viewer

两个入口共用 `viewer_common` 的按键、50 Hz 节拍与落盘逻辑，都是 MuJoCo 被动 viewer、200 Hz 物理，关窗或到时落盘。批处理验收入口不受影响。

```bash
source env.sh
"$dex_python" -m realtime.carry_viewer --out /tmp/dex-carry-viewer                 # 0.65 kg 空箱
"$dex_python" -m realtime.carry_viewer --added-mass 10 --out /tmp/dex-carry-10kg   # 腕端各 5 kg
"$dex_python" -m realtime.carry_viewer --content-mass 10 --out /tmp/dex-carry-box10  # 箱内物品 10 kg
"$dex_python" -m realtime.hand_viewer --out /tmp/dex-hand-viewer                    # 空手，无箱
```

`--mass` 是空箱壳体质量，`--added-mass` 是两手抓持点各加一半的配重，`--content-mass` 是躺在箱底的物品质量（约 6 cm 高的一块，重心在箱底上方 3 cm，无额外碰撞形状）。三者相加即总重（空箱 0.65 kg；`--content-mass 10` 为 10.65 kg）。窗口内不落盘，退出时写 `telemetry.npy`、`clearance.npy`、`rollout.npz`、`run.json`，空手入口另写 `rollout.trajectory.npz` 与 `commands.npz`；指定 `--out` 时抱箱入口另存 `box_model.mjb`。产物字段与批处理入口一致，`realtime.carry_report` 可直接读取，但交互路线不按验收时序走停，报告的分段与速度检查不代表验收结论。

带载（`--content-mass` 或 `--added-mass` 配 `--brace`）会自动打开 `realtime.carry_load_transfer` 里的 `BracedStanding`：用腕部实测力驱动腰后仰、腿部移重和手臂阻尼，并在 `--transfer-start`（默认 5 s）把外部承重托架撤掉，复现"搬起箱子"的接重过程。`--no-brace` 可关掉这层补偿（10 kg 下会明显下沉）。箱内物品的重心在箱子中间，比腕端配重点前移 0.157 m，因此走 `content_profile` 的混合标定：前进用空箱档位乘 1.27，横移/转向沿用带载档位；`--lean-gain`/`--com-gain` 默认取 0.30/1.25（腕端配重仍为 0.20/1.0）。

按键：`W`/`S`（或 `↑`/`↓`）调 vx，`A`/`D`（或 `←`/`→`）调 vy，`Q`/`E` 调转向角速度 wz，`SPACE`、`C` 或 `0` 全部归零，`1`–`4` 只把 vx 设为 0.10、0.20、0.30、0.40，`G` 开关抓持（空手入口无箱，按下只提示），`R` 重置到起始站姿，`P` 打印估重（空手入口打印骨盆高度与胸腔姿态），`F` 存盘，`ESC` 退出。各轴按标定档位步进：vx 为 0、±0.10、±0.15…0.40（正向到 0.40，后退到 −0.30），vy 为 0、±0.05…±0.30，wz 为 0、±0.15、±0.30、±0.45。改速仍受规划前瞻限制，指令约 0.25 s 后才进入参考。`R` 重置前会把上一段写到 `segment_N/`。窗口按键需先点击窗口取得焦点；若某个键没反应，用 `--log-keys` 启动可看到窗口实际送达的 GLFW 键码。

三轴走的是 `realtime.carry_omni` 的 `CarryOmniPreview` 与 `carry_omni_calibration.json`（空手入口用 `release/calibration.json` 加 `realtime/omni_calibration.json` 的 `OmniPreview`），即 19 档六方向试验的那套标定；前进档与各自批处理入口的验证结果一致。同时给两个轴（例如边走边转）属于标定外的组合，可跑但不在已验证范围内。

本机实测（无头，同一路线：持握静立 → 0.1 m/s 前进 → 停 → 原地左转 0.3 rad/s → 停，窗口分别取 11–13.8 s、17–20 s、24–26.5 s）：

| 入口 | 前进实速 | 转向实速 | 停车残余速度 | 估重 |
| --- | --- | --- | --- | --- |
| 空箱 0.65 kg | 0.101 m/s | 0.21 rad/s | 0.010 m/s | 0.650 kg |
| 箱内物品 10 kg | 0.109 m/s | 0.20 rad/s | 0.023 m/s | 10.64 kg（真值 10.65） |
| 腕端配重 10 kg | 0.103 m/s | 0.28 rad/s | 0.010 m/s | 10.65 kg（真值 10.65） |

转向实速普遍低于 0.3 rad/s 指令：空箱在这一路线也只有 0.21，属 `OmniPreview` 的既有跟踪误差；腕端配重档因为用带载转向标定能到 0.28。箱内物品的重心比腕端配重点前移 0.157 m，腰腿补偿按腕端标定，转完停车的收敛靠提高 `--com-gain`（1.25）换来，代价是前进略快、转向偏低。需要另行调整时可用 `--lean-gain`/`--com-gain`，已超出已验证范围。估重列取接重（5 s）之后的有效窗口。

无窗口复现同一条闭环：`"$dex_python" -m realtime.carry_viewer --headless --script "5:0.1,15:0.2,25:0" --duration 30 --out /tmp/dex-carry-headless`，空手入口同样支持 `--headless`/`--script`。脚本项可写 `t:vx` 或 `t:vx/vy/wz`，例如 `--script "3:0,8:0/0.2,13:0,18:0/0/0.3,23:0"` 依次跑横移和原地转向。viewer 走 GLFW 窗口，不要设 `MUJOCO_GL=egl`；远程会话需要 X 转发。窗口刷新默认 30 Hz（`--render-hz`），低于 50 Hz 控制节拍，因为本模型一次完整场景同步约 17 ms；控制仍按 20 ms 节拍推进，实测 12 s 仿真用 12.05 s 墙钟。覆盖文字默认 2 Hz（`--overlay-hz`）：本机 macOS 实测 `set_texts` 每次约 100 ms；若按每帧提交，控制循环会等待 viewer 的渲染循环消费文字请求，仿真时钟明显变慢。Linux 的 viewer 虽走独立线程，但同一套 C++ 文字提交在请求未及时消费时也可能阻塞。带重接重转移（`realtime.carry_load_transfer`）和验收记录仍用各自的批处理入口。
