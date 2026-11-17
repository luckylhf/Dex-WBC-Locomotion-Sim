"""Generate the six-axis acceptance gate and a concise per-axis summary."""
import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("out", type=Path)
out = parser.parse_args().out
metrics = json.loads((out / "metrics.json").read_text())
run = json.loads((out / "run.json").read_text())
trace = np.load(out / "telemetry.npy")
stand = trace[(trace[:,0] >= 1.) & (trace[:,0] <= 3.)]
stand_drift = float(np.max(np.linalg.norm(stand[:,1:3] - stand[0,1:3], axis=1))) if len(stand) else None
stand_angle = float(np.abs(stand[:,6:8]).max()) if len(stand) else None
stand_min_load = float(stand[:,10:12].min()) if len(stand) else None
static_ok = bool(len(stand) == 101 and stand_drift <= .025 and stand_angle < 10.
                 and stand_min_load > .10 and not np.any(stand[:,18:20]))
limits = dict(speed_mps=.025, yaw_radps=.035, cadence_hz=[1.9,2.1], park_xy_m=.025,
              park_yaw_deg=.5, park_ankle_gap_m=.025, parking_min_foot_load_bw=.10)
results = []
for trial in metrics["trials"]:
    failures = []
    axis = int(np.argmax(np.abs(trial["command"])))
    value = hz = None
    if not trial["completed"]:
        failures.append("Incomplete")
    if trial.get("steady") is None:
        failures.append("Steady gait unavailable")
    else:
        steady = trial["steady"]
        value, hz = steady["actual"][axis], steady["cadence_hz"]
        if abs(value - trial["command"][axis]) > limits["yaw_radps" if axis == 2 else "speed_mps"]:
            failures.append("Primary speed")
        if hz is None or not 1.9 <= hz <= 2.1 or steady["repeat_foot_events"]:
            failures.append("2 Hz alternating cadence")
    park = trial.get("park")
    if park is None:
        failures.append("Parking window unavailable")
    else:
        if park["xy_net_drift_m"] > limits["park_xy_m"] or abs(park["yaw_drift_deg"]) > limits["park_yaw_deg"]:
            failures.append("Parking drift")
        if park["final_ankle_gap_m"] > limits["park_ankle_gap_m"]:
            failures.append("Final ankle alignment")
        if min(park["final_load_bw"]) < limits["parking_min_foot_load_bw"]:
            failures.append("Bilateral support")
    if any(trial.get(key, 0) for key in ("crossing_frames", "foot_rub_substeps", "nonsupport_ground_substeps")):
        failures.append("Foot crossing, rub, or non-foot ground contact")
    results.append(dict(label=trial["label"], primary_command=trial["command"][axis],
                        primary_actual=value, cadence_hz=hz, park=park,
                        max_swing_inward_m=trial.get("max_swing_inward_m"), failures=failures))
contact_ok = (run.get("ground_contact_peak_impulse_ns", 0.) > 0.
              and run["rub_substeps"] == run["nonsupport_ground_substeps"] == 0)
gate = dict(passed=bool(run["completed"] and static_ok and contact_ok and all(not x["failures"] for x in results)),
            contact_passed=contact_ok,
            stand=dict(passed=static_ok, drift_m=stand_drift, peak_chest_deg=stand_angle,
                       minimum_foot_load_bw=stand_min_load),
            limits=limits, trials=results)
(out / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")
print("V7_GATE", gate["passed"], "STATIC", gate["stand"])
for entry in results:
    park = entry["park"] or {}
    print(entry["label"], entry["primary_actual"], entry["cadence_hz"],
          {key: park.get(key) for key in ("xy_net_drift_m","yaw_drift_deg","final_ankle_gap_m")},
          "; ".join(entry["failures"]))
