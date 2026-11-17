"""Summarize the PhysX transfer, wrist estimates, and existing v7 route gates."""
import argparse
import json
from pathlib import Path

import numpy as np
from gait import dt


def report(folder):
    run = json.loads((folder / "run.json").read_text())
    metrics = json.loads((folder / "metrics.json").read_text())
    pose = np.load(folder / "rollout.npz")
    time = pose["physics_time"]
    filtered = pose["balance_trace"][:, 7] / 9.81
    estimate = pose["estimates"]
    rows = []
    for trial in metrics["trials"]:
        result = trial.get("result") or {}
        flags = [key for key, value in result.items() if isinstance(value, bool) and not value]
        if not trial["completed"] or not result:
            flags.append("incomplete_or_metrics_unavailable")
        if trial.get("nonsupport_ground_substeps", 0):
            flags.append("non_foot_ground_contact")
        rows.append({
            "label": trial["label"], "command": trial["command"],
            "completed": trial["completed"], "passed": not flags, "failed": flags,
            "start_s": trial["start_s"], "zero_s": trial["zero_s"], "end_s": trial["end_s"],
            "steady": trial.get("steady"), "park": trial.get("park"), "box": trial.get("box"),
            "motion": trial.get("motion"),
            "foot_rub_substeps": trial.get("foot_rub_substeps"),
            "nonsupport_ground_substeps": trial.get("nonsupport_ground_substeps"),
            "crossing_frames": trial.get("crossing_frames"),
            "minimum_gaps_m": [trial.get("minimum_foot_clearance_m"),
                               trial.get("minimum_foot_to_opposite_shin_m")],
            "max_swing_inward_m": trial.get("max_swing_inward_m"),
        })
    walked = [row for row in rows if row["motion"]]
    mass_real = run["model_box_mass_kg"]
    after = time[:len(filtered)] >= run["load_transfer"]["fully_transferred_s"]
    tail_stand = (time[:len(filtered)] >= 9.) & (time[:len(filtered)] < 10.)
    if len(time) > 10.:
        e = estimate[round(10 / dt):]
        valid = e[:,2] > 0
        walking_valid_fraction = float(np.mean(valid)) if len(e) else None
        valid_error = e[valid,1] - mass_real
    else:
        walking_valid_fraction, valid_error = None, []
    result = dict(
        passed=bool(metrics["passed"] and run["nonsupport_ground_substeps"] == 0
                    and run["box_ground_steps"] == 0 and all(r["passed"] for r in rows)),
        physics_resets=run["physics_resets"], physics_steps=run["physics_steps"],
        duration_s=run["duration_s"], scope_reached_all=len(rows) == 16 and all(r["completed"] for r in rows),
        completed_slots=sum(r["completed"] for r in rows), passed_slots=sum(r["passed"] for r in rows),
        foot_rub_substeps=run["rub_substeps"], nonfoot_ground_substeps=run["nonsupport_ground_substeps"],
        box_contact_substeps=run["box_contact_steps"], box_ground_substeps=run["box_ground_steps"],
        total_mass_kg=mass_real, transfer=metrics["initial_transfer"],
        wrist_filtered_total_kg_last_stand_second=float(filtered[tail_stand].mean()) if np.any(tail_stand) else None,
        wrist_filtered_total_kg_post_transfer_mean=float(filtered[after].mean()) if np.any(after) else None,
        wrist_filtered_steady_kg_range=(
            [float(min(r["motion"]["steady_filtered_wrist_total_kg"] for r in walked)),
             float(max(r["motion"]["steady_filtered_wrist_total_kg"] for r in walked))] if walked else None),
        strict_valid_fraction_after_first_command=walking_valid_fraction,
        strict_valid_abs_error_kg_mean=float(np.abs(valid_error).mean()) if len(valid_error) else None,
        maximum_grip_mm=metrics["all_frames_grip_error_mm"],
        maximum_compensation_rad=float(np.max(np.abs(pose["load_compensation_joint_delta_rad"]))),
        trials=rows,
    )
    (folder / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    result = report(p.parse_args().folder)
    print(json.dumps({key: val for key, val in result.items() if key != "trials"}, indent=2))
