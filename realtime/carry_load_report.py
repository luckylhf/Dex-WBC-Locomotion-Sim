"""Measure settle time from the physical transfer event and verify both feet stay planted."""
import argparse
import json
from pathlib import Path

import numpy as np

from gait import dt
from .carry_model import ARM_ORDER
from .carry_report import angles


WINDOW_S = 1.
ANGLE_DEG = 1.
MOTION_M = .005
WRIST_RANGE_N = 8.
WALKOFF_M = .005
FOOT_MIN_WEIGHT = .10


def evaluate(folder):
    run = json.loads((folder / 'run.json').read_text())
    t0 = run['load_transfer']['start_s']
    fully_loaded = run['load_transfer']['fully_transferred_s']
    trials = run.get('trials', [])
    limit = round(trials[0]['start_s'] / dt) + 1 if trials else None
    keep = slice(None, limit)
    minimum_hold_s = 2. if trials else 5.
    with np.load(folder / 'rollout.npz') as trace:
        t, q = trace['physics_time'][keep], trace['qpos'][keep]
        grip = trace['grip_errors'][keep, :, 0]
        wrist_n = trace['wrist_compensated_world'][keep, :, 0, 2].sum(axis=1)
        balance = trace['balance_trace'][:None if limit is None else limit-1]
        box_contacts = int(trace['box_contact_counts'][len(t)-1])
    telemetry = np.load(folder / 'telemetry.npy')[keep]
    foot_rubs = int(np.load(folder / 'clearance.npy')[len(t)-1, 3])
    completed = (len(t) == limit) if trials else run['completed']
    feet = telemetry[:, 12:18].reshape(-1, 2, 3)
    chest = telemetry[:, 6:8]
    box = q[:, run['box_free_qadr']:]
    box_angles = np.rad2deg(angles(box[:, 3:]))
    clock = np.array_equal(np.arange(len(t)), np.rint(t / dt).astype(int))
    start = int(np.searchsorted(t, t0 - 1e-6))
    full = int(np.searchsorted(t, fully_loaded - 1e-6))
    settled = None
    interval = slice(start, None)
    foot_xy = np.linalg.norm(feet[interval, :, :2] - feet[start-1, :, :2], axis=2)
    foot_z = np.abs(feet[interval, :, 2] - feet[start-1, :, 2])
    support = telemetry[interval, 10:12]
    untouched = bool(foot_xy.size and max(foot_xy.ravel()) <= WALKOFF_M and
                     max(foot_z.ravel()) <= WALKOFF_M and np.min(support) >= FOOT_MIN_WEIGHT)
    stable = []
    n = round(WINDOW_S / dt) + 1
    for i in range(full, len(t)-n+1):
        w = slice(i, i+n)
        ranges = np.ptp(np.column_stack((q[w, :3], box[w, :3])), axis=0)
        plateaus = (np.max(np.ptp(chest[w], axis=0)) <= ANGLE_DEG and
                    np.max(np.ptp(box_angles[w], axis=0)) <= ANGLE_DEG and
                    np.linalg.norm(ranges[:3]) <= MOTION_M and
                    np.linalg.norm(ranges[3:]) <= MOTION_M and
                    np.ptp(wrist_n[w]) <= WRIST_RANGE_N)
        stable.append(plateaus)
    if stable and stable[-1]:
        last_failed = next((i for i in range(len(stable)-1, -1, -1) if not stable[i]), -1)
        candidate = full + last_failed + 1
        if t[candidate] + minimum_hold_s <= t[-1] + 1e-8:
            settled = candidate
    pre = slice(max(0, start-round(WINDOW_S/dt)), start)
    pre_stable = bool(t[-1] >= t0 and np.min(telemetry[pre, 10:12]) >= FOOT_MIN_WEIGHT and
                      np.max(np.ptp(chest[pre], axis=0)) <= ANGLE_DEG and
                      np.max(np.ptp(box_angles[pre], axis=0)) <= ANGLE_DEG and
                      np.linalg.norm(np.ptp(q[pre, :3], axis=0)) <= MOTION_M and
                      np.linalg.norm(np.ptp(box[pre, :3], axis=0)) <= MOTION_M and
                      np.ptp(wrist_n[pre]) <= WRIST_RANGE_N)
    end = slice(max(full, settled or full, len(t)-round(5./dt)), None)
    arms = np.rad2deg(q[:, 7+ARM_ORDER])
    usable = completed and clock and run['physics_resets'] == 0
    passed = bool(usable and pre_stable and untouched and settled is not None and foot_rubs == 0 and
                  box_contacts == 0 and np.max(grip[interval]) <= .015)
    def pair_or_none(x, reducer):
        return reducer(x, axis=0).tolist() if x.size else None
    return dict(passed=passed, completed=completed, continuous_50hz_clock=clock,
                pre_transfer_stable=pre_stable,
                physics_resets=run['physics_resets'], transfer_start_s=t0,
                full_load_s=fully_loaded,
                stable_at_s=float(t[settled]) if settled is not None else None,
                settle_from_start_s=float(t[settled]-t0) if settled is not None else None,
                settle_from_full_load_s=float(t[settled]-fully_loaded) if settled is not None else None,
                sustained_hold_s=float(t[-1]-t[settled]) if settled is not None else None,
                foot_horizontal_max_mm=pair_or_none(foot_xy*1000, np.max),
                foot_vertical_max_mm=pair_or_none(foot_z*1000, np.max),
                feet_min_normal_fraction=pair_or_none(support, np.min),
                grip_peak_mm=float(grip[interval].max()*1000) if grip[interval].size else None,
                foot_rub_steps=foot_rubs, box_contact_steps=box_contacts,
                root_net_xy_mm=(1000*(q[-1, :2]-q[start-1, :2])).tolist()
                    if start < len(t) else None,
                final_arm_max_range_deg=float(np.max(np.ptp(arms[end], axis=0)))
                    if t[end].size else None,
                final_chest_mean_deg=pair_or_none(chest[end], np.mean),
                final_chest_range_deg=pair_or_none(chest[end], np.ptp),
                final_box_mean_deg=pair_or_none(box_angles[end], np.mean),
                final_box_range_deg=pair_or_none(box_angles[end], np.ptp),
                final_wrist_mean_n=float(wrist_n[end].mean()) if wrist_n[end].size else None,
                final_wrist_range_n=float(np.ptp(wrist_n[end])) if wrist_n[end].size else None,
                final_filtered_wrist_n=float(balance[-1, 7]),
                final_estimated_total_kg=float(balance[-1, 7]/9.81),
                final_shift_reference_mm=float(balance[-1, 6]*1000),
                limits=dict(window_s=WINDOW_S, angular_range_deg=ANGLE_DEG,
                            position_range_mm=MOTION_M*1000, wrist_range_n=WRIST_RANGE_N,
                            ankle_displacement_mm=WALKOFF_M*1000,
                            foot_min_normal_fraction=FOOT_MIN_WEIGHT,
                            grip_peak_mm=15., sustained_hold_s=minimum_hold_s))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    result = evaluate(args.folder)
    report = json.dumps(result, indent=2) + '\n'
    (args.folder / 'settle_metrics.json').write_text(report)
    print(report, end='')


if __name__ == '__main__':
    main()
