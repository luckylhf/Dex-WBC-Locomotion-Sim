"""Evaluate the initial load transfer and loaded start/stop trials."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gait import dt
from .carry_load_report import evaluate as evaluate_transfer
from .carry_omni_report import report as evaluate_trials
from .carry_report import angles
from .omni_preview import yaw_rot


def evaluate(folder):
    metrics = evaluate_trials(folder)
    transfer = evaluate_transfer(folder)
    run = json.loads((folder / 'run.json').read_text())
    with np.load(folder / 'rollout.npz') as trace:
        q = trace['qpos']
        clock = trace['physics_time']
        measured = trace['balance_trace']
    position = q[:, run['box_free_qadr']:]
    box_angles = np.rad2deg(angles(position[:, 3:]))[:, [1, 0]]
    yaw = angles(q[:, 3:7])[:, 2]
    telemetry = np.load(folder / 'telemetry.npy')
    feet = telemetry[:, 12:18].reshape(-1, 2, 3)
    chest_angles = telemetry[:, 6:8]
    blocks = run['planned_blocks']
    foot_model = None
    for trial in metrics['trials']:
        if not trial['completed'] or trial['park'] is None:
            continue
        start, zero, end = [round(trial[k] / dt) for k in ('start_s', 'zero_s', 'end_s')]
        steady = slice(start + round(2.5 / dt), zero)
        window = slice(end - round(1. / dt), end+1)
        gather = next(b for b in blocks if b['kind'] == 'gather' and trial['zero_s'] <= b['start_s'] < trial['end_s'])
        foot = 'LR'.index(gather['side'])
        landings = [e for e in metrics['foot_events'] if e['foot'] == gather['side']
                    and gather['start_s']-.10 <= e['land_s'] <= gather['end_s']+.35]
        landed = landings[0] if landings else None
        before = round(landed['release_s']/dt)-1 if landed else round(gather['start_s']/dt)
        after = round(landed['land_s']/dt) if landed else round(gather['end_s']/dt)
        vx, vy, omega = trial['command']
        direction = yaw_rot(yaw[before])[:2, 0 if vx else 1] * np.sign(vx or vy)
        delta = (feet[[before, after], 1-foot, :2] - feet[[before, after], foot, :2]) @ direction
        closing = bool(delta[0] > 0. and abs(delta[1]) < delta[0])
        turn_delta = None
        if omega:
            if foot_model is None:
                foot_model = mujoco.MjModel.from_binary_path(str(folder/'box_model.mjb'))
                state = mujoco.MjData(foot_model)
                feet_ids = [foot_model.body(f'robot/ankle_roll_{side}_link').id for side in ('l', 'r')]
            separation = []
            for frame in (before, after, end):
                state.qpos[:] = q[frame]
                mujoco.mj_kinematics(foot_model, state)
                rotation = state.xmat[feet_ids].reshape(2, 3, 3)
                angles_rad = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
                diff = angles_rad[1-foot] - angles_rad[foot]
                separation.append(float(np.rad2deg(np.arctan2(np.sin(diff), np.cos(diff)))))
            turn_delta = separation
            closing = abs(turn_delta[1]) < abs(turn_delta[0])
        extra = [e for e in metrics['foot_events'] if gather['end_s']+.35 < e['land_s'] <= trial['end_s']]
        park = trial['park']
        root_shift = float(np.linalg.norm(np.ptp(q[window, :2], axis=0)))
        angle_range = np.maximum(np.ptp(chest_angles[window], axis=0), np.ptp(box_angles[window], axis=0))
        quiet = bool(root_shift <= .005 and angle_range.max() <= 1.
                     and np.min(telemetry[window, 10:12]) >= .10)
        trial['motion'] = dict(
            steady_chest_pitch_roll_range_deg=np.ptp(chest_angles[steady], axis=0).tolist(),
            steady_box_pitch_roll_range_deg=np.ptp(box_angles[steady], axis=0).tolist(),
            steady_filtered_wrist_total_kg=float(measured[steady, 7].mean() / 9.81),
            park_last_second_chest_pitch_roll_range_deg=np.ptp(chest_angles[window], axis=0).tolist(),
            park_last_second_box_pitch_roll_range_deg=np.ptp(box_angles[window], axis=0).tolist(),
            park_last_second_root_range_m=root_shift,
            park_last_second_quiet=quiet,
            gather_began_on_trailing_foot=bool(landed and delta[0] > 0.) if not omega else None,
            gather_trailing_distance_before_after_m=delta.tolist() if not omega else None,
            gather_foot_yaw_gap_before_after_final_deg=turn_delta,
            gather_contact_s=landed['land_s'] if landed else None,
            later_landings=extra,
        )
        trial['result']['trailing_foot_gathered'] = bool(landed and closing and not extra)
        trial['result']['quiet_last_second'] = quiet
    metrics['initial_transfer'] = transfer
    metrics['continuous_clock'] = bool(np.allclose(np.diff(clock), dt, rtol=0, atol=1e-10))
    metrics['passed'] = bool(metrics['completed'] and metrics['continuous_clock'] and metrics['physics_resets'] == 0
                             and transfer['passed'] and len(metrics['trials']) == (16 if run['options'].get('omni') else 4)
                             and all(trial.get('result') and all(v for v in trial['result'].values() if isinstance(v, bool))
                                     for trial in metrics['trials']))
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('folder', type=Path)
    evaluate(parser.parse_args().folder)
