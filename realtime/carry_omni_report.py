"""Score the empty-box speed ladder from robot and box states and wrist wrenches."""
import argparse
import json
from pathlib import Path

import numpy as np

from gait import dt
from .carry_report import angles
from .omni_report import evaluate


def report(folder):
    m=evaluate(folder,trajectory='rollout.npz',block_key='planned_blocks',save=False)
    r=json.loads((folder/'run.json').read_text())
    z=np.load(folder/'rollout.npz')
    start=r['box_free_qadr']
    box_angles=np.rad2deg(angles(z['qpos'][:,start+3:start+7]))
    contact_counts=z['box_contact_counts']
    grips=z['grip_errors']
    support=z['grip_wrench_on_box'][...,2]
    masses=z['estimates']
    held=z['grip_active']
    for trial in m['trials']:
        if not trial.get('reached'):
            continue
        begin,end=[round(trial[k]/dt) for k in ('start_s','end_s')]
        end=min(end,len(grips)-1)
        motion_start=round((trial['start_s']+2.5)/dt)
        motion_end=min(round(trial['zero_s']/dt),end)
        sl=slice(begin,end+1)
        steady=slice(motion_start,motion_end)
        sec=masses[steady]
        valid=sec[:,2]>0
        force=support[steady]
        trial['box']=dict(contact_substeps=int(contact_counts[end]-contact_counts[begin]),
                          max_grip_gap_mm=1000*float(grips[sl,:,0].max()),
                          max_grip_angle_deg=float(np.rad2deg(grips[sl,:,1].max())),
                          both_grips_active=bool(held[sl].all()),
                          peak_roll_pitch_deg=np.abs(box_angles[sl,:2]).max(0).tolist(),
                          steady_wrist_mass_kg=float(sec[valid,1].mean()) if np.any(valid) else None,
                          steady_mass_valid_fraction=float(valid.mean()) if len(valid) else 0.,
                          mean_hand_vertical_force_n=force.mean(0).tolist() if len(force) else None,
                          phase_hand_nonpositive_frames=(support[sl]<=0).sum(0).tolist())
        if trial.get('steady') is None or trial.get('park') is None:
            continue
        axis=next(i for i,v in enumerate(trial['command']) if v)
        actual=trial['steady']['actual'][axis]
        error=abs(actual-trial['command'][axis])
        park=trial['park']
        trial['result']=dict(
            primary_actual=actual,primary_abs_error=error,
            nominal_speed=error<=(.035 if axis==2 else .025),
            cadence=1.9 <= (trial['steady']['cadence_hz'] or 0.) <=2.1 and trial['steady']['repeat_foot_events']==0,
            stopped=park['xy_net_drift_m']<=.025 and abs(park['yaw_drift_deg'])<=.5,
            aligned=park['final_ankle_gap_m']<=.025,
            no_foot_collision=not(trial['foot_rub_substeps'] or trial['crossing_frames']),
            held=trial['box']['both_grips_active'] and trial['box']['max_grip_gap_mm']<=5. and not trial['box']['contact_substeps'])
    m['box_mass_kg']=r['model_box_mass_kg']
    m['box_contact_steps']=r['box_contact_steps']
    m['min_measured_hand_vertical_force_n']=support.min(0).tolist()
    m['box_roll_pitch_absolute_max_deg']=np.abs(box_angles[:,:2]).max(0).tolist()
    m['all_frames_grip_error_mm']=1000*float(grips[:,:,0].max())
    m['all_frames_grip_active']=bool(held.all())
    (folder/'metrics.json').write_text(json.dumps(m,indent=2,allow_nan=False)+'\n')
    for trial in m['trials']:
        b=trial.get('result')
        print(trial['label'],trial['command'],b if b is None else
              [round(b['primary_actual'],4),[k for k,v in b.items() if isinstance(v,bool) and not v]],
              'gather',trial.get('park'),flush=True)
    return m


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('folder',type=Path)
    report(p.parse_args().folder)
