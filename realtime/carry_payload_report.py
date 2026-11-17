"""Check blind mass bins and compare identical-physics payload compensation on/off."""
import argparse
import json
from pathlib import Path

import numpy as np

from gait import dt
from .carry_omni_report import report
from .carry_report import angles, evaluate


def measure(folder):
    run=json.loads((folder/'run.json').read_text())
    if run.get('trials'):
        locomotion=report(folder)
        spans=[('initial_stand',2.5,run['trials'][0]['start_s']-.2)]
        for i,t in enumerate(locomotion['trials'],1):
            if t.get('steady'):
                spans.append((f'{i}_walk',t['start_s']+2.5,t['zero_s']))
            if t.get('park'):
                w=t['park']
                spans.append((f'{i}_park',w['window_start_s'],w['window_end_s']))
    else:
        locomotion=evaluate(folder)
        spans=[('stand',2.5,run['duration_s'])]
    z=np.load(folder/'rollout.npz')
    mass=z['estimates']
    bins=z['estimated_mass_bin']
    time=np.rint(z['physics_time']/dt)*dt
    phase=z['stage']
    actual=run['model_box_mass_kg']
    added=z['estimated_added_mass_kg']
    truth_added=run['added_mass_kg']
    expected_bin=round(truth_added)
    poses=z['qpos']
    box_q=run['box_free_qadr']
    yaw_pose=np.rad2deg(angles(poses[:,box_q+3:box_q+7]))
    world_force=z['wrist_compensated_world'][...,0,:]
    eq_force=z['grip_wrench_on_box'][...,2]
    windows={}
    for label,a,b in spans:
        active=(time>=a-1e-8)&(time<=b+1e-8)
        if not np.any(active):
            continue
        valid=active & (mass[:,2]>0)
        if valid.any():
            mean=float(mass[valid,1].mean())
            max_error=float(np.abs(mass[valid,1]-actual).max())
            mass_error=float(np.abs(mass[valid,1]-actual).mean())
            correct=float((bins[valid]==expected_bin).mean())
        else:
            mean,max_error,mass_error,correct=None,None,None,None
        grip=z['grip_errors'][active]
        tilt=yaw_pose[active,:2]
        windows[label]=dict(mean_estimated_kg=mean,
                            mean_estimated_added_kg=float(added[valid].mean()) if valid.any() else None,
                            added_mean_abs_error_kg=float(np.abs(added[valid]-truth_added).mean()) if valid.any() else None,
                            max_sample_error_kg=max_error,mean_abs_error_kg=mass_error,
                            correct_bin_fraction=correct,
                            valid_fraction=float(valid.sum()/active.sum()),
                            mean_vertical_wrist_load_n=world_force[active,:,2].mean(0).tolist(),
                            min_physics_vertical_support_n=eq_force[active].min(0).tolist(),
                            mean_physics_vertical_support_n=eq_force[active].mean(0).tolist(),
                            mean_box_z_above_pelvis_m=float((poses[active,box_q+2]-poses[active,2]).mean()),
                            box_roll_pitch_span_deg=np.ptp(tilt,axis=0).tolist(),
                            box_roll_pitch_rms_deg=np.sqrt(np.mean(tilt**2,axis=0)).tolist(),
                            rms_grip_offset_mm=1000*float(np.sqrt(np.mean(grip[...,0]**2))),
                            peak_grip_offset_mm=1000*float(grip[...,0].max()))
    transient=np.isin(phase,('shift','brake','gather'))
    result=dict(physics_true_mass_kg=actual,physics_true_added_kg=truth_added,
                expected_added_bin=expected_bin,windows=windows,
                nonpositive_support_counts=(eq_force<=0).sum(0).tolist(),
                both_grips_active=bool(z['grip_active'].all()),
                transient_frames=int(transient.sum()),
                valid_estimate_in_transient_frames=int(np.count_nonzero(transient&(bins>=0))),
                peak_applied_payload_delta_deg=float(np.rad2deg(np.abs(z['load_compensation_joint_delta_rad']).max())),
                completed=run['completed'],physics_resets=run['physics_resets'],physics_steps=run['physics_steps'],
                box_contact_steps=run['box_contact_steps'],foot_rub_steps=run['foot_rub_steps'],
                peak_box_roll_pitch_deg=np.abs(yaw_pose[:,:2]).max(0).tolist(),
                max_grip_gap_mm=1000*float(z['grip_errors'][...,0].max()))
    (folder/'mass_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result,locomotion


def compare(root):
    runs={}
    for label in ('stand_on','stand_off','forward_on','forward_off'):
        folder=root/label
        result,locomotion=measure(folder)
        runs[label]=result
        print(label,result['physics_true_mass_kg'],result['completed'],
              'applied delta deg',result['peak_applied_payload_delta_deg'],
              'max grip mm',result['max_grip_gap_mm'],
              'box contacts',result['box_contact_steps'],
              'box peak tilt',result['peak_box_roll_pitch_deg'])
        if locomotion.get('trials'):
            for t in locomotion['trials']:
                a=t.get('result',{})
                print(t['command'],a, 'park',t.get('park'))
    contrasts={}
    for shape in ('stand','forward'):
        a=root/(shape+'_on')
        b=root/(shape+'_off')
        ha,hb=[json.loads((p/'run.json').read_text()) for p in (a,b)]
        ra,rb=[np.load(p/'rollout.npz') for p in (a,b)]
        same=ha['model_sha256']==hb['model_sha256'] and np.array_equal(ra['qpos'][0],rb['qpos'][0])
        assert same
        window_a=runs[shape+'_on']['windows']
        window_b=runs[shape+'_off']['windows']
        contrasts[shape]=dict(
            same_mjb_and_initial_qpos=same,
            windows={key:{name:dict(compensated=va[name],uncompensated=window_b[key][name])
                         for name in ('rms_grip_offset_mm','peak_grip_offset_mm',
                                      'mean_box_z_above_pelvis_m','box_roll_pitch_span_deg')}
                      for key,va in window_a.items() if key in window_b})
    result=dict(runs=runs,ablation=contrasts)
    (root/'mass_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    args=p.parse_args()
    if (args.root/'run.json').is_file():
        measure(args.root)
    else:
        compare(args.root)
