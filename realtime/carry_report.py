"""Measure carried-box support, wrist estimate, and locomotion from recorded states."""
import argparse
import json
from pathlib import Path

import numpy as np

from gait import dt
from .omni_report import cycles
from .omni_preview import yaw_rot


def angles(q):
    w,x,y,z=q.T
    roll=np.arctan2(2*(w*x+y*z),1.-2*(x*x+y*y))
    pitch=np.arcsin(np.clip(2*(w*y-z*x),-1.,1.))
    yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1.-2*(y*y+z*z)))
    return np.stack([roll,pitch,yaw],axis=1)


def evaluate(folder):
    run=json.loads((folder/'run.json').read_text())
    trace=np.load(folder/'telemetry.npy')
    clearance=np.load(folder/'clearance.npy')
    z=np.load(folder/'rollout.npz')
    poses=z['qpos']
    box=poses[:,run['box_free_qadr']:run['box_free_qadr']+7]
    box_angle=np.rad2deg(angles(box[:,3:7]))
    robot_yaw=angles(poses[:,3:7])[:,2]
    events=cycles(trace,robot_yaw)
    qtime=trace[:,0]
    known_mass=run['model_box_mass_kg']
    carry=run['go_s'] is not None
    fields={'stand':(2.5,run['go_s']-.2 if carry else run['duration_s'])}
    if carry:
        fields['walk']=(run['go_s']+3.0,run['zero_s'])
        gather=next((b for b in run['planned_blocks'] if b['kind']=='gather'),None)
        if gather:
            fields['park']=(gather['end_s']+.55,run['duration_s'])
    sections={}
    for name,(a,b) in fields.items():
        ids=np.flatnonzero((qtime>=a-1e-10)&(qtime<=b+1e-10))
        if len(ids)<2:
            continue
        first,last=ids[[0,-1]]
        data=trace[ids]
        estimation=z['estimates'][ids]
        f=z['grip_wrench_on_box'][ids,:,2]
        valid=estimation[:,2]>0.
        values=estimation[valid,1]
        section=dict(start_s=float(qtime[first]),end_s=float(qtime[last]),
                     actual_vx_mps=float(data[:,4].mean()),actual_vy_mps=float(data[:,5].mean()),
                     speed_xy_mean_mps=float(np.linalg.norm(data[:,4:6],axis=1).mean()),
                     root_xy_net_m=float(np.linalg.norm(data[-1,1:3]-data[0,1:3])),
                     box_roll_pitch_span_deg=np.ptp(box_angle[ids,:2],axis=0).tolist(),
                     box_roll_pitch_absolute_max_deg=np.abs(box_angle[ids,:2]).max(0).tolist(),
                     grip_pos_error_max_mm=1000*float(z['grip_errors'][ids,:,0].max()),
                     grip_rot_error_max_deg=float(np.rad2deg(z['grip_errors'][ids,:,1].max())),
                     actual_hand_vertical_force_mean_n=f.mean(0).tolist(),
                     actual_hand_vertical_force_min_n=f.min(0).tolist(),
                     actual_hand_nonpositive_frames=np.count_nonzero(f<=0,axis=0).tolist(),
                     wrist_compensated_vertical_force_mean_n=z['wrist_compensated_world'][ids,:,0,2].mean(0).tolist(),
                     wrist_compensated_torque_rms_nm=np.sqrt(np.mean(z['wrist_compensated_world'][ids,:,1]**2,axis=0)).tolist(),
                     mass_valid_fraction=float(valid.mean()),
                     mass_estimate_mean_kg=float(values.mean()) if len(values) else None,
                     mass_estimate_minmax_kg=[float(values.min()),float(values.max())] if len(values) else None,
                     mass_estimate_abs_mean_error_kg=float(abs(values-known_mass).mean()) if len(values) else None,
                     feedforward_mass_mean_kg=float(estimation[:,3].mean()))
        if name=='walk':
            walking=[e for e in events if a<=e['land_s']<b]
            section['cadence_hz']=(len(walking)-1)/(walking[-1]['land_s']-walking[0]['land_s']) if len(walking)>1 else None
            section['alternating']=all(x['foot']!=y['foot'] for x,y in zip(walking,walking[1:]))
            section['max_swing_inward_cm']=100*max([e['inward_chord_deviation_m'] for e in walking],default=0.)
        sections[name]=section
    last=trace[-1]
    ap_diff=(last[12:14]-last[15:17])@yaw_rot(robot_yaw[-1])[:2,0]
    checks=dict(completed=run['completed'],
        no_nonsupport_ground=not run['body_ground_steps'],no_box_contacts=not run['box_contact_steps'],
        bilateral_vertical_support=bool(np.all(z['grip_wrench_on_box'][...,2]>0)),
        no_foot_rub=not run['foot_rub_steps'],no_crossing=bool(np.min(clearance[:,2])>0),
        grip_pose=bool(z['grip_errors'][:,:,0].max()<=.005 and np.rad2deg(z['grip_errors'][:,:,1].max())<=7.),
        standing_stable=bool('stand' in sections and sections['stand']['root_xy_net_m']<=.025))
    for name in fields:
        sec=sections.get(name,{})
        ok=(sec.get('mass_estimate_mean_kg') is not None and
            abs(sec['mass_estimate_mean_kg']-known_mass)<=.05 and min(sec['actual_hand_vertical_force_mean_n'])>0.)
        checks[name+'_mass_agreement']=bool(ok)
    if carry:
        walk=sections.get('walk',{})
        park=sections.get('park',{})
        checks['speed']=abs(walk.get('actual_vx_mps',0.)-.1)<=.015
        checks['walk_cadence']=bool(walk.get('alternating') and 1.9 <= (walk.get('cadence_hz') or 0.) <= 2.1)
        checks['park_stable']=park.get('root_xy_net_m',1.)<=.025 and abs(ap_diff)<=.025
    checks={k:bool(v) for k,v in checks.items()}
    summary=dict(passed=all(checks.values()),checks=checks,duration_s=run['duration_s'],
                 physics_resets=run['physics_resets'],box_mass_kg=known_mass,
                 grip_error_max_mm=1000*float(z['grip_errors'][:,:,0].max()),
                 box_roll_pitch_absolute_max_deg=np.abs(box_angle[:,:2]).max(0).tolist(),
                 box_vertical_span_m=float(np.ptp(box[:,2])),
                 root_peak_chest_deg=float(np.abs(trace[:,6:8]).max()),
                 min_feet_geom_clearance_m=float(np.min(clearance[:,0])),
                 min_foot_shin_geom_clearance_m=float(np.min(clearance[:,1])),
                 cross_frames=int(np.count_nonzero(clearance[:,2]<=0.)),
                 final_ankle_gap_cm=100*float(abs(ap_diff)),
                 lateral_end_m=float(trace[-1,2]-trace[0,2]),
                 box_contact_steps=run['box_contact_steps'],
                 max_applied_payload_joint_delta_deg=float(np.rad2deg(np.abs(z['load_compensation_joint_delta_rad']).max()))
                    if 'load_compensation_joint_delta_rad' in z else None,
                 all_frames_hand_vertical_min_n=z['grip_wrench_on_box'][...,2].min(0).tolist(),
                 all_frames_hand_vertical_nonpositive_count=(z['grip_wrench_on_box'][...,2]<=0).sum(0).tolist(),
                 sections=sections,
                 events=events)
    (folder/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='events'},indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    evaluate(parser.parse_args().folder)
