"""Evaluate each command from actual pose, support loads, and collision geometry."""
import argparse
import json
from pathlib import Path

import numpy as np

from gait import bouts, dt
from .omni import GAP_LIMIT
from .omni_preview import yaw_rot


def cycles(t, yaw):
    result = []
    for foot in range(2):
        col = 12+foot*3
        for first,last in bouts(t[:,10+foot]<.07):
            if first<1 or last>=len(t) or last-first<3:
                continue
            real = t[first-1:last+1,col:col+2]
            step = float(np.linalg.norm(real[-1]-real[0]))
            lift = float(t[first:last,col+2].max()-min(t[first-1,col+2],t[last,col+2]))
            if step < .008 or lift < .003 or t[first:last,11-foot].mean()<.40:
                continue
            edge = real[-1]-real[0]
            along = np.clip((real-real[0])@edge/step**2,0.,1.)
            chord = real[0]+along[:,None]*edge
            y = yaw_rot(yaw[(first+last)//2])[:2,1]
            inward = float(max(0.,np.max((chord-real)@y*(1-2*foot))))
            result.append(dict(foot='LR'[foot],release_s=float(t[first,0]),land_s=float(t[last,0]),
                               lift_m=lift,swing_xy_m=step,inward_chord_deviation_m=inward))
    return sorted(result,key=lambda x:x['land_s'])


def evaluate(folder, *, trajectory="rollout.trajectory.npz", block_key="blocks", save=True):
    run = json.loads((folder/'run.json').read_text())
    trace = np.load(folder/'telemetry.npy')
    clearance = np.load(folder/'clearance.npy')
    q = np.load(folder/trajectory)['qpos']
    w,x,y,z = q[:,3:7].T
    yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    events=cycles(trace,yaw)
    rows=[]
    for trial in run['trials']:
        start,zero,end=[round(trial[k]/dt) for k in ('start_s','zero_s','end_s')]
        if start>=len(q):
            rows.append(dict(**trial,reached=False,completed=False))
            continue
        ends=min(end,len(q)-1)
        mask=clearance[start:ends+1]
        axes=yaw_rot(yaw[start])[:2,:2]
        path=(q[start:ends+1,:2]-q[start,:2])@axes
        a,b=start+round(2.5/dt),min(zero,len(q)-1)
        steady=None
        vx,vy,omega=trial['command']
        slices=[c for c in events if trace[start,0] <= c['land_s'] < trace[ends,0]]
        if b>a:
            actual=np.r_[(q[b,:2]-q[a,:2])@axes, yaw[b]-yaw[a]]/((b-a)*dt)
            c=[e for e in slices if a*dt<=e['land_s']<b*dt]
            f=(len(c)-1)/(c[-1]['land_s']-c[0]['land_s']) if len(c)>1 else None
            steady=dict(actual=actual.tolist(),cadence_hz=f,landings=len(c),
                        repeat_foot_events=sum(m['foot']==n['foot'] for m,n in zip(c,c[1:])))
        settled=None
        gather=next((r for r in run[block_key] if r['kind']=='gather' and zero*dt<=r['start_s']<ends*dt),None)
        if gather:
            settled_from=round((gather['end_s']+.55)/dt)
            if ends>settled_from:
                v=trace[settled_from:ends+1,4:6]
                later=trace[ends]
                gap=abs((later[12:14]-later[15:17])@yaw_rot(yaw[ends])[:2,0])
                drift=float(np.linalg.norm(q[ends,:2]-q[settled_from,:2]))
                speeds=np.linalg.norm(v,axis=1)
                settled=dict(window_start_s=settled_from*dt,window_end_s=ends*dt,
                             xy_net_drift_m=drift,mean_speed_mps=float(speeds.mean()),
                             yaw_drift_deg=float(np.rad2deg(yaw[ends]-yaw[settled_from])),
                             final_ankle_gap_m=float(gap),final_load_bw=later[10:12].tolist(),
                             gather_foot=gather['side'])
        rows.append(dict(**trial,reached=True,completed=ends==end,steady=steady,park=settled,
                    minimum_foot_clearance_m=float(mask[:,0].min()),
                    minimum_foot_to_opposite_shin_m=float(mask[:,1].min()),
                    minimum_transverse_ankle_distance_m=float(mask[:,2].min()),
                    crossing_frames=int((mask[:,2]<=0).sum()),foot_rub_substeps=int(mask[-1,3]-mask[0,3]),
                    nonsupport_ground_substeps=int(mask[-1,4]-mask[0,4]),
                    peak_chest_angle_deg=float(np.abs(trace[start:ends+1,6:8]).max()),
                    max_swing_inward_m=max((e['inward_chord_deviation_m'] for e in slices),default=0.),
                    max_cross_track_m=float(np.max(np.abs(path[:,1 if vx else 0]))) if vx or vy else None,
                    peak_turn_center_displacement_m=float(np.max(np.linalg.norm(path,axis=1))) if omega else None))
    result=dict(completed=run['completed'],duration_s=run['duration_s'],
                physics_steps=run['physics_steps'],physics_resets=run['physics_resets'],
                gap_caps_m=GAP_LIMIT,trials=rows,foot_events=events)
    if save:
        (folder/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(rows,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('folder',type=Path)
    evaluate(p.parse_args().folder)
