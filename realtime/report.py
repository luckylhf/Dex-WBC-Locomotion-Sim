"""Measure actual contacts, steady speeds, and the independently recorded stop."""
import argparse
import json
from pathlib import Path

import numpy as np

from gait import bouts, dt


def landings(trace):
    events = []
    for side in range(2):
        col = 12+3*side
        for a,b in bouts(trace[:,10+side]<.07):
            if a<8 or b>=len(trace) or b-a<3:
                continue
            z = trace[a:b,col+2].max()-min(trace[a-1,col+2],trace[b,col+2])
            dx = trace[b,col]-trace[a-1,col]
            if dx>=.011 and z>=.003 and trace[a:b,11-side].mean()>=.45:
                events.append(dict(side='LR'[side],release_s=float(trace[a,0]),land_s=float(trace[b,0]),
                                   swing_displacement_m=float(dx),lift_m=float(z)))
    return sorted(events,key=lambda event:event['land_s'])


def analyze(folder):
    folder = Path(folder)
    run = json.loads((folder/'run.json').read_text())
    trace = np.load(folder/'telemetry.npy')
    tape = np.load(folder/'commands.npz')
    poses = np.load(folder/'rollout.trajectory.npz')['qpos']
    stages = tape['stage']
    t = trace[:,0]
    events = landings(trace)
    speed_segments = []
    for i,r in enumerate(run['revisions']):
        if r['speed']==0:
            continue
        end = run['revisions'][i+1]['frame']*dt if i+1<len(run['revisions']) else t[-1]+dt
        start = r['frame']*dt
        steady_start = start+(2.5 if i==0 or run['revisions'][i-1]['speed']==0 else 1.5)
        mask = (t>=steady_start)&(t<end)&(stages=='walk')
        segment = trace[mask]
        if len(segment)==0:
            continue
        actual = float(segment[:,4].mean())
        lands = [e['land_s'] for e in events if steady_start <= e['land_s'] < end]
        cadence = (len(lands)-1)/(lands[-1]-lands[0]) if len(lands)>1 else None
        speed_segments.append(dict(command_mps=r['speed'],command_at_s=start,
                                   window_start_s=steady_start,window_end_s=end,
                                   actual_mean_mps=actual,actual_cadence_hz=cadence,
                                   effective_step_m=actual/cadence if cadence else None))
    walking_intervals = []
    alternating = True
    for a,b in zip(events,events[1:]):
        i,j = [round(e['land_s']/dt) for e in (a,b)]
        if np.all(np.isin(stages[i:j+1],('walk','brake'))):
            walking_intervals.append(b['land_s']-a['land_s'])
            alternating &= a['side']!=b['side']
    stops = []
    for b in run['blocks']:
        if b['kind']!='gather' or b['end_s']+.6>=t[-1]:
            continue
        start = round(b['start_s']/dt)
        forward = trace[start,1:3]
        stop = round((b['end_s']+.6)/dt)
        later = next((r['frame'] for r in run['revisions'] if r['frame']>stop),len(trace))
        before_restart = max(later-1,stop)
        w,x,y,z = poses[before_restart,3:7]
        yaw = np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        heading = np.array([np.cos(yaw),np.sin(yaw)])
        side = 'LR'.index(b['side'])
        j,opp = 12+3*side,12+3*(1-side)
        from_to = trace[[start,before_restart],j:j+2]
        stops.append(dict(gather_side=b['side'],gather_start_s=b['start_s'],
                          aligned_measured_at_s=float(t[before_restart]),
                          initial_trailing_gap_m=float(heading@(trace[start,opp:opp+2]-from_to[0])),
                          gather_forward_m=float(heading@(from_to[1]-from_to[0])),
                          final_ankle_gap_heading_m=abs(float(heading@(trace[before_restart,opp:opp+2]-from_to[1]))),
                          foot_load_at_stop_bw=trace[before_restart,10:12].tolist(),
                          settled_root_drift_m=float(np.linalg.norm(trace[before_restart,1:3]-trace[stop,1:3])),
                          initial_braking_to_park_displacement_m=float(np.linalg.norm(trace[stop,1:3]-forward))))
    result=dict(completed=run['completed'], duration_s=run['duration_s'], frames=run['frames'],
                physics_steps=run['physics_steps'], physics_resets=run['physics_resets_after_initialization'],
                crossing_frames=int(np.count_nonzero(trace[:,13]<=trace[:,16])),
                rub_substeps=run['rub_substeps'],nonsupport_ground_substeps=run['nonsupport_ground_substeps'],
                speed_segments=speed_segments,walking_landings=len(events),alternating=alternating,
                walking_interval_min_s=float(min(walking_intervals)) if walking_intervals else None,
                walking_interval_max_s=float(max(walking_intervals)) if walking_intervals else None,
                walking_cadence_hz=1/float(np.mean(walking_intervals)) if walking_intervals else None,
                peak_chest_angle_deg=float(np.abs(trace[:,6:8]).max()),stops=stops,events=events)
    (folder/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='events'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    analyze(parser.parse_args().folder)
