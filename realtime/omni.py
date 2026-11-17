"""Run the six signed single-axis start/stop commands without resetting MuJoCo."""
import argparse
import json
from pathlib import Path
import socket
import time

import mujoco
import numpy as np

from gait import contract, dt
from .omni_preview import OmniPreview, ZERO, yaw_rot
from .sim import CALIBRATION, Measurements, initialize

PASSES = [('Forward',(.4,0.,0.)), ('Backward',(-.3,0.,0.)),
          ('Left', (0.,.3,0.)), ('Right',(0.,-.3,0.)),
          ('Turn left',(0.,0.,.45)), ('Turn right',(0.,0.,-.45))]
LIMITS={(0,1):.4,(0,-1):.3,(1,1):.3,(1,-1):.3,(2,1):.45,(2,-1):.45}
TRIAL_SECONDS = 12
MOVE_SECONDS = 7
GAP_LIMIT = .10


def read_command(receiver):
    command = None
    while True:
        try:
            data = receiver.recv(128)
        except BlockingIOError:
            return command
        try:
            values = tuple(float(x) for x in data.split())
        except ValueError:
            continue
        if len(values)!=3 or not np.isfinite(values).all():
            continue
        axes = [i for i,x in enumerate(values) if x]
        if not axes:
            command = ZERO
        if len(axes)==1:
            a=axes[0]
            if abs(values[a]) <= LIMITS[a,int(np.sign(values[a]))]:
                command = values


class FootMeasurements(Measurements):
    def __init__(self, plant):
        super().__init__(plant)
        m=plant.model
        enabled=(m.geom_contype|m.geom_conaffinity)!=0
        self.foot_geoms=[{g for g in shapes if enabled[g]} for shapes in self.foot_geoms]
        shins=[set(np.flatnonzero(np.isin(m.geom_bodyid,list(ids)) & enabled)) for ids in self.shins]
        self.foot_pairs=self.pair_array(self.foot_geoms[0],self.foot_geoms[1])
        self.shin_pairs=np.concatenate([self.pair_array(self.foot_geoms[i],shins[1-i]) for i in range(2)])
        self.segment=np.zeros(6)

    @staticmethod
    def pair_array(x,y):
        return np.asarray([(a,b) for a in x for b in y],dtype=int).reshape(-1,2)

    def distance(self,pairs):
        m,d=self.plant.model,self.plant.data
        a,b=pairs.T
        lower=np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b],axis=1)-m.geom_rbound[a]-m.geom_rbound[b]
        gap=GAP_LIMIT
        for idx in np.argsort(lower):
            if lower[idx]>=gap:
                break
            gap=min(gap,mujoco.mj_geomDistance(m,d,int(a[idx]),int(b[idx]),gap,self.segment))
        return gap

    def snapshot(self,frame):
        p=self.plant
        d=p.data
        pelvis=d.xmat[p.model.body('robot/pelvis').id].reshape(3,3)
        f=d.xpos[self.feet]
        yaw=np.arctan2(pelvis[1,0],pelvis[0,0])
        transverse=(f[0]-f[1]) @ yaw_rot(yaw)[:,1]
        return self.record(frame),[self.distance(self.foot_pairs),self.distance(self.shin_pairs),transverse,
                                   self.rub_steps,self.body_steps]


def run(out, options, pace=False, schedule=None, duration=None):
    out.mkdir(parents=True)
    slots = schedule if schedule is not None else PASSES
    switches={}
    trials=[]
    for i,(label,cmd) in enumerate(slots):
        start=3+i*TRIAL_SECONDS
        stop=start+MOVE_SECONDS
        switches[round(start/dt)]=cmd
        switches[round(stop/dt)]=ZERO
        trials.append(dict(label=label,command=cmd,start_s=start,zero_s=stop,end_s=start+TRIAL_SECONDS))
    ticks=round((3+len(slots)*TRIAL_SECONDS if duration is None else duration)/dt)
    planner=OmniPreview(json.loads(CALIBRATION.read_text()),options)
    ctl,plant=initialize(planner,out,ticks)
    ctl.reference['root_R']=ctl.reference['root_R'].copy()
    measures=FootMeasurements(plant)
    rows, distances, poses, commands, planned, stages, times = [],[],[],[],[],[],[]
    started=time.monotonic()
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as receiver, socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sender:
        receiver.bind(('127.0.0.1',0))
        receiver.setblocking(False)
        print(f'UDP {receiver.getsockname()}',flush=True)
        for frame in range(ticks+1):
            if pace:
                time.sleep(max(0.,started+frame*dt-time.monotonic()))
            if frame in switches:
                sender.sendto(' '.join(map(str,switches[frame])).encode(),receiver.getsockname())
            cmd=read_command(receiver)
            if cmd is not None:
                planner.command(frame,cmd)
            row,distance=measures.snapshot(frame)
            rows.append(row)
            distances.append(distance)
            poses.append(plant.data.qpos.copy())
            times.append(plant.data.time)
            goal=planner.goal(frame)
            b=planner.phase(frame)
            commands.append(planner.intent)
            planned.append(b.cmd)
            stages.append(b.kind)
            if frame==ticks or measures.body_steps:
                break
            ctl.reference['goal'][frame]=goal
            ctl.reference['root_R'][frame]=yaw_rot(planner.angles[frame])
            measures.step(ctl.step(*plant.sensors()))
    assert ctl.frame==len(poses)-1
    assert np.allclose(np.diff(times),dt,atol=1e-10,rtol=0)
    assert not np.any(plant.data.xfrc_applied) and not np.any(plant.data.qfrc_applied)
    np.save(out/'telemetry.npy',rows)
    np.save(out/'clearance.npy',distances)
    np.savez_compressed(out/'rollout.trajectory.npz',qpos=poses,control_dt=dt,physics_time=times,
                        policy_sha256=contract['policy_sha256'],model_sha256=contract['model_sha256'])
    np.savez_compressed(out/'commands.npz',requested=commands,planned=planned,stage=stages,
                        reference_root=np.array(planner.positions[:len(poses)])-planner.offset,
                        ref_yaw=planner.angles[:len(poses)])
    meta=dict(completed=len(poses)==ticks+1 and not measures.body_steps,physics_resets=0,
              duration_s=times[-1],frames=len(poses),physics_steps=(len(poses)-1)*contract['decimation'],
              rub_substeps=measures.rub_steps,nonsupport_ground_substeps=measures.body_steps,
              max_ik_error_m=float(planner.ik.max_error),trials=trials,options=planner.options,
              revisions=planner.revisions,blocks=[
                  dict(kind=b.kind,side='LR'[b.side],start_s=b.start*dt,end_s=b.end*dt,cmd=b.cmd)
                  for b in planner.blocks if b.start<len(poses)])
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({k:v for k,v in meta.items() if k not in ('blocks','options')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--options',type=Path,default=Path(__file__).with_name('omni_calibration.json'))
    p.add_argument('--pace',action='store_true')
    a=p.parse_args()
    run(a.out,json.loads(a.options.read_text()),a.pace)
