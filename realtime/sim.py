"""Live speed input and a single native MuJoCo closed-loop rollout."""
import argparse
import json
import math
from pathlib import Path
import socket
import time

import mujoco
import numpy as np

from dex_mimic.deployment.native_mujoco import Plant
from dex_mimic.deployment.runtime import Controller
from gait import bundle, contract, dt

from .preview import LOOKAHEAD, ROT, Preview

DEMO = [(3.,.1),(9.,.2),(15.,.3),(21.,.4),(27.,.2),(33.,0.),(39.,.1)]
CALIBRATION = bundle.parents[1]/'release'/'calibration.json'


class Commands:
    def __init__(self, receiver, max_speed=.4):
        self.receiver = receiver
        self.max_speed = max_speed
        self.rejected = 0

    def latest(self):
        speed = None
        while True:
            try:
                packet = self.receiver.recv(128)
            except BlockingIOError:
                return speed
            try:
                value = float(packet)
            except ValueError:
                self.rejected += 1
                continue
            if math.isfinite(value) and (value == 0. or .1 <= value <= self.max_speed):
                speed = value
            else:
                self.rejected += 1


class Measurements:
    def __init__(self, plant):
        self.plant = plant
        m = plant.model
        self.ground = set(np.flatnonzero(m.geom_type==mujoco.mjtGeom.mjGEOM_PLANE))
        self.feet = [m.body(f'robot/ankle_roll_{s}_link').id for s in 'lr']
        self.foot_geoms = [set(np.flatnonzero(m.geom_bodyid==foot)) for foot in self.feet]
        self.shins = [{m.body(f'robot/{j}_{s}_link').id for j in ('knee_pitch','ankle_pitch')} for s in 'lr']
        self.chest = m.body('robot/waist_pitch_link').id
        self.weight = mujoco.mj_getTotalmass(m)*abs(m.opt.gravity[2])
        self.force = np.zeros(6)
        self.rub_steps = 0
        self.body_steps = 0

    def contacts(self, with_load=False):
        m,d = self.plant.model,self.plant.data
        rub = body = False
        loads = np.zeros(2)
        for index,c in enumerate(d.contact):
            a,b = int(c.geom1),int(c.geom2)
            floor_other = b if a in self.ground else a if b in self.ground else -1
            if floor_other >= 0:
                feet = [i for i,geoms in enumerate(self.foot_geoms) if floor_other in geoms]
                body |= not feet and c.dist < 0
                if with_load and feet:
                    mujoco.mj_contactForce(m,d,index,self.force)
                    loads[feet[0]] += self.force[0]*abs(c.frame[2])/self.weight
            elif c.dist < 0:
                for i,geoms in enumerate(self.foot_geoms):
                    other = self.foot_geoms[1-i]
                    rub |= ((a in geoms and (b in other or m.geom_bodyid[b] in self.shins[1-i]))
                            or (b in geoms and (a in other or m.geom_bodyid[a] in self.shins[1-i])))
        return loads, rub, body

    def record(self, frame):
        p,d = self.plant,self.plant.data
        loads,rub,body = self.contacts(with_load=True)
        R = d.xmat[self.chest].reshape(3,3)
        pitch,roll = np.rad2deg([np.arcsin(np.clip(-R[2,0],-1,1)),np.arctan2(R[2,1],R[2,2])])
        return [frame*dt,*d.qpos[p.rq:p.rq+3],*d.qvel[p.rv:p.rv+2],pitch,roll,0.,0.,
                *loads,*d.xpos[self.feet].ravel(),int(rub),int(body)]

    def step(self, target):
        p,m,d = self.plant,self.plant.model,self.plant.data
        d.ctrl[p.aids] = target
        for _ in range(contract['decimation']):
            mujoco.mj_step(m,d)
            _,rub,body = self.contacts()
            self.rub_steps += rub
            self.body_steps += body
        mujoco.mj_forward(m,d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise FloatingPointError('Non-finite MuJoCo state')


def initialize(preview, out, ticks):
    metadata = {key:contract[key] for key in ('feature_contract','model_sha256','joint_names',
                                             'goal_frames','control_dt','tracked_bodies')}
    metadata['num_frames'] = 2
    path = out/'initial_reference.npz'
    p0 = preview.positions[0]-preview.offset
    q0 = preview.joints[0]
    np.savez_compressed(path,metadata=json.dumps(metadata),root_p=np.tile(p0,(2,1)),
                        root_R=np.repeat(ROT[None],2,axis=0),q=np.tile(q0,(2,1)),
                        root_v=np.zeros((2,3)),root_w=np.zeros((2,3)),qd=np.zeros((2,len(q0))),
                        goal=np.tile(preview.goal(0),(2,1)))
    controller = Controller(bundle,path)
    plant = Plant(controller)
    controller.frames = ticks+1
    controller.reference['goal'] = np.zeros((ticks+1,contract['goal_dim']),dtype=np.float32)
    controller.reference['root_R'] = np.broadcast_to(ROT,(ticks+1,3,3))
    return controller,plant


def simulate(args, receiver, sender, *, schedule=None, calibration=CALIBRATION, max_speed=.4):
    ticks = round(args.duration/dt)
    out = args.out
    out.mkdir(parents=True)
    planner = Preview(json.loads(calibration.read_text()))
    ctl,plant = initialize(planner,out,ticks)
    metrics = Measurements(plant)
    commands = Commands(receiver, max_speed)
    if schedule is None:
        schedule = DEMO if args.demo else ()
    timeline = {round(t/dt):v for t,v in schedule}
    poses,telemetry,states,loop_times,targets,physics_times = [],[],[],[],[],[]
    elapsed = time.monotonic()
    print(f'Command UDP {receiver.getsockname()[0]}:{receiver.getsockname()[1]}',flush=True)
    for frame in range(ticks+1):
        if args.pace:
            delay = elapsed + frame*dt-time.monotonic()
            if delay>0:
                time.sleep(delay)
        iteration = time.monotonic()
        if frame in timeline:
            val = timeline[frame]
            sender.sendto(str(val).encode('ascii'),receiver.getsockname())
        speed = commands.latest()
        if speed is not None:
            planner.command(frame,speed)
        poses.append(plant.data.qpos.copy())
        telemetry.append(metrics.record(frame))
        physics_times.append(plant.data.time)
        goal = planner.goal(frame)
        block = planner.phase(frame)
        states.append((planner.speed,block.speed,block.kind,block.side))
        if frame==ticks or metrics.body_steps:
            break
        ctl.reference['goal'][frame] = goal
        target = ctl.step(*plant.sensors())
        metrics.step(target)
        targets.append(target)
        loop_times.append(time.monotonic()-iteration)
    assert ctl.frame == len(poses)-1
    assert np.allclose(np.diff(physics_times),dt,rtol=0,atol=1e-10)
    assert not np.any(plant.data.xfrc_applied) and not np.any(plant.data.qfrc_applied)
    np.save(out/'telemetry.npy',telemetry)
    np.savez_compressed(out/'rollout.trajectory.npz',qpos=poses,control_dt=dt,
                        physics_time=physics_times,joint_targets=targets,
                        policy_sha256=contract['policy_sha256'],model_sha256=contract['model_sha256'])
    np.savez_compressed(out/'commands.npz',requested=[r[0] for r in states],
                        planned=[r[1] for r in states],stage=[r[2] for r in states],
                        swing_side=[r[3] for r in states],
                        reference_root=np.array(planner.positions[:len(poses)])-planner.offset)
    report = dict(completed=len(poses)==ticks+1 and not metrics.body_steps,
                  planned_duration_s=ticks*dt,planned_commands=[list(e) for e in schedule],
                  physics_resets_after_initialization=0,frames=len(poses),physics_steps=len(targets)*contract['decimation'],
                  duration_s=physics_times[-1],revisions=planner.revisions,
                  blocks=[dict(kind=b.kind,side='LR'[b.side],start_s=b.start*dt,end_s=b.end*dt,
                               speed=b.speed,target_xy=b.path_xy.tolist()) for b in planner.blocks
                          if b.start<=ticks],rub_substeps=metrics.rub_steps,
                  nonsupport_ground_substeps=metrics.body_steps,rejected_commands=commands.rejected,
                  planning_and_control_p50_ms=float(np.median(loop_times)*1000),
                  planning_and_control_p95_ms=float(np.quantile(loop_times,.95)*1000),
                  planning_and_control_max_ms=float(np.max(loop_times)*1000),
                  over_20ms_ticks=int(np.count_nonzero(np.array(loop_times)>.02)),
                  max_ik_error_m=float(planner.ik.max_error))
    (out/'run.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'blocks'}),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--demo',action='store_true',help='Send the verification commands at the current simulation time')
    parser.add_argument('--port',type=int,default=0,help='Listen on localhost UDP; 0 chooses a free port')
    parser.add_argument('--duration',type=float,default=46.)
    parser.add_argument('--pace',action='store_true',help='Pace playback to the 50 Hz control clock for live input')
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration < 1.:
        parser.error('Duration must be a finite number of seconds ≥ 1')
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as receiver, socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sender:
        receiver.bind(('127.0.0.1',args.port))
        receiver.setblocking(False)
        simulate(args,receiver,sender)
