"""Two-hand empty-box stand, 0.1 m/s walk, gather and settle."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from dex_mimic.deployment.runtime import Controller
from gait import bundle, contract, dt
from .carry_model import ARM_ORDER, BOX, BOX_HALF, GRIP_OFFSETS, SIDES, WALL, WRISTS, BoxPlant, arm_pose, build_box
from .omni import FootMeasurements
from .omni_preview import ZERO, yaw_rot
from .preview import Preview, ROT, STEP
from .sim import CALIBRATION

GO=6.
STOP=14.
END=21.


class CarryPreview(Preview):
    def __init__(self,calibration,arms):
        self.arms=arms
        super().__init__(calibration)

    def fill(self,through):
        self.ik.data.qpos[self.ik.qadr[ARM_ORDER]]=self.arms
        super().fill(through)


class WristSensors:
    def __init__(self,m):
        self.ids=np.array([m.site(f'carry/wrench_{s}').id for s in SIDES])
        self.bodies=m.site_bodyid[self.ids]
        self.distal_mass=m.body_subtreemass[self.bodies].copy()
        self.addr=np.array([[m.sensor_adr[m.sensor(f'carry/{k}_{s}').id] for k in ('force','torque')] for s in SIDES])
        self.gravity=m.opt.gravity.copy()
        self.g=abs(self.gravity[2])

    def read(self,d):
        local=np.asarray([[d.sensordata[adr:adr+3].copy() for adr in arm] for arm in self.addr])
        axes=d.site_xmat[self.ids].reshape(2,3,3)
        world=np.einsum('sij,skj->ski',axes,local)
        hand_weight=self.distal_mass[:,None]*(-self.gravity)
        lever=d.subtree_com[self.bodies]-d.site_xpos[self.ids]
        correction=np.stack([hand_weight,np.cross(lever,hand_weight)],axis=1)
        return local,world-correction


class LoadEstimate:
    def __init__(self,g):
        self.g=g
        self.window=deque(maxlen=4*STEP)
        self.feedforward=0.
        self.stage=None

    def update(self,force,stage,grip_ok,box_contact):
        vertical=force[:,2].copy()
        instantaneous=float(vertical.sum()/self.g)
        stable=stage in ('stand','walk') and grip_ok and not box_contact
        if stage!=self.stage or not stable:
            self.window.clear()
        self.stage=stage
        if stable:
            self.window.append(vertical)
        n=4*STEP if stage=='walk' else 2*STEP
        if len(self.window)<n:
            return instantaneous,None,False
        samples=np.array(self.window)[-n:]
        half_steps=samples.reshape(-1,STEP,2).mean(1)
        kg=half_steps.sum(1)/self.g
        valid=bool(np.std(kg)<.08 and np.min(half_steps.mean(0))>0.)
        mass=float(kg.mean()) if valid else None
        if valid:
            goal=max(0.,mass)
            self.feedforward+=.04*(goal-self.feedforward)
        return instantaneous,mass,valid



# Rounded wrist estimate from the verified empty-box stand, used only as tare.
EMPTY_TARE_KG = .650
LOAD_BINS = np.arange(11)


def mass_bin(estimate):
    """0 is the empty box; -1 defers classification until a valid window."""
    return -1 if estimate is None else int(np.argmin(np.abs(LOAD_BINS-(estimate-EMPTY_TARE_KG))))


class Holding:
    def __init__(self,plant,arms,gravity_gain,payload_gain):
        self.m,self.d=plant.model,plant.data
        self.arm=arms
        self.qa,self.va=plant.qadr[ARM_ORDER],plant.vadr[ARM_ORDER]
        self.kp=self.m.actuator_gainprm[plant.aids[ARM_ORDER],0].copy()
        self.bias=np.zeros(self.m.nv)
        self.jac=np.zeros((3,self.m.nv))
        self.data=mujoco.MjData(self.m)
        self.grip=[self.m.site(f'carry/hand_{s}').id for s in SIDES]
        self.gravity_gain=gravity_gain
        self.payload_gain=payload_gain

    def targets(self,mass):
        self.data.qpos[:]=self.d.qpos
        mujoco.mj_fwdPosition(self.m,self.data)
        mujoco.mj_rne(self.m,self.data,0,self.bias)
        torque=self.gravity_gain*self.bias[self.va].copy()
        support = np.empty_like(torque)
        for i,site in enumerate(self.grip):
            mujoco.mj_jacSite(self.m,self.data,self.jac,None,site)
            support[i]=self.payload_gain*self.jac[2,self.va[i]]*.5*abs(self.m.opt.gravity[2])*mass
        corrected=np.clip((torque+support)/self.kp,-.12,.12)
        effect=corrected-np.clip(torque/self.kp,-.12,.12)
        return self.arm+corrected,effect


class CarryMeasurements(FootMeasurements):
    def __init__(self,plant):
        super().__init__(plant)
        m=plant.model
        self.box=plant.box_id
        self.box_contact_steps=0
        self.box_ground_steps=0
        self.hand_sites=np.array([m.site(f'carry/hand_{s}').id for s in SIDES])
        self.box_sites=np.array([m.site(f'carry/box_{s}').id for s in SIDES])
        self.grips=np.array([m.equality(f'carry/grip_{s}').id for s in SIDES])
        self.eq_wrench=np.zeros(m.nv)
        self.grip_overlimit_steps=0

    def contact_box(self):
        m,d=self.plant.model,self.plant.data
        other=[]
        for c in d.contact:
            if c.dist>=0:
                continue
            a,b=int(c.geom1),int(c.geom2)
            if m.geom_bodyid[a]==self.box:
                other.append(b)
            elif m.geom_bodyid[b]==self.box:
                other.append(a)
        return bool(other),any(g in self.ground for g in other)

    def grip_errors(self):
        d=self.plant.data
        delta=d.site_xpos[self.hand_sites]-d.site_xpos[self.box_sites]
        R,H=[d.site_xmat[sites].reshape(2,3,3) for sites in (self.hand_sites,self.box_sites)]
        angle=np.arccos(np.clip((np.einsum('sij,sij->s',R,H)-1)*.5,-1.,1.))
        return np.linalg.norm(delta,axis=1),angle

    def support(self):
        p=self.plant
        forces=[]
        for eq in self.grips:
            same=(p.data.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY)&(p.data.efc_id==eq)
            selection=np.where(same,p.data.efc_force,0.)
            mujoco.mj_mulJacTVec(p.model,p.data,self.eq_wrench,selection)
            forces.append(self.eq_wrench[p.bv:p.bv+6].copy())
        return np.asarray(forces)

    def step(self,target,fixture=None):
        p,m,d=self.plant,self.plant.model,self.plant.data
        d.ctrl[p.aids]=target
        for _ in range(contract['decimation']):
            if fixture is not None:
                fixture.apply()
            mujoco.mj_step(m,d)
            _,rub,body=self.contacts()
            contact,ground=self.contact_box()
            self.rub_steps+=rub
            self.body_steps+=body
            self.box_contact_steps+=contact
            self.box_ground_steps+=ground
        mujoco.mj_forward(m,d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise FloatingPointError('Non-finite box rollout')
        if np.max(self.grip_errors()[0])>.02:
            self.grip_overlimit_steps+=1


def controller_for(preview,out,ticks):
    meta={key:contract[key] for key in ('feature_contract','model_sha256','joint_names','goal_frames','control_dt','tracked_bodies')}
    meta['num_frames']=2
    path=out/'initial_reference.npz'
    p0,q0=preview.positions[0]-preview.offset,preview.joints[0]
    np.savez_compressed(path,metadata=json.dumps(meta),root_p=np.tile(p0,(2,1)),
                        root_R=np.repeat(ROT[None],2,axis=0),q=np.tile(q0,(2,1)),
                        goal=np.tile(preview.goal(0),(2,1)))
    ctl=Controller(bundle,path)
    ctl.frames=ticks+1
    ctl.reference['goal']=np.zeros((ticks+1,contract['goal_dim']),dtype=np.float32)
    ctl.reference['root_R']=np.broadcast_to(ROT,(ticks+1,3,3))
    return ctl


def run(args, *, trials=(), build_preview=CarryPreview, added_mass=0., observe_load=None, balance_factory=None,
        fixture_factory=None):
    args.out.mkdir(parents=True)
    cfg=json.loads(CALIBRATION.read_text())
    cfg['anchors'][0]['recipe']['x_cop']+=args.cop_shift
    if args.stride_scale is not None:
        cfg['anchors'][0]['recipe']['stride_scale']=args.stride_scale
    if args.heading_trim is not None:
        cfg['anchors'][0]['recipe']['heading_trim']=args.heading_trim
    pose,orientation=arm_pose(contract['default_q'],args.reach,args.hand_height)
    preview=build_preview(cfg,pose)
    if trials:
        stops = [(trial['start_s'],tuple(trial['command'])) for trial in trials]
        stops += [(trial['zero_s'],ZERO) for trial in trials]
        end = trials[-1]['end_s']
    else:
        stops = [] if args.stand_only else [(GO,.1),(STOP,0.)]
        end = args.stand_seconds if args.stand_only else END
    switches = {round(when/dt):cmd for when,cmd in stops}
    ticks=round(end/dt)
    controller=controller_for(preview,args.out,ticks)
    if trials:
        controller.reference['root_R']=controller.reference['root_R'].copy()
    model=build_box(args.mass,orientation,args.grip_softness,added_mass=added_mass)
    plant=BoxPlant(controller,model)
    d=plant.data
    fixture=fixture_factory(plant,added_mass) if fixture_factory is not None else None
    if fixture is not None:
        fixture.apply()
        mujoco.mj_forward(model,d)
    sensors=WristSensors(model)
    measures=CarryMeasurements(plant)
    holder=Holding(plant,pose,args.gravity_gain,args.payload_gain)
    estimate=LoadEstimate(sensors.g)
    balance=balance_factory(plant,sensors,preview) if balance_factory is not None else None
    balance_trace=[]
    if args.seed:
        rng=np.random.default_rng(args.seed)
        d.qvel[plant.rv:plant.rv+2]=rng.uniform(-.018,.018,2)
        d.qvel[plant.bv:plant.bv+2]=d.qvel[plant.rv:plant.rv+2]
        mujoco.mj_forward(model,d)
    timeline, poses, clearance, rows, wrist, adjusted, supports = [],[],[],[],[],[],[]
    masses,errors,stages,applied_targets,payload_effect=[],[],[],[],[]
    grip_active,box_contact_counts,mass_bins,added_estimates=[],[],[],[]
    for frame in range(ticks+1):
        if frame in switches:
            preview.command(frame,switches[frame])
        goal=preview.goal(frame)
        stage=preview.phase(frame).kind
        state,margin=measures.snapshot(frame)
        gpos,grot=measures.grip_errors()
        actual=sensors.read(d)
        good=bool(np.all(d.eq_active[measures.grips]) and np.max(gpos)<.015 and np.max(grot)<.15)
        box_contact=measures.contact_box()[0]
        raw,value,valid=estimate.update(actual[1][:,0],stage,good,box_contact)
        if observe_load is not None:
            observe_load(preview,estimate.feedforward)
        timeline.append(d.time)
        poses.append(d.qpos.copy())
        rows.append(state)
        clearance.append(margin)
        wrist.append(actual[0])
        adjusted.append(actual[1])
        supports.append(measures.support())
        masses.append([raw,np.nan if value is None else value,float(valid),estimate.feedforward])
        mass_bins.append(mass_bin(value))
        added_estimates.append(np.nan if value is None else value-EMPTY_TARE_KG)
        errors.append(np.c_[gpos,grot])
        grip_active.append(d.eq_active[measures.grips].copy())
        box_contact_counts.append(measures.box_contact_steps)
        stages.append(stage)
        if frame==ticks or measures.body_steps or measures.box_ground_steps:
            break
        controller.reference['goal'][frame]=goal
        if trials:
            controller.reference['root_R'][frame]=yaw_rot(preview.angles[frame])
        target=controller.step(*plant.sensors())
        arm_mass=estimate.feedforward if balance is None else balance.arm_mass(estimate.feedforward)
        target[ARM_ORDER],effect=(holder.targets(arm_mass) if balance is None else
                                  balance.arm_targets(holder,arm_mass))
        if balance is not None:
            attached=np.all(d.eq_active[measures.grips]) and not box_contact
            balance_trace.append(balance.step(target,actual[1],attached and balance.grip_engaged(gpos,grot)))
        payload_effect.append(effect)
        controller.previous_action=np.clip((target-controller.q0)/controller.alpha,
                                           -contract['action_clip'],contract['action_clip'])
        applied_targets.append(target.copy())
        measures.step(target,fixture)
    assert controller.frame==len(poses)-1 and np.allclose(np.diff(timeline),dt,rtol=0,atol=1e-10)
    assert not d.xfrc_applied.any()
    if fixture is None or (len(poses)==ticks+1 and d.time>=fixture.end):
        assert not d.qfrc_applied.any()
    mjb=args.out/'box_model.mjb'
    mujoco.mj_saveModel(model,str(mjb))
    h=hashlib.sha256()
    with mjb.open('rb') as f:
        while block:=f.read(8*1024*1024):
            h.update(block)
    np.save(args.out/'telemetry.npy',rows)
    np.save(args.out/'clearance.npy',clearance)
    np.savez_compressed(args.out/'rollout.npz',qpos=poses,physics_time=timeline,
                        wrist_local=wrist,wrist_compensated_world=adjusted,grip_wrench_on_box=supports,
                        estimates=masses,grip_errors=errors,stage=stages,actuator_targets=applied_targets,
                        load_compensation_joint_delta_rad=payload_effect,
                        grip_active=grip_active,box_contact_counts=box_contact_counts,
                        estimated_mass_bin=mass_bins,estimated_added_mass_kg=added_estimates,
                        balance_trace=balance_trace)
    meta=dict(completed=len(poses)==ticks+1 and not measures.body_steps and not measures.box_ground_steps,
              frames=len(poses),physics_steps=(len(poses)-1)*contract['decimation'],physics_resets=0,
              duration_s=float(d.time),go_s=None if args.stand_only or trials else GO,zero_s=None if args.stand_only or trials else STOP,empty_box_enclosure_dimensions_m=(2*BOX_HALF).tolist(),
              empty_box_wall_thickness_m=2*WALL,model_box_mass_kg=float(model.body_mass[plant.box_id]),
              empty_shell_mass_kg=args.mass,added_mass_kg=added_mass,
              added_mass_locations_m=GRIP_OFFSETS.tolist() if added_mass>0 else [],
              added_mass_per_grip_kg=.5*added_mass,estimator_empty_tare_kg=EMPTY_TARE_KG,
              box_local_com_m=model.body_ipos[plant.box_id].tolist(),
              distal_masses_kg=sensors.distal_mass.tolist(),policy_sha256=contract['policy_sha256'],
              model_sha256=h.hexdigest(),box_free_qadr=plant.bq,body_ground_steps=measures.body_steps,
              box_contact_steps=measures.box_contact_steps,box_ground_steps=measures.box_ground_steps,
              foot_rub_steps=measures.rub_steps,grip_over_20mm_frames=measures.grip_overlimit_steps,
              options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='out'},
              planned_blocks=[dict(kind=b.kind,start_s=b.start*dt,end_s=b.end*dt,side='LR'[b.side]) for b in preview.blocks])
    if trials:
        meta['trials']=trials
        meta['carry_profile']=preview.profile
        meta['revisions']=preview.revisions
        if observe_load is not None:
            meta['measured_load_gather']=preview.load_gather_log
    if balance is not None:
        meta['balance_fields']=balance.fields
    if fixture is not None:
        meta['load_transfer']=fixture.metadata()
    (args.out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({k:v for k,v in meta.items() if k!='planned_blocks'}),flush=True)


def argument_parser(description=__doc__):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mass',type=float,default=.65)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--reach',type=float,default=.37)
    p.add_argument('--hand-height',type=float,default=-.02)
    p.add_argument('--grip-softness',type=float,default=.06)
    p.add_argument('--gravity-gain',type=float,default=1.)
    p.add_argument('--payload-gain',type=float,default=1.)
    p.add_argument('--cop-shift',type=float,default=0.)
    p.add_argument('--stride-scale',type=float,default=1.42)
    p.add_argument('--heading-trim',type=float,default=-.31)
    p.add_argument('--stand-only',action='store_true')
    p.add_argument('--stand-seconds',type=float,default=60.)
    return p


def main():
    p=argument_parser()
    args=p.parse_args()
    if not .1<=args.mass<1.:
        p.error('Empty-box mass must lie in [0.1,1.0) kg')
    if not np.isfinite(args.stand_seconds) or args.stand_seconds<6.:
        p.error('Independent stand duration must be at least 6 seconds')
    run(args)


if __name__=='__main__':
    main()
