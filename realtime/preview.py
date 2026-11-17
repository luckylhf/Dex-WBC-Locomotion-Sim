"""Receding contact plan and continuous 50 Hz reference for record12."""
from dataclasses import dataclass
from functools import lru_cache

import mujoco
import numpy as np

from command import recipe_for
from gait import bundle, contract, dt, smooth

LOOKAHEAD = contract['goal_frames']
STEP = 25
ROT = np.eye(3, dtype=np.float32)
ROT6 = ROT[:, :2].reshape(-1)


@dataclass
class Block:
    start: int
    end: int
    kind: str
    side: int
    feet: np.ndarray
    cop: np.ndarray
    height: float
    path_xy: np.ndarray
    speed: float
    swing_end: float = 0.0
    lift: float = 0.0
    transfer: np.ndarray | None = None


class LegIK:
    def __init__(self, cfg):
        self.model = m = mujoco.MjModel.from_binary_path(str(bundle/'model.mjb'))
        self.data = d = mujoco.MjData(m)
        names = contract['joint_names']
        ids = [m.joint('robot/'+name).id for name in names]
        self.qadr = m.jnt_qposadr[ids]
        self.vadr = m.jnt_dofadr[ids]
        self.limits = m.jnt_range[ids] + [.005, -.005]
        self.legs = [np.array([i for i,n in enumerate(names) if f'_{s}_' in n
                              and n.startswith(('hip','knee','ankle'))]) for s in 'lr']
        self.feet = [m.body(f'robot/ankle_roll_{s}_link').id for s in 'lr']
        self.pelvis = m.body('robot/pelvis').id
        d.qpos[:7] = [0,0,.97,1,0,0,0]
        d.qpos[self.qadr] = contract['default_q']
        mujoco.mj_forward(m,d)
        self.neutral_feet = d.xpos[self.feet].copy()
        self.rot = d.xmat[self.feet].reshape(2,3,3).copy()
        for i, foot in enumerate(self.feet):
            geom = np.flatnonzero((m.geom_bodyid == foot) & (m.geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER))
            a = d.geom_xmat[geom].reshape(-1,3,3)[:,2,2]
            r,h = m.geom_size[geom,:2].T
            floor = np.min(d.geom_xpos[geom,2]-r*np.sqrt(np.maximum(0,1-a*a))-h*np.abs(a))
            self.neutral_feet[i,2] -= floor
        self.neutral_feet[:,1] -= np.sign(self.neutral_feet[:,1])*cfg.narrow
        self.neutral_height = .971 + (self.neutral_feet[:,2]-d.xpos[self.feet,2]).mean()
        self.jp = np.zeros((3,m.nv))
        self.jr = self.jp.copy()
        self.damping = np.eye(6)*.00012
        self.max_error = 0.

    def fit(self, desired):
        m,d = self.model,self.data
        for foot,rot,leg,target in zip(self.feet,self.rot,self.legs,desired):
            qa,va = self.qadr[leg],self.vadr[leg]
            for _ in range(12):
                mujoco.mj_kinematics(m,d)
                mujoco.mj_comPos(m,d)
                have = d.xmat[foot].reshape(3,3)
                e = np.r_[target-d.xpos[foot],.5*np.cross(have.T,rot.T).sum(0)]
                if np.linalg.norm(e) < 2e-5:
                    break
                mujoco.mj_jacBody(m,d,self.jp,self.jr,foot)
                jac = np.r_[self.jp[:,va],self.jr[:,va]]
                delta = jac.T @ np.linalg.solve(jac@jac.T+self.damping,e)
                d.qpos[qa] = np.clip(d.qpos[qa]+np.clip(delta,-.13,.13),*self.limits[leg].T)

    def solve(self,feet,com,height):
        m,d = self.model,self.data
        d.qpos[2] = height
        for _ in range(3):
            self.fit(feet)
            mujoco.mj_kinematics(m,d)
            mujoco.mj_comPos(m,d)
            d.qpos[:2] += .8*(com-d.subtree_com[self.pelvis,:2])
        self.fit(feet)
        mujoco.mj_kinematics(m,d)
        self.max_error = max(self.max_error, np.linalg.norm(d.xpos[self.feet]-feet,axis=1).max())
        return d.qpos[:3].copy(), d.qpos[self.qadr].copy()


class Preview:
    """Commit contact blocks near the policy lookahead, and extrapolate the last command."""

    def __init__(self, calibration):
        self.cfg = lru_cache(maxsize=128)(lambda speed: recipe_for(speed,calibration))
        nominal = self.cfg(.1)
        self.ik = LegIK(nominal)
        feet = self.ik.neutral_feet.copy()
        height = self.ik.neutral_height-nominal.crouch
        cop = self.center(feet,nominal)
        self.origin = Block(0,0,'stand',0,feet,cop,height,np.zeros(2),0.)
        self.blocks = [Block(0,250,'stand',0,feet,cop,height,np.zeros(2),0.)]
        self.speed = 0.
        self.positions = []
        self.joints = []
        self.revisions = []
        self.correction = None
        self.rebuild()
        self.fill(LOOKAHEAD+1)
        self.offset = self.positions[0].copy()
        self.offset[2] = 0.

    @staticmethod
    def center(feet,cfg):
        return feet.mean(0)[:2]+[cfg.x_cop,0.]

    @staticmethod
    def stance(feet,side,cfg):
        return feet[side,:2]+[cfg.x_cop,-np.sign(feet[side,1])*cfg.support_y]

    def add(self, kind, duration, side, feet, cop, height, xy, speed, swing=0., lift=0., **extras):
        start = self.blocks[-1].end if self.blocks else 0
        self.blocks.append(Block(start,start+duration,kind,side,feet,cop,height,xy,speed,
                                 swing,lift,**extras))

    def shift(self, previous, speed):
        cfg = self.cfg(speed)
        self.add('shift',round(cfg.first_hold/dt),0,previous.feet,
                 self.stance(previous.feet,1,cfg), self.ik.neutral_height-cfg.crouch,
                 previous.path_xy,speed)

    def step(self, previous, speed, scale, kind='walk'):
        cfg = self.cfg(speed)
        side = previous.side if previous.kind=='shift' else 1-previous.side
        dx = speed*STEP*dt*cfg.stride_scale*scale
        xy = previous.path_xy + [dx,cfg.heading_trim*dx]
        feet = previous.feet.copy()
        feet[side,:2] = self.ik.neutral_feet[side,:2]+xy
        stance = self.stance(previous.feet,1-side,cfg)
        self.add(kind,STEP,side,feet,self.stance(feet,side,cfg),
                 self.ik.neutral_height-cfg.crouch,xy,speed,STEP*dt-cfg.double,cfg.lift,
                 transfer=stance)

    def gather(self, previous):
        cfg = self.cfg(previous.speed)
        side = 1-previous.side
        feet = previous.feet.copy()
        initial = self.ik.neutral_feet
        trim = cfg.gather_trim_right if side else cfg.gather_trim
        feet[side,:2] = feet[1-side,:2]+initial[side,:2]-initial[1-side,:2]+[trim,0.]
        swing = STEP*dt-cfg.double
        self.add('gather',round((swing+cfg.gather)/dt),side,feet,self.center(feet,cfg),
                 previous.height,previous.path_xy,0.,swing,cfg.lift,
                 transfer=self.stance(previous.feet,1-side,cfg))

    def extend(self, through):
        while self.blocks[-1].end < through:
            b = self.blocks[-1]
            if b.kind in ('stand','gather'):
                if self.speed and b.kind=='stand':
                    self.shift(b,self.speed)
                else:
                    hold = 22 if self.speed else 200
                    self.add('stand',hold,b.side,b.feet,b.cop,b.height,b.path_xy,0.)
            else:
                n = 0
                for prior in reversed(self.blocks):
                    if prior.kind != 'walk':
                        break
                    n += 1
                ramp = (.8,.9)[n] if n < 2 else 1.
                self.step(b,self.speed,ramp)

    def command(self, frame, speed):
        if speed == self.speed:
            return
        earliest = frame+LOOKAHEAD+2
        splice = next(i for i,b in enumerate(self.blocks) if b.end>earliest)
        block = self.blocks[splice]
        stopping = self.blocks[splice-1].kind=='brake' if splice else False
        if block.kind in ('stand',) and not stopping:
            retained = Block(block.start,earliest,block.kind,block.side,block.feet,block.cop,
                             block.height,block.path_xy,block.speed)
            self.blocks = self.blocks[:splice]+[retained]
        else:
            self.blocks = self.blocks[:splice+1]
            stopping |= block.kind in ('brake','gather')
        self.speed = speed
        tail = self.blocks[-1]
        if stopping and tail.kind=='brake':
            self.gather(tail)
        elif not speed and tail.kind in ('shift','walk'):
            old_speed = tail.speed
            self.step(tail,old_speed,.80,'brake')
            self.step(self.blocks[-1],old_speed,.70,'brake')
            self.gather(self.blocks[-1])
        old_com = self.com
        self.extend(frame+260)
        self.rebuild()
        first = len(self.positions)
        # Old and new trajectories share committed footsteps. Fade the change in the
        # balance preview over 0.36 s to avoid a jump in root position or velocity.
        fade = min(19,len(old_com)-first)
        if fade>1:
            idx = slice(first,first+fade)
            mix = smooth(np.linspace(0.,1.,fade))[:,None]
            self.com[idx] = (1-mix)*old_com[idx]+mix*self.com[idx]
            self.correction = (first,self.com[idx].copy())
        self.revisions.append(dict(frame=frame,speed=speed,splice_frame=tail.end))

    def rebuild(self):
        length = self.blocks[-1].end
        self.feet = np.empty((length,2,3))
        self.cop = np.empty((length,2))
        self.height = np.empty(length)
        previous = self.origin
        for b in self.blocks:
            t = np.arange(b.end-b.start)*dt
            if not len(t):
                continue
            ix = slice(b.start,b.end)
            whole = smooth(t/((b.end-b.start)*dt))
            self.height[ix] = previous.height + whole*(b.height-previous.height)
            if b.swing_end:
                u = np.clip(t/b.swing_end,0.,1.)
                h = smooth(u)
                self.feet[ix] = previous.feet + h[:,None,None]*(b.feet-previous.feet)
                self.feet[ix,b.side,2] += b.lift*(4*u*(1-u))**3
                remain = (b.end-b.start)*dt-b.swing_end
                mix = smooth(np.clip((t-b.swing_end)/remain,0.,1.))
                self.cop[ix] = (previous.cop+h[:,None]*(b.transfer-previous.cop)
                                + mix[:,None]*(b.cop-b.transfer))
            else:
                self.feet[ix] = b.feet
                self.cop[ix] = previous.cop + whole[:,None]*(b.cop-previous.cop)
            previous = b
        tim = np.arange(-125,126)*dt
        omega = np.sqrt(9.81/max(.6,self.origin.height-.14))
        weights = np.exp(-omega*np.abs(tim))
        weights /= weights.sum()
        self.com = np.stack([np.convolve(np.pad(x,125,mode='edge'),weights,mode='valid')
                             for x in self.cop.T],axis=1)
        if self.correction is not None:
            first,value = self.correction
            self.com[first:first+len(value)] = value

    def fill(self, through):
        if through+126 >= self.blocks[-1].end:
            self.extend(through+206)
            self.rebuild()
        for idx in range(len(self.positions),through+1):
            p,q = self.ik.solve(self.feet[idx],self.com[idx],self.height[idx])
            self.positions.append(p.astype(np.float32))
            self.joints.append(q.astype(np.float32))

    def phase(self,frame):
        return next(b for b in self.blocks if b.start<=frame<b.end)

    def goal(self, frame):
        end = frame+LOOKAHEAD+1
        self.fill(end)
        p = np.asarray(self.positions[frame:end+1],dtype=np.float32)-self.offset
        q = np.asarray(self.joints[frame:end+1],dtype=np.float32)
        velocity = np.diff(p,n=1,axis=0)
        v = (velocity[1:]+velocity[:-1])*(.5/dt)
        dq = (q[2:]-q[:-2])*(.5/dt)
        k = LOOKAHEAD
        features = np.concatenate((p[1:-1]-p[0],np.broadcast_to(ROT6,(k,6)),v,np.zeros_like(v),
                                   q[1:-1],.1*dq),axis=1)
        return features.reshape(-1)
