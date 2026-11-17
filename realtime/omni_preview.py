"""Signed planar footstep planning for the frozen record12 reference controller."""
from dataclasses import dataclass, replace

import numpy as np

from gait import dt, smooth
from .preview import LOOKAHEAD, STEP, Block, Preview

ZERO = (0.,0.,0.)
ZERO_FEET = np.zeros(2)
DEFAULTS = dict(back_stride=1.07,back_lift=.055,back_double=.155,back_crouch=.048,
                back_heading=0.,back_gather_l=0.,back_gather_r=0.,side_stride=1.11,side_lift=.060,
                side_double=.18,side_crouch=.057,side_tuck=.055,side_forward=0.,side_stride_right=None,
                yaw_gain=1.0,yaw_lift=.051,yaw_double=.19,yaw_crouch=.048,
                yaw_center_x=.0,yaw_center_y=.0,yaw_step_lead=.45,
                side_land_ap=0.,yaw_stance_width=1.0)


def yaw_rot(yaw):
    yaw = np.asarray(yaw)
    c,s = np.cos(yaw),np.sin(yaw)
    matrix = np.zeros(yaw.shape+(3,3))
    matrix[...,0,0] = c
    matrix[...,0,1] = -s
    matrix[...,1,0] = s
    matrix[...,1,1] = c
    matrix[...,2,2] = 1
    return matrix


@dataclass
class OmniBlock(Block):
    yaw: float = 0.
    foot_yaw: np.ndarray | None = None
    cmd: tuple = ZERO


class OmniPreview(Preview):
    def __init__(self, calibration, options):
        self.intent = ZERO
        self.angles = []
        self.fixed_foot_rotation = None
        self.options = DEFAULTS | options
        super().__init__(calibration)
        self.origin = OmniBlock(**vars(self.origin))
        self.blocks = [OmniBlock(**vars(b)) for b in self.blocks]

    def add(self,kind,duration,side,feet,cop,height,xy,speed,swing=0.,lift=0.,**extras):
        start = self.blocks[-1].end
        self.blocks.append(OmniBlock(start,start+duration,kind,side,feet,cop,height,xy,speed,
                                    swing,lift,**extras))

    @staticmethod
    def feet_angles(block):
        value = getattr(block,'foot_yaw',None)
        return ZERO_FEET if value is None else value

    @staticmethod
    def center_at(feet, yaw, cfg):
        return feet[:,:2].mean(0)+yaw_rot(yaw)[:2,:2] @ [cfg.x_cop,0.]

    @staticmethod
    def stance_at(feet, side, angle, cfg):
        return feet[side,:2]+yaw_rot(angle)[:2,:2] @ [cfg.x_cop,(2*side-1)*cfg.support_y]

    def tuning(self,cmd):
        return self.options

    def recipe(self,cmd):
        vx,vy,w = cmd
        o = self.tuning(cmd)
        if vx>0:
            return self.cfg(max(.1,vx))
        cfg = self.cfg(.3 if vy or vx else .1)
        if vx:
            return replace(cfg,stride_scale=o['back_stride'],lift=o['back_lift'],
                           crouch=o['back_crouch'],double=o['back_double'])
        if vy:
            stride = (o['side_stride_right'] or o['side_stride']) if vy<0 else o['side_stride']
            return replace(cfg,stride_scale=stride,lift=o['side_lift'],
                           crouch=o['side_crouch'],double=o['side_double'])
        return replace(cfg,lift=o['yaw_lift'],crouch=o['yaw_crouch'],double=o['yaw_double'])

    def start(self,b):
        vx,vy,w = self.intent
        cfg = self.recipe(self.intent)
        side = int(vy<0 or w<0)
        angles = self.feet_angles(b)
        center = b.feet[:,:2].mean(0)-yaw_rot(b.yaw)[:2,:2]@self.ik.neutral_feet[:,:2].mean(0)
        self.add('shift',round(cfg.first_hold/dt),side,b.feet,
                 self.stance_at(b.feet,1-side,angles[1-side],cfg),
                 self.ik.neutral_height-cfg.crouch,center,max(map(abs,self.intent)),
                 yaw=b.yaw,foot_yaw=angles,cmd=self.intent)

    def move(self,b,cmd,factor,kind='walk'):
        vx,vy,w = cmd
        cfg = self.recipe(cmd)
        o = self.tuning(cmd)
        side = b.side if b.kind=='shift' else 1-b.side
        theta_step = w*STEP*dt*o['yaw_gain']*factor
        yaw = b.yaw+theta_step
        dx = vx*cfg.stride_scale
        dy = (dx*cfg.heading_trim if vx>=0 else -dx*o['back_heading'])+vy*cfg.stride_scale
        if vy:
            dx += abs(dy)*o['side_forward']
        if w:
            dx += abs(w)*o['yaw_center_x']
            dy += abs(w)*o['yaw_center_y']
        xy = b.path_xy+yaw_rot(.5*(b.yaw+yaw))[:2,:2] @ [dx*STEP*dt*factor,dy*STEP*dt*factor]
        feet = b.feet.copy()
        foot_yaw = self.feet_angles(b).copy()
        foot_yaw[side] = yaw+o['yaw_step_lead']*theta_step
        offset = self.ik.neutral_feet[side,:2].copy()
        if w:
            offset[1] *= o['yaw_stance_width']
        if vy:
            offset[0] += (1-2*side)*o['side_land_ap']
            if side != int(vy<0):
                offset[1] -= np.sign(vy)*o['side_tuck']*factor
        feet[side,:2] = xy + yaw_rot(foot_yaw[side])[:2,:2] @ offset
        stance = self.stance_at(b.feet,1-side,foot_yaw[1-side],cfg)
        goal_cop = self.stance_at(feet,side,foot_yaw[side],cfg)
        self.add(kind,STEP,side,feet,goal_cop,self.ik.neutral_height-cfg.crouch,xy,
                 max(abs(vx),abs(vy),abs(w)),STEP*dt-cfg.double,cfg.lift,transfer=stance,
                 yaw=yaw,foot_yaw=foot_yaw,cmd=cmd)

    def park_trim(self,cmd,side,cfg):
        vx,_,_=cmd
        if vx>0:
            return cfg.gather_trim_right if side else cfg.gather_trim
        if vx<0:
            return self.tuning(cmd)['back_gather_r' if side else 'back_gather_l']
        return 0.

    def park_center(self,feet,yaw,cfg,cmd):
        return self.center_at(feet,yaw,cfg)

    def close(self,b):
        cfg = self.recipe(b.cmd)
        _,_,w = b.cmd
        side = 1-b.side
        feet = b.feet.copy()
        angles = self.feet_angles(b).copy()
        yaw = float(angles[1-side]) if w else b.yaw
        angles[side] = yaw
        neutral = self.ik.neutral_feet
        trim = self.park_trim(b.cmd,side,cfg)
        diff = neutral[side,:2]-neutral[1-side,:2] + [trim,0.]
        feet[side,:2] = feet[1-side,:2] + yaw_rot(yaw)[:2,:2] @ diff
        swing = STEP*dt-cfg.double
        self.add('gather',round((swing+cfg.gather)/dt),side,feet,self.park_center(feet,yaw,cfg,b.cmd),
                 b.height,b.path_xy,0.,swing,cfg.lift,
                 transfer=self.stance_at(b.feet,1-side,angles[1-side],cfg),
                 yaw=yaw,foot_yaw=angles)

    def hold(self,b,n):
        self.add('stand',n,b.side,b.feet,b.cop,b.height,b.path_xy,0.,
                 yaw=b.yaw,foot_yaw=self.feet_angles(b))

    def extend(self,through):
        while self.blocks[-1].end < through:
            b=self.blocks[-1]
            if b.kind in ('stand','gather'):
                if self.intent!=ZERO and b.kind=='stand':
                    self.start(b)
                else:
                    self.hold(b,22 if self.intent!=ZERO else 200)
            else:
                n=0
                for prior in reversed(self.blocks):
                    if prior.kind!='walk':
                        break
                    n+=1
                self.move(b,self.intent,(.8,.9)[n] if n<2 else 1.)

    def command(self,frame,cmd):
        if self.intent==cmd:
            return
        earliest=frame+LOOKAHEAD+2
        splice=next(i for i,b in enumerate(self.blocks) if b.end>earliest)
        b=self.blocks[splice]
        self.blocks=self.blocks[:splice]+[replace(b,end=earliest) if b.kind=='stand' else b]
        self.intent=cmd
        b=self.blocks[-1]
        if b.kind=='brake':
            self.close(b)
        elif cmd==ZERO and b.kind in ('shift','walk'):
            self.move(b,b.cmd,.80,'brake')
            self.move(self.blocks[-1],b.cmd,.70,'brake')
            self.close(self.blocks[-1])
        old=self.com
        self.extend(frame+260)
        self.rebuild()
        first=len(self.positions)
        fade=min(19,len(old)-first)
        if fade>1:
            ix=slice(first,first+fade)
            weight=smooth(np.linspace(0.,1.,fade))[:,None]
            self.com[ix]=(1-weight)*old[ix]+weight*self.com[ix]
            self.correction=(first,self.com[ix].copy())
        self.revisions.append(dict(frame=frame,command=cmd,splice_frame=b.end))

    def rebuild(self):
        super().rebuild()
        n=len(self.cop)
        self.yaw=np.zeros(n)
        self.foot_yaw=np.zeros((n,2))
        previous=self.origin
        for b in self.blocks:
            samples=np.arange(b.end-b.start)*dt
            if not len(samples):
                continue
            region=slice(b.start,b.end)
            h=smooth(samples/((b.end-b.start)*dt))
            old_yaw=getattr(previous,'yaw',0.)
            self.yaw[region]=old_yaw+h*(getattr(b,'yaw',0.)-old_yaw)
            w = smooth(np.clip(samples/b.swing_end,0.,1.)) if b.swing_end else h
            past,after=self.feet_angles(previous),self.feet_angles(b)
            self.foot_yaw[region]=past+w[:,None]*(after-past)
            previous=b

    def com_at(self,index):
        return self.com[index]

    def fill(self,through):
        if through+126>=self.blocks[-1].end:
            self.extend(through+206)
            self.rebuild()
        if self.fixed_foot_rotation is None:
            self.fixed_foot_rotation=self.ik.rot.copy()
        for idx in range(len(self.positions),through+1):
            rotation = self.yaw[idx]
            self.ik.data.qpos[3:7]=[np.cos(rotation/2),0.,0.,np.sin(rotation/2)]
            self.ik.rot=yaw_rot(self.foot_yaw[idx]) @ self.fixed_foot_rotation
            p,q=self.ik.solve(self.feet[idx],self.com_at(idx),self.height[idx])
            self.positions.append(p.astype(np.float32))
            self.joints.append(q.astype(np.float32))
            self.angles.append(rotation)

    def goal(self,frame):
        end=frame+LOOKAHEAD+1
        self.fill(end)
        yaw=np.asarray(self.angles[frame:end+1])
        rotations=yaw_rot(yaw)
        p=np.asarray(self.positions[frame:end+1])-self.offset
        q=np.asarray(self.joints[frame:end+1])
        velocity=(p[2:]-p[:-2])*(.5/dt)
        velocity=np.einsum('ni,nij->nj',velocity,rotations[1:-1])
        angular=np.zeros_like(velocity)
        angular[:,2]=(yaw[2:]-yaw[:-2])*(.5/dt)
        difference=(p[1:-1]-p[0])@rotations[0]
        relative=yaw_rot(yaw[1:-1]-yaw[0])[:,:,:2].reshape(LOOKAHEAD,6)
        goal=np.concatenate((difference,relative,velocity,angular,q[1:-1],(q[2:]-q[:-2])*(.05/dt)),axis=1)
        return goal.astype(np.float32).reshape(-1)
