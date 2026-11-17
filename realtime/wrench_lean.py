"""Use measured wrist wrenches to counter the sustained forward load at the waist."""
import numpy as np

from gait import contract, dt


class WrenchLean:
    fields=('raw_forward_torque_nm','filtered_forward_torque_nm','target_chest_pitch_deg',
            'actual_chest_pitch_deg','waist_target_rad','wrench_valid',
            'reference_com_shift_m','filtered_support_force_n')

    def __init__(self,plant,sensors,gain_deg_nm,limit_deg,preview,com_gain,wrench_arm_support):
        self.d=plant.data
        self.chest=plant.model.body('robot/waist_pitch_link').id
        self.pelvis=plant.model.body('robot/pelvis').id
        self.sites=sensors.ids
        self.index=contract['joint_names'].index('waist_pitch_joint')
        model=plant.model
        self.qadr=plant.qadr[self.index]
        jid=model.joint('robot/waist_pitch_joint').id
        self.limits=model.jnt_range[jid]+[.005,-.005]
        self.kp=model.actuator_gainprm[plant.aids[self.index],0]
        self.gain=np.deg2rad(gain_deg_nm)
        self.limit=np.deg2rad(limit_deg)
        self.filtered=0.
        self.vertical=0.
        self.preview=preview
        self.com_gain=com_gain
        self.wrench_arm_support=wrench_arm_support
        self.g=abs(model.opt.gravity[2])
        self.robot_weight=model.body_subtreemass[self.pelvis]*self.g
        self.com_shift=0.
        self.target_pitch=0.
        self.smoothing=1-np.exp(-dt/.32)

    def arm_mass(self,default):
        return self.vertical/self.g if self.wrench_arm_support else default

    def grip_engaged(self,gpos,grot):
        return np.max(gpos)<.015 and np.max(grot)<.15

    def arm_targets(self,holder,mass):
        return holder.targets(mass)

    def waist_target(self,measured):
        return self.d.qpos[self.qadr]+.95*(self.target_pitch-measured)-self.filtered/self.kp

    def step(self,target,wrist_world,engaged):
        d=self.d
        R=d.xmat[self.pelvis].reshape(3,3)
        heading=np.arctan2(R[1,0],R[0,0])
        robot_left=np.array([-np.sin(heading),np.cos(heading),0.])
        # The sensor reports the supporting wrench from the robot. Its reaction
        # produces a forward moment on the robot about the waist.
        arms=d.site_xpos[self.sites]-d.xpos[self.chest]
        f,t=wrist_world[:,0],wrist_world[:,1]
        raw=-float(np.sum(t+np.cross(arms,f),axis=0)@robot_left)
        valid=bool(engaged and np.isfinite(raw) and f[:,2].sum()>0.)
        if valid:
            bounded=float(np.clip(raw,0.,self.kp*.20))
            self.filtered+=self.smoothing*(bounded-self.filtered)
            vertical=float(np.clip(np.sum(f[:,2]),0.,160.))
            self.vertical+=self.smoothing*(vertical-self.vertical)
        desired=-min(self.gain*self.filtered,self.limit)
        self.target_pitch+=float(np.clip(desired-self.target_pitch,-dt*.50,dt*.50))
        self.com_shift=-self.com_gain*self.filtered/(self.robot_weight+self.vertical)
        if self.com_gain:
            self.preview.lean_reference=(self.target_pitch,self.com_shift)
        measured=-np.arcsin(np.clip(d.xmat[self.chest,6],-1.,1.))
        target[self.index]=np.clip(self.waist_target(measured),*self.limits)
        return [raw,self.filtered,np.rad2deg(self.target_pitch),np.rad2deg(measured),
                target[self.index],float(valid),self.com_shift,self.vertical]
