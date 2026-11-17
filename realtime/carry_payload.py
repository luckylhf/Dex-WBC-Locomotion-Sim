"""Known mass in MuJoCo physics, wrist-only estimation, forward stand/walk/stop."""
import json
import math
from pathlib import Path

import numpy as np

from gait import contract, dt
from .carry_empty import EMPTY_TARE_KG, argument_parser, run
from .carry_omni import CarryOmniPreview, make_trials
from .wrench_lean import WrenchLean
from .omni_preview import yaw_rot



class CarryPayloadPreview(CarryOmniPreview):
    def __init__(self,calibration,arms,profile):
        self.estimated_added_mass=0.
        self.load_gather_log=[]
        self.lean_reference=None
        super().__init__(calibration,arms,profile)

    def observe_load(self,total_estimate):
        self.estimated_added_mass=max(0.,total_estimate-EMPTY_TARE_KG)

    def com_at(self,index):
        center=super().com_at(index)
        if self.lean_reference is None:
            return center
        pitch,shift=self.lean_reference
        waist=contract['joint_names'].index('waist_pitch_joint')
        self.ik.data.qpos[self.ik.qadr[waist]]=pitch
        return center+shift*yaw_rot(self.yaw[index])[:2,0]

    def park_trim(self,cmd,side,cfg):
        nominal=super().park_trim(cmd,side,cfg)
        gain=float(np.interp(cmd[0],[.1,.2,.3,.4],[0.,0.,.026,.019]))
        correction=gain*self.estimated_added_mass*(-1. if side==0 else 1.)
        self.load_gather_log.append(dict(start_s=self.blocks[-1].end*dt,
            forward_command_mps=cmd[0],side='LR'[side],measured_added_mass_kg=self.estimated_added_mass,
            nominal_trim_m=nominal,measured_load_offset_m=correction))
        return nominal+correction


def main():
    p=argument_parser(__doc__)
    p.set_defaults(grip_softness=.075)
    p.add_argument('--added-mass',type=float,default=1.,help='Mass added across the two wrist grips, excluding the shell')
    p.add_argument('--profile',type=Path,default=Path(__file__).with_name('carry_omni_calibration.json'))
    p.add_argument('--wrench-lean',action='store_true',help='Test waist pitch driven by the measured wrist wrench')
    p.add_argument('--lean-gain-deg-nm',type=float,default=.26)
    p.add_argument('--lean-limit-deg',type=float,default=16.)
    p.add_argument('--lean-com-gain',type=float,default=0.)
    p.add_argument('--wrench-arm-support',action='store_true')
    args=p.parse_args()
    if not math.isfinite(args.added_mass) or args.added_mass<0. or not .1<=args.mass<1.:
        p.error('Use a nonnegative added mass and an empty shell in [0.1,1.0) kg')
    if not math.isfinite(args.stand_seconds) or args.stand_seconds<6.:
        p.error('Independent stand duration must be at least 6 seconds')
    profile=json.loads(args.profile.read_text())
    trials=[] if args.stand_only else make_trials()[:4]
    # Ground truth is only passed into the physical model construction.
    if not all(math.isfinite(x) and x>=0. for x in (args.lean_gain_deg_nm,args.lean_limit_deg,args.lean_com_gain)):
        p.error('Lean gain and limit must be finite and nonnegative')
    def balance_factory(plant,sensors,preview):
        return WrenchLean(plant,sensors,args.lean_gain_deg_nm,args.lean_limit_deg,
                          preview,args.lean_com_gain,args.wrench_arm_support)
    run(args,trials=trials,build_preview=lambda c,a:CarryPayloadPreview(c,a,profile),
        added_mass=args.added_mass,observe_load=CarryPayloadPreview.observe_load,
        balance_factory=balance_factory if args.wrench_lean else None)


if __name__=='__main__':
    main()
