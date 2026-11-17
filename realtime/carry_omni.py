"""Continuous six-axis speed ladder with the physical empty box."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .carry_empty import argument_parser, run
from .carry_model import ARM_ORDER
from .omni_preview import OmniPreview

FIRST=6.
MOVE=7.
HOLD=5.
COMMANDS=[
    ('Forward',0,1,[.1,.2,.3,.4]), ('Backward',0,-1,[.1,.2,.3]),
    ('Left',1,1,[.1,.2,.3]), ('Right',1,-1,[.1,.2,.3]),
    ('Turn left',2,1,[.15,.3,.45]), ('Turn right',2,-1,[.15,.3,.45]),
]


def make_trials():
    trials=[]
    for name,axis,sign,speeds in COMMANDS:
        for speed in speeds:
            cmd=[0.,0.,0.]
            cmd[axis]=sign*speed
            t=FIRST+len(trials)*(MOVE+HOLD)
            trials.append(dict(label=name,command=cmd,start_s=t,zero_s=t+MOVE,end_s=t+MOVE+HOLD))
    return trials


class CarryOmniPreview(OmniPreview):
    def __init__(self,calibration,arms,profile):
        self.arms=arms
        self.profile=profile
        super().__init__(calibration,profile.get('omni',{}))

    def fill(self,through):
        self.ik.data.qpos[self.ik.qadr[ARM_ORDER]]=self.arms
        super().fill(through)

    def values(self,cmd,section):
        axis=next((i for i,v in enumerate(cmd) if v),None)
        if axis is None:
            return {}
        label=(('Forward','Backward'),('Left','Right'),('Turn left','Turn right'))[axis][int(cmd[axis]<0)]
        overrides=self.profile.get(section,{}).get(label,{})
        if not overrides:
            return {}
        speeds=overrides['speeds']
        return {key:float(np.interp(abs(cmd[axis]),speeds,value))
                for key,value in overrides.items() if key!='speeds'}

    def tuning(self,cmd):
        return self.options | self.values(cmd,'tuning')

    def recipe(self,cmd):
        return replace(super().recipe(cmd),**self.values(cmd,'ranges'))

    def park_trim(self,cmd,side,cfg):
        values=self.values(cmd,'parking')
        val=values.get('right' if side else 'left')
        return super().park_trim(cmd,side,cfg) if val is None else val


def main():
    p=argument_parser(__doc__)
    p.set_defaults(grip_softness=.075)
    p.add_argument('--profile',type=Path,default=Path(__file__).with_name('carry_omni_calibration.json'))
    p.add_argument('--through',type=int,help='Stop after this many consecutive slots')
    args=p.parse_args()
    if args.stand_only:
        p.error('Use realtime.carry_empty --stand-only for independent static verification')
    if not .1<=args.mass<1.:
        p.error('Empty-box mass must lie in [0.1,1.0) kg')
    trials=make_trials()
    if args.through is not None:
        if not 1<=args.through<=len(trials):
            p.error('through must select at least one and no more than 19 slots')
        trials=trials[:args.through]
    profile=json.loads(args.profile.read_text())
    run(args,trials=trials,build_preview=lambda c,a:CarryOmniPreview(c,a,profile))


if __name__=='__main__':
    main()
