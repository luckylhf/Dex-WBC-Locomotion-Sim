"""Run the calibrated speed-conditioned reference under record12 in native MuJoCo."""
from dataclasses import replace

import numpy as np

from gait import Recipe

PARAMS = ('stride_scale', 'double', 'lift', 'crouch', 'heading_trim', 'gather_trim', 'narrow', 'support_y')


def recipe_for(speed, calibration):
    if speed == 0:
        cfg = recipe_for(.1,calibration)
        return replace(cfg,name='stand',speed=0.,steps=0)
    rows=[Recipe(**item['recipe']) for item in calibration['anchors']]
    speeds=[r.speed for r in rows]
    if not speeds[0] <= speed <= speeds[-1]:
        raise ValueError(f'Use 0 for stand or a forward speed from {speeds[0]:g} to {speeds[-1]:g} m/s')
    fields={key:float(np.interp(speed,speeds,[getattr(r,key) for r in rows])) for key in PARAMS}
    if all(r.gather_trim_right is not None for r in rows):
        fields['gather_trim_right']=float(np.interp(speed,speeds,[r.gather_trim_right for r in rows]))
    return replace(rows[0], name=f'v{speed:.3f}',speed=speed,**fields)

