from pathlib import Path
import json
from dataclasses import dataclass
import numpy as np

root = Path(__file__).resolve().parent / 'baseline'
bundle=root/'bundle'
contract=json.loads((bundle/'contract.json').read_text())
dt=.02

def smooth(x):
    return x**3*(10+x*(-15+6*x))
def bouts(mask):
    return np.flatnonzero(np.diff(np.r_[False,mask,False])).reshape(-1,2)

@dataclass
class Recipe:
    name:str
    step:float
    double:float
    crouch:float
    lift:float
    support_y:float
    x_cop:float=.025
    dynamic_mix:float=1.
    first_hold:float=.75
    lead:float=.5
    narrow:float=0.
    stride_scale:float=1.
    speed:float=.1
    steps:int=16
    stand:float=3.
    park:float=6.
    gather:float=.65
    start_ramp:int=2
    stop_ramp:int=2
    initial_seed:int=0
    heading_trim:float=0.
    gather_trim:float=0.
    speed_schedule:tuple=()
    stand_only:bool=False
    gather_trim_right:float|None=None
