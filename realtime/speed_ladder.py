"""Repeatable live-UDP speed ladder, using separate experimental calibration."""
import argparse
import json
from pathlib import Path
import socket

from .sim import simulate, CALIBRATION

OUT = CALIBRATION.parents[1]/'experiments'/'speed_ladder'
SPEEDS = tuple(i/10 for i in (*range(1,13),*range(11,0,-1),0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--calibration',type=Path,default=OUT/'hold_v04.json')
    p.add_argument('--pace',action='store_true')
    p.add_argument('--dwell',type=float,default=5.)
    a=p.parse_args()
    if a.dwell < 3:
        p.error('Dwell must be at least 3 seconds')
    schedule = [(round(3.+i*a.dwell,6),v) for i,v in enumerate(SPEEDS)]
    a.duration = schedule[-1][0]+6.
    cal = json.loads(a.calibration.read_text())
    max_speed = cal['anchors'][-1]['recipe']['speed']
    if max_speed < max(SPEEDS):
        p.error('Experimental calibration must include 1.2 m/s')
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as receiver, socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sender:
        receiver.bind(('127.0.0.1',0))
        receiver.setblocking(False)
        simulate(a,receiver,sender,schedule=schedule,calibration=a.calibration,max_speed=max_speed)


if __name__=='__main__':
    main()
