"""Synchronized front and side views with a plan-view trace of each test."""
import argparse
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gait import bundle, dt
from .omni_preview import yaw_rot
from .video import LABEL, FONT, BACKGROUND, TEXT, ACTUAL, REQUEST

W,H=600,404
WIDTH,HEIGHT=2*W,584


def render(folder,out):
    q = np.load(folder/'rollout.trajectory.npz')['qpos']
    t = np.load(folder/'telemetry.npy')
    c = np.load(folder/'commands.npz')
    gaps = np.load(folder/'clearance.npy')
    trials=json.loads((folder/'run.json').read_text())['trials']
    w,x,y,z=q[:,3:7].T
    yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    model=mujoco.MjModel.from_binary_path(str(bundle/'model.mjb'))
    model.vis.global_.offwidth=W
    model.vis.global_.offheight=H
    data=mujoco.MjData(model)
    camera=mujoco.MjvCamera()
    camera.distance=2.85
    camera.elevation=-9
    fonts={n:ImageFont.truetype(FONT,n) for n in (15,17,23)}
    for trial in trials:
        s=round(trial['start_s']/dt)
        e=min(round(trial['end_s']/dt),len(q)-1)
        if s>=len(q):
            continue
        trial['heading']=yaw[s]
        shift=(q[s:e+1,:2]-q[s,:2])@yaw_rot(yaw[s])[:2,:2]
        trial['shift']=shift
        trial['path_scale']=min(92.,48/max(.19,np.max(np.abs(shift))))
    tmp=out.with_suffix('.part.mp4')
    cmd=['ffmpeg','-nostdin','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',
         f'{WIDTH}x{HEIGHT}','-r','25','-i','-','-an','-c:v','libx264','-threads','3','-preset','fast',
         '-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)]
    ff=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model,height=H,width=W) as renderer, ff.stdin as pipe:
            for i in range(0,len(q),2):
                seconds=i*dt
                trial=next((x for x in reversed(trials) if x['start_s']<=seconds),None)
                heading=trial['heading'] if trial else 0.
                data.qpos[:]=q[i]
                mujoco.mj_forward(model,data)
                canvas=Image.new('RGB',(WIDTH,HEIGHT),BACKGROUND)
                draw=ImageDraw.Draw(canvas)
                camera.lookat[:]=[data.qpos[0],data.qpos[1],.85]
                for col,(label,base) in enumerate((('Front',180.),('Side',90.))):
                    camera.azimuth=base+np.rad2deg(heading)
                    renderer.update_scene(data,camera)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=0
                    start=col*W
                    canvas.paste(Image.fromarray(renderer.render()),(start,46))
                    draw.text((start+12,58),label,font=fonts[23],fill='white',stroke_width=2,stroke_fill=BACKGROUND)
                stage=LABEL[str(c['stage'][i])]
                command=c['requested'][i]
                a,b=max(0,i-9),min(len(q)-1,i+9)
                world_v=(q[b,:2]-q[a,:2])/((b-a)*dt)
                current=yaw_rot(heading)[:2,:2]
                actual=np.r_[world_v@current,(yaw[b]-yaw[a])/((b-a)*dt)]
                name=trial['label'] if trial else 'Initial stand'
                draw.text((14,11),f'v7  /  {seconds:05.2f} s  /  {name}  /  {stage}',
                          font=fonts[23],fill=TEXT)
                axis=int(np.argmax(np.abs(trial['command']))) if trial else 0
                unit='rad/s' if axis==2 else 'm/s'
                draw.text((16,464),f'Command {command[axis]:+.2f} {unit}   |   Measured {actual[axis]:+.3f} {unit}',
                          font=fonts[17],fill=TEXT)
                feet=t[i,10:12]*100
                foot_gap='10 cm or more' if gaps[i,:2].min()>=.1 else f'{100*gaps[i,:2].min():.1f} cm'
                draw.text((16,489),f'Foot load L/R  {feet[0]:.0f}% / {feet[1]:.0f}% BW   |   Foot / opposite leg gap {foot_gap}',
                          font=fonts[17],fill=TEXT)
                draw.text((WIDTH-332,462),'Top view relative to segment start',font=fonts[15],fill=TEXT)
                cx,cy=WIDTH-145,519
                draw.line((cx-52,cy,cx+52,cy),fill='#405064',width=1)
                draw.line((cx,cy-47,cx,cy+47),fill='#405064',width=1)
                draw.text((cx+54,cy-10),'L',font=fonts[15],fill=TEXT)
                draw.text((cx+4,cy-49),'F',font=fonts[15],fill=TEXT)
                if trial:
                    shift=trial['shift'][:i-round(trial['start_s']/dt)+1]*trial['path_scale']
                    points=np.c_[cx+shift[:,1],cy-shift[:,0]]
                    if len(points)>1:
                        draw.line([tuple(row) for row in points],fill=ACTUAL,width=2)
                    x0,y0=points[-1]
                    draw.ellipse((x0-3,y0-3,x0+3,y0+3),fill=ACTUAL)
                    rot=yaw[i]-heading
                    draw.line((x0,y0,x0+12*np.sin(rot),y0-12*np.cos(rot)),fill=REQUEST,width=2)
                draw.line((16,554,820,554),fill='#405064',width=3)
                progress=16+804*i/(len(q)-1)
                draw.line((16,554,progress,554),fill=ACTUAL,width=3)
                for tr in trials:
                    tick=16+804*tr['start_s']/(len(q)-1)/dt
                    draw.line((tick,550,tick,558),fill=TEXT,width=1)
                pipe.write(canvas.tobytes())
        if ff.wait()!=0:
            raise RuntimeError('ffmpeg encode failed')
        tmp.replace(out)
    finally:
        if ff.poll() is None:
            ff.terminate()
            ff.wait()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('folder',type=Path)
    p.add_argument('out',type=Path)
    a=p.parse_args()
    render(a.folder,a.out)
