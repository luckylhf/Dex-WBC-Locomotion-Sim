"""Render the physical box and robot from the saved MuJoCo state."""
import argparse
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from gait import dt
from .video import FONT,BACKGROUND,TEXT,ACTUAL,REQUEST,LABEL
from .carry_report import angles
from .omni_preview import yaw_rot

W,H=600,444
SIZE=2*W,566


def render(folder,out):
    run=json.loads((folder/'run.json').read_text())
    data=np.load(folder/'rollout.npz')
    q=data['qpos']
    yaw=angles(q[:,3:7])[:,2]
    stages=data['stage']
    wrist=data['wrist_compensated_world']
    estimates=data['estimates']
    bins=data['estimated_mass_bin'] if 'estimated_mass_bin' in data.files else None
    added=data['estimated_added_mass_kg'] if 'estimated_added_mass_kg' in data.files else None
    box_label='Loaded box' if run['added_mass_kg']>0 else 'Empty box'
    gaps=data['grip_errors']
    balance=data['balance_trace'] if 'balance_trace' in data.files and data['balance_trace'].ndim==2 else None
    transfer=run.get('load_transfer')
    settle_report=folder/'settle_metrics.json'
    settled=json.loads(settle_report.read_text())['stable_at_s'] if transfer and settle_report.exists() else None
    trials=run.get('trials',[])
    if transfer and trials and (folder/'metrics.json').exists():
        settled=json.loads((folder/'metrics.json').read_text()).get('initial_transfer',{}).get('stable_at_s')
    box_pitch=np.rad2deg(angles(q[:,run['box_free_qadr']+3:run['box_free_qadr']+7])[:,1])
    starts=np.array([trial['start_s'] for trial in trials])
    trace=np.load(folder/'telemetry.npy')
    model=mujoco.MjModel.from_binary_path(str(folder/'box_model.mjb'))
    model.vis.global_.offwidth=W
    model.vis.global_.offheight=H
    state=mujoco.MjData(model)
    camera=mujoco.MjvCamera()
    camera.distance=2.83
    camera.elevation=-13
    font=ImageFont.truetype(FONT,18)
    title=ImageFont.truetype(FONT,25)
    temp=out.with_suffix('.part.mp4')
    cmd=['ffmpeg','-nostdin','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{SIZE[0]}x{SIZE[1]}','-r','25','-i','-','-an','-c:v','libx264',
         '-threads','3','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model,height=H,width=W) as renderer,process.stdin as pipe:
            for i in range(0,len(q),2):
                t=i*dt
                state.qpos[:]=q[i]
                mujoco.mj_forward(model,state)
                trial_index=int(np.searchsorted(starts,t,side='right'))-1
                trial=trials[trial_index] if trial_index>=0 else None
                heading=yaw[round(trial['start_s']/dt)] if trial else 0.
                focus=q[i,:2]+.19*yaw_rot(yaw[i])[:2,0]
                camera.lookat[:]=[*focus,.92]
                canvas=Image.new('RGB',SIZE,BACKGROUND)
                draw=ImageDraw.Draw(canvas)
                for col,(azimuth,side) in enumerate(((180.,'Front'),(90.,'Side'))):
                    camera.azimuth=azimuth+np.rad2deg(heading)
                    renderer.update_scene(state,camera)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=0
                    x=col*W
                    canvas.paste(Image.fromarray(renderer.render()),(x,46))
                    draw.text((x+14,58),side,font=title,fill='white',stroke_width=2,stroke_fill=BACKGROUND)
                stage=LABEL[str(stages[i])]
                zero=trial['zero_s'] if trial else run['zero_s']
                if stage=='Stand' and zero is not None and t>=zero:
                    stage='Park hold'
                segment=f' / {trial_index+1:02d}/{len(trials):02d} {trial["label"]}' if trial else ''
                header=f'v7 / {box_label} / {t:06.2f} s{segment} / Ref: {stage}'
                if transfer and trial:
                    axis=next(k for k,value in enumerate(trial['command']) if value)
                    pace=abs(trial['command'][axis])
                    unit='rad/s' if axis==2 else 'm/s'
                    header=(f'v7 / +{run["added_mass_kg"]:g} kg / {t:06.2f} s / {trial_index+1}/{len(trials)}  '
                            f'{trial["label"]} {pace:.2f} {unit} / {stage}')
                elif transfer:
                    if t < transfer['start_s']:
                        phase='Standing before load'
                    elif t < transfer['fully_transferred_s']:
                        phase='Transferring load to wrists'
                    elif settled is None or t < settled-1e-6:
                        phase='Holding load / settling'
                    else:
                        phase='Holding load / settled'
                    header=f'v7 / Wrist load transfer / {t:06.2f} s / {phase}'
                draw.text((14,10),header,font=title,fill=TEXT)
                ids=slice(max(0,i-5),min(len(q),i+6))
                l,r=wrist[ids,:,0,2].mean(0)
                est=estimates[i]
                mass=f'{est[1]:.3f} kg' if est[2] else 'window pending'
                if added is not None and bins[i]>0 and run['added_mass_kg']>0:
                    mass+=f' (+{added[i]:.3f})'
                vx=float(trace[ids,4].mean())
                detail=f'Actual forward {vx:+.3f} m/s'
                if trial:
                    axis=next(k for k,v in enumerate(trial['command']) if v)
                    test=trial['command'][axis]
                    unit='rad/s' if axis==2 else 'm/s'
                    v=0. if t>=zero else test
                    if axis==2:
                        a,b=max(0,i-5),min(len(q)-1,i+5)
                        vx=float((yaw[b]-yaw[a])/((b-a)*dt))
                    else:
                        vx=float(trace[ids,4:6].mean(0) @ yaw_rot(heading)[:2,axis])
                    detail=f'Cmd {v:+.2f}  |  measured {vx:+.3f} {unit}'
                draw.text((14,500),f'{detail}   |   Wrist support L/R {l:.2f} / {r:.2f} N',
                          font=font,fill=TEXT)
                if transfer and balance is not None:
                    mass=f'{balance[min(i,len(balance)-1),7]/9.81:.2f} kg (filtered wrist force)'
                    draw.text((W+120 if trials else W+90,500),f'Est. total: {mass}',font=font,fill=ACTUAL)
                else:
                    draw.text((W+172,500),f'Estimated mass: {mass}',font=font,fill=ACTUAL if est[2] else REQUEST)
                gap=gaps[i,:,0].max()*1000
                grip=f'Dual grip displacement {gap:.1f} mm'
                if transfer:
                    if t >= transfer['fully_transferred_s']:
                        grip+=f'  |  Fully transferred +{run["added_mass_kg"]:g} kg'
                    else:
                        grip+='  |  Added weight externally supported'
                else:
                    grip+=f'  |  Physical box mass {run["model_box_mass_kg"]:.2f} kg'
                draw.text((14,529),grip,font=font,fill=TEXT)
                distance=float(np.linalg.norm(q[i,:2]-q[0,:2]))
                if not trials:
                    if balance is None:
                        info=f'Root displacement {distance:.2f} m'
                    else:
                        row=balance[min(i,len(balance)-1)]
                        info=f'Chest pitch {row[3]:+.1f} deg  |  Root drift {distance:.2f} m'
                    draw.text((W+172,529),info,font=font,fill=ACTUAL)
                if transfer and trials:
                    draw.text((W+120,529),f'Box pitch {box_pitch[i]:+.1f} deg  |  Chest pitch {trace[i,6]:+.1f} deg',
                              font=font,fill=ACTUAL)
                if trials:
                    path_width=492
                    x0=SIZE[0]-path_width-15
                    y0=556 if transfer else 544
                    draw.line((x0,y0,x0+path_width,y0),fill='#39465b',width=3)
                    draw.line((x0,y0,x0+path_width*i/(len(q)-1),y0),fill=ACTUAL,width=3)
                    for start in starts:
                        mark=x0+path_width*start/(len(q)-1)/dt
                        draw.line((mark,y0-5,mark,y0+4),fill=TEXT,width=1)
                pipe.write(canvas.tobytes())
        if process.wait()!=0:
            raise RuntimeError('ffmpeg encode failed')
        temp.replace(out)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('folder',type=Path)
    p.add_argument('out',type=Path)
    a=p.parse_args()
    render(a.folder,a.out)
