"""Render synchronized front and side views from recorded simulation poses."""
import argparse
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gait import bundle, dt

LABEL = {'stand':'Stand', 'shift':'Weight shift', 'walk':'Walk',
         'brake':'Brake', 'gather':'Bring feet level'}

WIDTH, VIEW_HEIGHT = 600, 404
VIDEO_WIDTH, VIDEO_HEIGHT = 2*WIDTH, 574
BACKGROUND = '#151b25'
TEXT = '#e8edf6'
REQUEST = '#f4bc70'
ACTUAL = '#71dbea'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def render(folder, out):
    poses = np.load(folder/'rollout.trajectory.npz')['qpos']
    t = np.load(folder/'telemetry.npy')
    commands = np.load(folder/'commands.npz')
    run = json.loads((folder/'run.json').read_text())
    highest_command = float(commands['requested'].max())
    vx = t[:,4]
    window = 17
    speed = np.convolve(np.pad(vx,window//2,mode='edge'),np.ones(window)/window,mode='valid')
    m = mujoco.MjModel.from_binary_path(str(bundle/'model.mjb'))
    m.vis.global_.offwidth = WIDTH
    m.vis.global_.offheight = VIEW_HEIGHT
    d = mujoco.MjData(m)
    camera = mujoco.MjvCamera()
    camera.distance = 2.58 if highest_command <= .4 else 2.9
    camera.elevation = -8
    options = mujoco.MjvOption()
    font = ImageFont.truetype(FONT,17)
    title = ImageFont.truetype(FONT,24)
    chart_x = np.linspace(45,VIDEO_WIDTH-25,len(poses))
    y0, y1 = 549, 496
    y_max = .45 if highest_command <= .4 else max(highest_command*1.1,float(speed.max()))
    def chart_y(values):
        return y0-(y0-y1)*np.clip(values,0.,y_max)/y_max
    cmd_y = chart_y(commands['requested'])
    actual_y = chart_y(speed)
    chart_layer = Image.new('RGB',(VIDEO_WIDTH,VIDEO_HEIGHT),BACKGROUND)
    base = ImageDraw.Draw(chart_layer)
    for y, label in ((chart_y(highest_command),f'{highest_command:g}'),(y0,'0')):
        base.line((45,y,VIDEO_WIDTH-25,y),fill='#364357',width=1)
        base.text((12,y-9),label,font=font,fill='#aab8c6')
    base.text((45,464),'Requested',font=font,fill=REQUEST)
    base.text((153,464),'Measured vx',font=font,fill=ACTUAL)
    base.text((287,464),'m/s',font=font,fill='#aab8c6')
    base.text((45,550),'0 s',font=font,fill='#aab8c6')
    base.text((VIDEO_WIDTH-64,550),f"{run['duration_s']:.0f} s",font=font,fill='#aab8c6')
    tmp = out.with_suffix('.part.mp4')
    args = ['ffmpeg','-nostdin','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',
            f'{VIDEO_WIDTH}x{VIDEO_HEIGHT}','-r','25','-i','-','-an','-c:v','libx264','-threads','3',
            '-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)]
    process = subprocess.Popen(args,stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(m,height=VIEW_HEIGHT,width=WIDTH) as renderer, process.stdin as pipe:
            for i in range(0,len(poses),2):
                d.qpos[:] = poses[i]
                mujoco.mj_forward(m,d)
                camera.lookat[:] = [d.qpos[0],d.qpos[1],.90 if highest_command <= .4 else .85]
                canvas = chart_layer.copy()
                draw = ImageDraw.Draw(canvas)
                for col,(label,azimuth) in enumerate((('Front',180.),('Side',90.))):
                    camera.azimuth = azimuth
                    renderer.update_scene(d,camera,options)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
                    x = col*WIDTH
                    canvas.paste(Image.fromarray(renderer.render()),(x,49))
                    draw.text((x+12,62),label,font=title,fill='white',stroke_width=2,stroke_fill=BACKGROUND)
                stage = LABEL[str(commands['stage'][i])]
                planned = commands['planned'][i]
                info = f'v7  /  {i*dt:05.2f} s  /  {stage}  |  Command {commands["requested"][i]:.2f} m/s'
                if stage=='Walk':
                    info += f'  |  Planned step {50*planned:.0f} cm'
                draw.text((12,12),info,font=title,fill=TEXT)
                load = 100*t[i,10:12]
                draw.text((VIDEO_WIDTH-392,464),f'Foot load L/R  {load[0]:.0f}% / {load[1]:.0f}% body weight',
                          font=font,fill=TEXT)
                if i:
                    for values,color in ((cmd_y,REQUEST),(actual_y,ACTUAL)):
                        draw.line(list(zip(chart_x[:i+1],values[:i+1])), fill=color,width=2)
                pipe.write(canvas.tobytes())
        if process.wait() != 0:
            raise RuntimeError('ffmpeg failed')
        tmp.replace(out)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
    print(out,flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folder',type=Path)
    p.add_argument('out',type=Path)
    args = p.parse_args()
    render(args.folder,args.out)
