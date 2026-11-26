"""Shared keyboard teleop, 50 Hz pacing loop and reference growth for the MuJoCo viewers.

Both ``realtime.carry_viewer`` (with the box) and ``realtime.hand_viewer`` (bare robot)
drive the same closed loop; only the plant, the planner class and the recorded channels
differ. This module keeps the input handling, the wall-clock pacing and the viewer
plumbing in one place so the two entries cannot drift apart.
"""
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from gait import contract, dt

HORIZON = 250            # reference frames reserved per grow(); 5 s at 50 Hz
OVERLAY_FONT = mujoco.mjtFontScale.mjFONTSCALE_100
OVERLAY_POS = mujoco.mjtGridPos.mjGRID_TOPLEFT
ROT = np.eye(3, dtype=np.float32)


class Teleop:
    """Keyboard planar command, mirroring the Isaac Lab play and MuJoCo sim2sim bindings."""

    # Rungs are the calibrated speeds of realtime.omni, plus the 0.05 fine steps that
    # the calibration profiles interpolate between.
    VX = tuple(sorted([0.] + [round(.1 + .05*i, 2) for i in range(7)]
                      + [-round(.1 + .05*i, 2) for i in range(5)]))
    VY = tuple(round(.05*i, 2) for i in range(-6, 7))
    WZ = (-.45, -.3, -.15, 0., .15, .3, .45)
    LADDERS = (VX, VY, WZ)
    PRESETS = (.1, .2, .3, .4)

    def __init__(self, speed=0., verbose=False):
        # mujoco.glfw is only imported by the default GLFW context; importing it here
        # keeps the constants available and avoids a headless dependency.
        import mujoco.glfw
        self.glfw = mujoco.glfw.glfw
        self.keys = deque(maxlen=64)
        self.verbose = verbose
        self.command = (self.nearest(0, speed), 0., 0.)
        self.reset_requested = False
        self.grip_requested = False
        self.estimate_requested = False
        self.save_requested = False
        self.quit_requested = False

    @classmethod
    def nearest(cls, axis, value):
        ladder = np.asarray(cls.LADDERS[axis])
        return float(ladder[int(np.argmin(np.abs(ladder - float(value))))])

    def bump(self, axis, sign):
        """Move one rung along vx, vy or yaw; the ends of each ladder stop the axis."""
        ladder = np.asarray(self.LADDERS[axis])
        index = int(np.argmin(np.abs(ladder - self.command[axis])))
        index = min(max(index + sign, 0), len(ladder) - 1)
        command = list(self.command)
        command[axis] = float(ladder[index])
        self.command = tuple(command)

    def key_callback(self, key):
        """Called from the viewer thread for every key pressed in the sim window."""
        if self.verbose:
            print(f'[KEY] {key}', flush=True)
        self.keys.append(key)

    def update(self):
        """Apply the keys seen since the last control tick; returns True on a command change."""
        glfw = self.glfw
        before = self.command
        while self.keys:
            key = self.keys.popleft()
            if key in (glfw.KEY_W, glfw.KEY_UP):
                self.bump(0, 1)
            elif key in (glfw.KEY_S, glfw.KEY_DOWN):
                self.bump(0, -1)
            elif key in (glfw.KEY_A, glfw.KEY_LEFT):
                self.bump(1, 1)
            elif key in (glfw.KEY_D, glfw.KEY_RIGHT):
                self.bump(1, -1)
            elif key == glfw.KEY_Q:
                self.bump(2, 1)
            elif key == glfw.KEY_E:
                self.bump(2, -1)
            elif key in (glfw.KEY_SPACE, glfw.KEY_C, glfw.KEY_0):
                self.command = (0., 0., 0.)
            elif key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4):
                command = list(self.command)
                command[0] = self.PRESETS[key - glfw.KEY_1]
                self.command = tuple(command)
            elif key == glfw.KEY_G:
                self.grip_requested = True
            elif key == glfw.KEY_R:
                self.reset_requested = True
            elif key == glfw.KEY_P:
                self.estimate_requested = True
            elif key == glfw.KEY_F:
                self.save_requested = True
            elif key == glfw.KEY_ESCAPE:
                self.quit_requested = True
        if self.command != before:
            print(f'[TELEOP] command vx={self.command[0]:+.2f} '
                  f'vy={self.command[1]:+.2f} wz={self.command[2]:+.2f}', flush=True)
        return self.command != before


def ensure_models(names=("model",)):
    """Compile or verify the frozen MuJoCo caches before a viewer starts."""
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root/'tools'/'compile_models.py'), *names],
                   check=True)


def script_frames(text):
    """Parse '6:0.1,14:0,20:0.1/0.2,26:0/0/0.3' into {frame: (vx, vy, wz)}."""
    frames = {}
    for item in filter(None, (part.strip() for part in (text or '').split(','))):
        when, _, values = item.partition(':')
        command = tuple(float(v) for v in values.split('/') if v != '')
        command = tuple(command + (0.,)*(3-len(command)))
        if len(command) != 3 or not all(np.isfinite(v) for v in command):
            raise ValueError(f'Script command needs 1-3 numbers: {item}')
        if not -.4 <= command[0] <= .4 or not -.3 <= command[1] <= .3 or not -.45 <= command[2] <= .45:
            raise ValueError(f'Script command out of range: {item}')
        frames[round(float(when)/dt)] = command
    return frames


def writable_reference(controller):
    """The initial reference broadcasts one rotation matrix; the viewers write per frame."""
    controller.reference['root_R'] = np.array(controller.reference['root_R'], dtype=np.float32)


def grow_reference(controller, extra=HORIZON):
    """Extend the fixed reference horizon; controller.step stops at frames-1 otherwise."""
    controller.frames += extra
    controller.reference['goal'] = np.concatenate([controller.reference['goal'],
                                                   np.zeros((extra, contract['goal_dim']), np.float32)])
    controller.reference['root_R'] = np.concatenate([controller.reference['root_R'],
                                                     np.broadcast_to(ROT, (extra, 3, 3)).astype(np.float32)])


def run_viewer(session, args):
    """Paced 50 Hz loop with the passive viewer; returns when the window closes."""
    import mujoco.viewer  # only needed on the GUI path
    viewer = mujoco.viewer.launch_passive(session.model, session.data,
                                          key_callback=session.teleop.key_callback)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = session.model.body('robot/pelvis').id
    viewer.cam.distance = 2.6
    viewer.cam.elevation = -12
    viewer.cam.azimuth = 135
    text = None
    period = 1./args.render_hz
    text_period = 1./args.overlay_hz
    last_sync = 0.
    last_text = 0.
    session.wall_start = time.monotonic()
    print('[VIEWER] window open; W/S vx, A/D vy, Q/E yaw, SPACE stop, R reset, ESC quit', flush=True)
    try:
        while viewer.is_running():
            if args.duration and session.data.time >= args.duration:
                break
            now = time.monotonic()
            if session.deadline > now:
                time.sleep(session.deadline - now)
            session.deadline = max(session.deadline + dt, time.monotonic())
            started = time.monotonic()
            session.poll_input()
            if session.teleop.quit_requested:
                break
            if not session.failed:
                session.tick()
            # A full scene copy costs more than a control period on this model, so the
            # render rate is throttled below the 50 Hz control clock to stay real time.
            if started - last_sync >= period:
                viewer.sync()
                last_sync = time.monotonic()
            if time.monotonic() - started > dt:
                session.overruns += 1
            # set_texts blocks until the viewer's render loop consumes the text
            # request. In our local macOS run it averaged about 100 ms per call,
            # so do not refresh the overlay at the 50 Hz control rate. The state
            # and camera still render at render_hz.
            if time.monotonic() - last_text >= text_period:
                overlay = session.overlay()
                if overlay != text:
                    viewer.set_texts([(OVERLAY_FONT, OVERLAY_POS, overlay, '')])
                    text = overlay
                last_text = time.monotonic()
    finally:
        viewer.close()
        session.save()


def run_headless(session, args):
    ticks = round(args.duration/dt) if args.duration else round(21./dt)
    session.wall_start = time.monotonic()
    while session.frame < ticks and not session.failed:
        session.poll_input()
        session.tick()
    session.save()
