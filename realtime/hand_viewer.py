"""Interactive MuJoCo viewer for the bare robot (no box) with keyboard teleoperation.

Same 50 Hz control clock, six-direction planner and teleop keys as
``realtime.carry_viewer``, but the plant is the released frozen robot without the
box, the grips and the wrist load estimate. Artifacts follow ``realtime.omni``.
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np

from gait import contract, dt

from .omni import FootMeasurements
from .omni_preview import OmniPreview, yaw_rot
from .sim import CALIBRATION, initialize
from .viewer_common import (HORIZON, Teleop, ensure_models, grow_reference, run_headless,
                            run_viewer, script_frames, writable_reference)

OMNI_OPTIONS = Path(__file__).with_name('omni_calibration.json')


class Session:
    """One MuJoCo state that survives command changes and resets."""

    def __init__(self, args):
        self.args = args
        self.out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix='dex-hand-viewer-'))
        self.out.mkdir(parents=True, exist_ok=True)
        self.preview, self.controller, self.plant = self.plan()
        self.model = self.plant.model
        self.data = self.plant.data
        self.teleop = Teleop(args.speed, args.log_keys)
        self.script = script_frames(args.script)
        self.deadline = time.monotonic()
        self.wall_start = time.monotonic()
        self.resets = 0
        self.overruns = 0
        self.reset(initial=True)

    def plan(self):
        preview = OmniPreview(json.loads(CALIBRATION.read_text()),
                              json.loads(OMNI_OPTIONS.read_text()))
        ctl, plant = initialize(preview, self.out, HORIZON)
        writable_reference(ctl)
        return preview, ctl, plant

    def initial_state(self):
        p, d, m = self.plant, self.plant.data, self.plant.model
        ref = p.controller.reference
        joints = [m.joint('robot/'+name).id for name in contract['joint_names']]
        mujoco.mj_resetData(m, d)
        d.qpos[p.qadr] = np.clip(ref['q'][0], m.jnt_range[joints, 0], m.jnt_range[joints, 1])
        d.qvel[p.vadr] = ref['qd'][0]
        d.qpos[p.rq:p.rq+3] = ref['root_p'][0]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, ref['root_R'][0].astype(np.float64).reshape(9))
        d.qpos[p.rq+3:p.rq+7] = quat
        d.qvel[p.rv:p.rv+3] = ref['root_v'][0]
        d.qvel[p.rv+3:p.rv+6] = ref['root_R'][0].T @ ref['root_w'][0]
        d.ctrl[p.aids] = d.qpos[p.qadr]
        d.xfrc_applied[:] = 0
        d.qfrc_applied[:] = 0
        mujoco.mj_forward(m, d)

    def reset(self, initial=False):
        if not initial:
            if self.args.out and self.poses:
                self.save(self.out/f'segment_{self.resets+1}')
            self.preview, self.controller, _ = self.plan()
            self.plant.controller = self.controller
            self.resets += 1
        self.initial_state()
        self.measures = FootMeasurements(self.plant)
        self.timeline, self.poses, self.rows, self.clearance = [], [], [], []
        self.stages, self.requested, self.targets = [], [], []
        self.command_log = []
        self.frame = 0
        self.deadline = time.monotonic()
        self.failed = False
        self.heading = 0.
        self.go_s = self.zero_s = None
        if any(self.teleop.command):
            self.command(self.teleop.command)
        print(f'[SESSION] {"reset" if not initial else "start"} | out {self.out}', flush=True)

    def command(self, command):
        command = tuple(float(value) for value in command)
        if command == self.preview.intent:
            return
        self.preview.command(self.frame, command)
        now = self.data.time
        moving = any(command)
        if moving and self.go_s is None:
            self.go_s = now
        if not moving and self.go_s is not None and self.zero_s is None:
            self.zero_s = now
        self.command_log.append(dict(t_s=now, frame=self.frame, command=list(command)))

    def poll_input(self):
        changed = self.teleop.update()
        if self.teleop.reset_requested:
            self.teleop.reset_requested = False
            self.reset()
            return
        if self.teleop.grip_requested:
            self.teleop.grip_requested = False
            print('[SESSION] no box in this viewer', flush=True)
        if self.teleop.estimate_requested:
            self.teleop.estimate_requested = False
            pitch, roll = self.attitude()
            print(f'[STATE] pelvis z {self.data.qpos[self.plant.rq+2]:.3f} m | '
                  f'pitch {pitch:+.1f} deg roll {roll:+.1f} deg | '
                  f'ground contacts {self.measures.body_steps}', flush=True)
        if self.teleop.save_requested:
            self.teleop.save_requested = False
            self.save()
        if self.frame in self.script:
            self.teleop.command = self.script[self.frame]
            changed = True
        if changed:
            self.command(self.teleop.command)

    def attitude(self):
        chest = self.model.body('robot/waist_pitch_link').id
        R = self.data.xmat[chest].reshape(3, 3)
        return (float(np.rad2deg(np.arcsin(np.clip(-R[2, 0], -1, 1)))),
                float(np.rad2deg(np.arctan2(R[2, 1], R[2, 2]))))

    def tick(self):
        if self.frame + 2 >= self.controller.frames:
            grow_reference(self.controller)
        goal = self.preview.goal(self.frame)
        row, margin = self.measures.snapshot(self.frame)
        block = self.preview.phase(self.frame)
        self.timeline.append(self.data.time)
        self.poses.append(self.data.qpos.copy())
        self.rows.append(row)
        self.clearance.append(margin)
        self.stages.append(block.kind)
        self.requested.append(self.preview.intent)
        self.controller.reference['goal'][self.frame] = goal
        self.controller.reference['root_R'][self.frame] = yaw_rot(self.preview.angles[self.frame])
        self.heading = float(self.preview.angles[self.frame])
        target = self.controller.step(*self.plant.sensors())
        self.targets.append(target.copy())
        try:
            self.measures.step(target)
        except FloatingPointError:
            self.failed = True
            print('[SESSION] non-finite state, press R', flush=True)
            return
        self.frame += 1
        if self.measures.body_steps:
            self.failed = True
            print('[SESSION] fall, press R', flush=True)

    def overlay(self):
        vx, vy, wz = self.teleop.command
        stage = self.stages[-1] if self.stages else 'stand'
        pitch, roll = self.attitude()
        lines = [
            'Bare robot walk | keyboard teleop',
            'W/S or UP/DOWN  vx      A/D or LEFT/RIGHT  vy',
            'Q/E  yaw                SPACE or C or 0  stop',
            '1-4  vx preset 0.10-0.40 m/s',
            'R reset   P state   F save   ESC quit',
            f'cmd vx {vx:+.2f} m/s  vy {vy:+.2f} m/s  wz {wz:+.2f} rad/s',
            f'stage {stage}   heading {np.rad2deg(self.heading):+.0f} deg',
            f'pelvis {self.data.qpos[self.plant.rq+2]:.3f} m   pitch {pitch:+.1f}  roll {roll:+.1f} deg',
            f'ground contacts {self.measures.body_steps}   sim {self.data.time:.1f} s   frame {self.frame}',
        ]
        if self.failed:
            lines.append('FAILED - press R to reset')
        return '\n'.join(lines)

    def save(self, directory=None):
        if not self.poses:
            return
        out = Path(directory) if directory is not None else self.out
        out.mkdir(parents=True, exist_ok=True)
        np.save(out/'telemetry.npy', np.asarray(self.rows))
        np.save(out/'clearance.npy', np.asarray(self.clearance))
        np.savez_compressed(out/'rollout.trajectory.npz', qpos=np.asarray(self.poses), control_dt=dt,
                            physics_time=np.asarray(self.timeline), joint_targets=np.asarray(self.targets),
                            policy_sha256=contract['policy_sha256'], model_sha256=contract['model_sha256'])
        np.savez_compressed(out/'commands.npz', requested=np.asarray(self.requested),
                            stage=np.asarray(self.stages),
                            reference_root=np.asarray(self.preview.positions[:len(self.poses)])-self.preview.offset,
                            ref_yaw=np.asarray(self.preview.angles[:len(self.poses)]))
        meta = dict(interactive=True, completed=not self.failed, frames=len(self.poses),
                    physics_steps=self.frame*contract['decimation'], physics_resets=self.resets,
                    duration_s=float(self.data.time), wall_time_s=time.monotonic()-self.wall_start,
                    go_s=self.go_s, zero_s=self.zero_s, over_20ms_ticks=self.overruns,
                    rub_substeps=self.measures.rub_steps,
                    nonsupport_ground_substeps=self.measures.body_steps,
                    max_ik_error_m=float(self.preview.ik.max_error), revisions=self.preview.revisions,
                    commands=self.command_log, options=self.preview.options,
                    blocks=[dict(kind=b.kind, side='LR'[b.side], start_s=b.start*dt, end_s=b.end*dt,
                                 cmd=list(b.cmd)) for b in self.preview.blocks if b.start <= self.frame],
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(self.args).items()
                          if k != 'out'})
        (out/'run.json').write_text(json.dumps(meta, indent=2) + '\n')
        print(f'[SESSION] saved {len(self.poses)} frames to {out} | completed {meta["completed"]}', flush=True)


def argument_parser(description=__doc__):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--out', type=Path, help='Record folder; omit to keep artefacts in a temp dir')
    p.add_argument('--speed', type=float, default=0., help='Initial forward speed in m/s')
    p.add_argument('--script', default='',
                   help='Timed commands for hands-free runs: "t:vx" or "t:vx/vy/wz", e.g. "6:0.1,14:0,20:0/0.2"')
    p.add_argument('--duration', type=float, default=None, help='Stop after this simulated time in seconds')
    p.add_argument('--render-hz', type=float, default=30.,
                   help='Viewer refresh rate; below the 50 Hz control clock to keep real time')
    p.add_argument('--overlay-hz', type=float, default=2.,
                   help='Overlay text refresh rate; lower values keep the control clock closer to real time')
    p.add_argument('--log-keys', action='store_true', help='Print every GLFW key code the window delivers')
    p.add_argument('--headless', action='store_true', help='Run without the viewer window')
    return p


def main():
    p = argument_parser()
    args = p.parse_args()
    if not np.isfinite(args.overlay_hz) or args.overlay_hz <= 0.:
        p.error('Overlay refresh rate must be positive')
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration < 1.):
        p.error('Duration must be a finite number of seconds >= 1')
    ensure_models(('model',))
    session = Session(args)
    if args.headless:
        run_headless(session, args)
    else:
        run_viewer(session, args)


if __name__ == '__main__':
    main()
