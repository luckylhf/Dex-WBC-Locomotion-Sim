"""Interactive MuJoCo viewer for the two-hand box carry with keyboard teleoperation.

The closed-loop rollout of ``realtime.carry_empty`` and the six-direction planner of
``realtime.carry_omni`` are reused unchanged; only the fixed frame loop is replaced
by a paced loop that runs while the viewer window is open. Keys follow the same
bindings as the Isaac Lab play teleop and the MuJoCo sim2sim reference: vx, vy and
yaw ladders, stop, grip toggle, reset, save and quit.
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np

from gait import contract, dt

from .carry_empty import (CarryMeasurements, EMPTY_TARE_KG, Holding, LoadEstimate,
                          WristSensors, arm_pose, controller_for, mass_bin)
from .carry_model import ARM_ORDER, BOX_HALF, GRIP_OFFSETS, SIDES, WALL, BoxPlant, build_box
from .carry_load_transfer import BracedStanding, BracedCarryPreview, PayloadSupport, load_profile
from .carry_omni import CarryOmniPreview
from .omni_preview import yaw_rot
from .sim import CALIBRATION
from .viewer_common import (HORIZON, Teleop, ensure_models, grow_reference, run_headless,
                            run_viewer, script_frames, writable_reference)

OMNI_PROFILE = Path(__file__).with_name('carry_omni_calibration.json')


class Session:
    """One MuJoCo state that survives command changes, grip toggles and resets."""

    def __init__(self, args, model, pose):
        self.args = args
        self.model = model
        self.pose = pose
        self.out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix='dex-carry-viewer-'))
        self.out.mkdir(parents=True, exist_ok=True)
        self.preview, self.controller = self.plan()
        self.plant = BoxPlant(self.controller, model)
        self.data = self.plant.data
        self.sensors = WristSensors(model)
        self.teleop = Teleop(args.speed, args.log_keys)
        self.script = script_frames(args.script)
        self.deadline = time.monotonic()
        self.wall_start = time.monotonic()
        self.resets = 0
        self.overruns = 0
        self.total_mass = float(model.body_mass[self.plant.box_id])
        self.balance = None
        if args.brace:
            # Same waist lean, leg shift and arm damping as realtime.carry_load_transfer.
            # It writes model.actuator_biasprm once, so it is built here, not on reset.
            self.balance = BracedStanding(self.plant, self.sensors, self.preview,
                                          lean_gain=args.lean_gain, com_gain=args.com_gain,
                                          force_tau=args.force_tau,
                                          arm_damping_scale=args.arm_damping_scale,
                                          waist_damping_scale=args.waist_damping_scale)
        self.reset(initial=True)

    def plan(self):
        cfg = json.loads(CALIBRATION.read_text())
        cfg['anchors'][0]['recipe']['x_cop'] += self.args.cop_shift
        cfg['anchors'][0]['recipe']['stride_scale'] = self.args.stride_scale
        cfg['anchors'][0]['recipe']['heading_trim'] = self.args.heading_trim
        if self.args.brace:
            loaded = load_profile(Path(__file__).parent, True, True, None)
            profile = loaded if self.args.added_mass > 0 else self.content_profile(loaded)
            preview = BracedCarryPreview(cfg, self.pose, profile)
        else:
            preview = CarryOmniPreview(cfg, self.pose, json.loads(OMNI_PROFILE.read_text()))
        ctl = controller_for(preview, self.out, HORIZON)
        writable_reference(ctl)
        return preview, ctl

    @staticmethod
    def content_profile(loaded):
        """Calibration for items resting on the box floor instead of the wrist points.

        The loaded entries assume the payload split between the two grip sites, which sit
        0.157 m behind the box centre, so they under-compensate a centred load and over-
        stride it. Keeping the empty-box trim, the loaded lateral/yaw entries and 1.27x
        the forward stride tracks 0.1 m/s and stops cleanly.
        """
        profile = json.loads(OMNI_PROFILE.read_text())
        for section, values in loaded.items():
            if not isinstance(values, dict):
                profile[section] = values
                continue
            for label, entry in values.items():
                if label != 'Forward':
                    profile.setdefault(section, {})[label] = entry
        forward = profile.get('ranges', {}).get('Forward')
        if forward and 'stride_scale' in forward:
            forward['stride_scale'] = [value*1.27 for value in forward['stride_scale']]
        return profile

    def initial_state(self):
        p, d, m = self.plant, self.plant.data, self.plant.model
        ref = p.controller.reference
        mujoco.mj_resetData(m, d)
        d.qpos[p.qadr] = ref['q'][0]
        d.qpos[p.rq:p.rq+7] = np.r_[ref['root_p'][0], [1., 0., 0., 0.]]
        d.ctrl[p.aids] = d.qpos[p.qadr]
        mujoco.mj_forward(m, d)
        hand = np.array([d.site(f'carry/hand_{s}').xpos.copy() for s in SIDES])
        d.qpos[p.bq:p.bq+7] = np.r_[(hand - GRIP_OFFSETS).mean(0), [1., 0., 0., 0.]]
        mujoco.mj_forward(m, d)
        if self.args.seed:
            rng = np.random.default_rng(self.args.seed)
            d.qvel[p.rv:p.rv+2] = rng.uniform(-.018, .018, 2)
            d.qvel[p.bv:p.bv+2] = d.qvel[p.rv:p.rv+2]
            mujoco.mj_forward(m, d)

    def reset(self, initial=False):
        if not initial:
            if self.args.out and self.poses:
                self.save(self.out/f'segment_{self.resets+1}')
            self.preview, self.controller = self.plan()
            self.plant.controller = self.controller
            self.resets += 1
        self.initial_state()
        self.estimate = LoadEstimate(self.sensors.g)
        self.holder = Holding(self.plant, self.pose, self.args.gravity_gain, self.args.payload_gain)
        self.measures = CarryMeasurements(self.plant)
        self.data.eq_active[self.measures.grips] = True
        self.fixture = None
        if self.args.brace:
            self.balance.preview = self.preview
            self.balance.filtered = 0.
            self.balance.vertical = 0.
            self.balance.com_shift = 0.
            self.balance.target_pitch = 0.
            if self.args.transfer_start > 0:
                self.fixture = PayloadSupport(self.plant, self.args.content_mass + self.args.added_mass,
                                              self.args.transfer_start, self.args.transfer_ramp)
        self.balance_trace = []
        self.timeline, self.poses, self.rows, self.clearance = [], [], [], []
        self.wrist, self.adjusted, self.supports = [], [], []
        self.masses, self.errors, self.stages = [], [], []
        self.targets, self.effects = [], []
        self.grip_active, self.box_counts, self.bins, self.added = [], [], [], []
        self.command_log = []
        self.frame = 0
        self.deadline = time.monotonic()
        self.failed = False
        self.last = None
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

    def toggle_grips(self):
        grips = self.measures.grips
        self.data.eq_active[grips] = ~self.data.eq_active[grips]
        print(f'[SESSION] grips {"ON" if self.data.eq_active[grips[0]] else "OFF"}', flush=True)

    def poll_input(self):
        changed = self.teleop.update()
        if self.teleop.reset_requested:
            self.teleop.reset_requested = False
            self.reset()
            return
        if self.teleop.grip_requested:
            self.teleop.grip_requested = False
            self.toggle_grips()
        if self.teleop.estimate_requested:
            self.teleop.estimate_requested = False
            value = None if self.last is None else self.last[1]
            print(f'[ESTIMATE] feedforward {self.estimate.feedforward:.3f} kg | '
                  f'window {value if value is None else round(value, 3)} | '
                  f'bin {mass_bin(value)}', flush=True)
        if self.teleop.save_requested:
            self.teleop.save_requested = False
            self.save()
        if self.frame in self.script:
            self.teleop.command = self.script[self.frame]
            changed = True
        if changed:
            self.command(self.teleop.command)

    def tick(self):
        if self.frame + 2 >= self.controller.frames:
            grow_reference(self.controller)
        goal = self.preview.goal(self.frame)
        stage = self.preview.phase(self.frame).kind
        state, margin = self.measures.snapshot(self.frame)
        gpos, grot = self.measures.grip_errors()
        actual = self.sensors.read(self.data)
        good = bool(np.all(self.data.eq_active[self.measures.grips])
                    and np.max(gpos) < .015 and np.max(grot) < .15)
        box_contact = self.measures.contact_box()[0]
        raw, value, valid = self.estimate.update(actual[1][:, 0], stage, good, box_contact)
        self.last = (raw, value, valid)
        if self.balance is not None:
            self.preview.observe_load(self.estimate.feedforward)
        self.timeline.append(self.data.time)
        self.poses.append(self.data.qpos.copy())
        self.rows.append(state)
        self.clearance.append(margin)
        self.wrist.append(actual[0])
        self.adjusted.append(actual[1])
        self.supports.append(self.measures.support())
        self.masses.append([raw, np.nan if value is None else value, float(valid), self.estimate.feedforward])
        self.bins.append(mass_bin(value))
        self.added.append(np.nan if value is None else value - EMPTY_TARE_KG)
        self.errors.append(np.c_[gpos, grot])
        self.grip_active.append(self.data.eq_active[self.measures.grips].copy())
        self.box_counts.append(self.measures.box_contact_steps)
        self.stages.append(stage)
        self.controller.reference['goal'][self.frame] = goal
        self.controller.reference['root_R'][self.frame] = yaw_rot(self.preview.angles[self.frame])
        self.heading = float(self.preview.angles[self.frame])
        target = self.controller.step(*self.plant.sensors())
        if self.balance is None:
            target[ARM_ORDER], effect = self.holder.targets(self.estimate.feedforward)
        else:
            arm_mass = self.balance.arm_mass(self.estimate.feedforward)
            target[ARM_ORDER], effect = self.balance.arm_targets(self.holder, arm_mass)
            attached = bool(np.all(self.data.eq_active[self.measures.grips])) and not box_contact
            self.balance_trace.append(self.balance.step(
                target, actual[1], attached and self.balance.grip_engaged(gpos, grot)))
        self.effects.append(effect)
        self.controller.previous_action = np.clip((target - self.controller.q0) / self.controller.alpha,
                                                  -contract['action_clip'], contract['action_clip'])
        self.targets.append(target.copy())
        try:
            self.measures.step(target, self.fixture)
        except FloatingPointError:
            self.failed = True
            print('[SESSION] non-finite state', flush=True)
            return
        self.frame += 1
        if self.measures.body_steps or self.measures.box_ground_steps:
            self.failed = True
            print('[SESSION] fall or box-ground contact, press R', flush=True)

    def overlay(self):
        value = None if self.last is None else self.last[1]
        estimate = 'estimating' if value is None else f'{value:.3f} kg'
        stage = self.stages[-1] if self.stages else 'stand'
        grips = 'ON' if np.all(self.data.eq_active[self.measures.grips]) else 'OFF'
        vx, vy, wz = self.teleop.command
        lines = [
            'Two-hand box carry | keyboard teleop',
            'W/S or UP/DOWN  vx      A/D or LEFT/RIGHT  vy',
            'Q/E  yaw                SPACE or C or 0  stop',
            '1-4  vx preset 0.10-0.40 m/s',
            'G grips   R reset   P estimate   F save   ESC quit',
            f'cmd vx {vx:+.2f} m/s  vy {vy:+.2f} m/s  wz {wz:+.2f} rad/s',
            f'stage {stage}   grips {grips}   heading {np.rad2deg(self.heading):+.0f} deg',
            f'box {self.total_mass:.2f} kg [shell {self.args.mass:.2f} + content {self.args.content_mass:.2f}]',
            f'estimate {estimate}   ff {self.estimate.feedforward:.3f} kg',
            f'contact {self.measures.box_contact_steps}   sim {self.data.time:.1f} s   frame {self.frame}',
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
        np.savez_compressed(out/'rollout.npz', qpos=np.asarray(self.poses),
                            physics_time=np.asarray(self.timeline),
                            wrist_local=np.asarray(self.wrist),
                            wrist_compensated_world=np.asarray(self.adjusted),
                            grip_wrench_on_box=np.asarray(self.supports),
                            estimates=np.asarray(self.masses), grip_errors=np.asarray(self.errors),
                            stage=np.asarray(self.stages), actuator_targets=np.asarray(self.targets),
                            load_compensation_joint_delta_rad=np.asarray(self.effects),
                            grip_active=np.asarray(self.grip_active),
                            box_contact_counts=np.asarray(self.box_counts),
                            estimated_mass_bin=np.asarray(self.bins),
                            estimated_added_mass_kg=np.asarray(self.added),
                            balance_trace=np.asarray(self.balance_trace))
        meta = dict(interactive=True, completed=not self.failed,
                    frames=len(self.poses), physics_steps=self.frame*contract['decimation'],
                    physics_resets=self.resets, duration_s=float(self.data.time),
                    wall_time_s=time.monotonic()-self.wall_start,
                    go_s=self.go_s, zero_s=self.zero_s,
                    empty_box_enclosure_dimensions_m=(2*BOX_HALF).tolist(),
                    empty_box_wall_thickness_m=2*WALL,
                    model_box_mass_kg=float(self.model.body_mass[self.plant.box_id]),
                    empty_shell_mass_kg=self.args.mass, added_mass_kg=self.args.added_mass,
                    added_mass_locations_m=GRIP_OFFSETS.tolist() if self.args.added_mass > 0 else [],
                    content_mass_kg=self.args.content_mass,
                    content_center_m=(self.model.geom('carry/content').pos.tolist()
                                      if self.args.content_mass > 0 else []),
                    added_mass_per_grip_kg=.5*self.args.added_mass,
                    estimator_empty_tare_kg=EMPTY_TARE_KG,
                    box_local_com_m=self.model.body_ipos[self.plant.box_id].tolist(),
                    distal_masses_kg=self.sensors.distal_mass.tolist(),
                    policy_sha256=contract['policy_sha256'], box_free_qadr=self.plant.bq,
                    body_ground_steps=self.measures.body_steps,
                    box_contact_steps=self.measures.box_contact_steps,
                    box_ground_steps=self.measures.box_ground_steps,
                    foot_rub_steps=self.measures.rub_steps,
                    grip_over_20mm_frames=self.measures.grip_overlimit_steps,
                    over_20ms_ticks=self.overruns,
                    planned_blocks=[dict(kind=b.kind, start_s=b.start*dt, end_s=b.end*dt,
                                         side='LR'[b.side]) for b in self.preview.blocks
                                    if b.start <= self.frame],
                    commands=self.command_log,
                    options={k: str(v) if isinstance(v, Path) else v for k, v in vars(self.args).items()
                             if k != 'out'})
        (out/'run.json').write_text(json.dumps(meta, indent=2) + '\n')
        model_file = out/'box_model.mjb'
        if self.args.out and not model_file.exists():
            mujoco.mj_saveModel(self.model, str(model_file))
        print(f'[SESSION] saved {len(self.poses)} frames to {out} | completed {meta["completed"]}', flush=True)


def argument_parser(description=__doc__):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--out', type=Path, help='Record folder; omit to keep artefacts in a temp dir')
    p.add_argument('--mass', type=float, default=.65)
    p.add_argument('--added-mass', type=float, default=0.,
                   help='Mass added at both grip points on the box, split evenly')
    p.add_argument('--content-mass', type=float, default=0.,
                   help='Items inside the box, lumped on the inner bottom')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--reach', type=float, default=None)
    p.add_argument('--hand-height', type=float, default=-.02)
    p.add_argument('--grip-softness', type=float, default=None)
    p.add_argument('--brace', action='store_true',
                   help='Waist lean, leg shift and arm damping from realtime.carry_load_transfer')
    p.add_argument('--no-brace', action='store_true', help='Keep the plain hold even with contents')
    p.add_argument('--transfer-start', type=float, default=5.,
                   help='Release the external payload support at this simulated time; 0 keeps it')
    p.add_argument('--transfer-ramp', type=float, default=.24)
    p.add_argument('--lean-gain', type=float, default=None)
    p.add_argument('--com-gain', type=float, default=None)
    p.add_argument('--force-tau', type=float, default=.20)
    p.add_argument('--arm-damping-scale', type=float, default=5.)
    p.add_argument('--waist-damping-scale', type=float, default=7.)
    p.add_argument('--gravity-gain', type=float, default=1.)
    p.add_argument('--payload-gain', type=float, default=1.)
    p.add_argument('--cop-shift', type=float, default=0.)
    p.add_argument('--stride-scale', type=float, default=1.42)
    p.add_argument('--heading-trim', type=float, default=-.31)
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
    if not .1 <= args.mass < 1.:
        p.error('Empty-box mass must lie in [0.1,1.0) kg')
    if args.added_mass < 0.:
        p.error('Added mass must be non-negative')
    if args.content_mass < 0.:
        p.error('Content mass must be non-negative')
    if args.brace and args.no_brace:
        p.error('--brace and --no-brace are mutually exclusive')
    if args.content_mass > 0. and not args.no_brace:
        args.brace = True
    if args.brace and (args.transfer_start < 0. or args.transfer_ramp < dt):
        p.error('Transfer start must be non-negative and the ramp at least one control step')
    if args.reach is None:
        args.reach = .29 if args.brace else .37
    if args.grip_softness is None:
        args.grip_softness = .04 if args.brace else .06
    if args.lean_gain is None:
        args.lean_gain = .30 if args.content_mass > 0. else .20
    if args.com_gain is None:
        args.com_gain = 1.25 if args.content_mass > 0. else 1.
    if not np.isfinite(args.overlay_hz) or args.overlay_hz <= 0.:
        p.error('Overlay refresh rate must be positive')
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration < 1.):
        p.error('Duration must be a finite number of seconds >= 1')
    ensure_models(('model', 'empty', 'one_kg', 'braced_ten_kg'))
    pose, orientation = arm_pose(contract['default_q'], args.reach, args.hand_height)
    model = build_box(args.mass, orientation, args.grip_softness,
                      added_mass=args.added_mass, content_mass=args.content_mass)
    session = Session(args, model, pose)
    if args.headless:
        run_headless(session, args)
    else:
        run_viewer(session, args)


if __name__ == '__main__':
    main()
