"""Stand, transfer wrist-end payload, then hold or walk and park."""
import json
from pathlib import Path

import mujoco
import numpy as np

from gait import dt
from .carry_empty import argument_parser, run
from .carry_model import ARM_ORDER, SIDES
from .omni_preview import yaw_rot
from .carry_omni import CarryOmniPreview, make_trials
from .carry_payload import CarryPayloadPreview
from .wrench_lean import WrenchLean


class PayloadSupport:
    """Test fixture. Only the physics side receives the applied load and its schedule."""

    def __init__(self, plant, added_mass, start, ramp):
        self.m, self.d, self.box = plant.model, plant.data, plant.box_id
        self.grips = [self.m.site(f'carry/box_{side}').id for side in SIDES]
        self.force = -.5 * added_mass * self.m.opt.gravity
        self.zero = np.zeros(3)
        self.start, self.end = start, start + ramp
        self.ramp = ramp

    def apply(self):
        d = self.d
        d.qfrc_applied.fill(0.)
        if d.time >= self.end:
            return
        phase = np.clip((d.time - self.start) / self.ramp, 0., 1.)
        support = self.force * (1. - phase**2 * (3. - 2. * phase))
        for grip in self.grips:
            mujoco.mj_applyFT(self.m, d, support, self.zero, d.site_xpos[grip],
                              self.box, d.qfrc_applied)

    def metadata(self):
        return dict(start_s=self.start, fully_transferred_s=self.end,
                    initial_upward_force_per_grip_n=float(self.force[2]),
                    method='Smooth withdrawal of separate vertical support at both added-mass centres')


class BracedStanding(WrenchLean):
    """Use measured support for arm torque and the same wrench for waist and leg balance."""

    def __init__(self, plant, sensors, preview, *, lean_gain, com_gain, force_tau,
                 arm_damping_scale, waist_damping_scale):
        super().__init__(plant, sensors, lean_gain, 16., preview, com_gain, True)
        self.waist_neutral = self.d.qpos[self.qadr]
        m = plant.model
        arm_actuators = plant.aids[ARM_ORDER]
        self.arm_force_limits = np.moveaxis(m.actuator_forcerange[arm_actuators], -1, 0)
        self.arm_control_limits = np.moveaxis(m.actuator_ctrlrange[arm_actuators], -1, 0)
        # Retain posture stiffness and force limits. The servo applies the higher
        # velocity damping on each physics substep while holding the load.
        m.actuator_biasprm[arm_actuators, 2] *= arm_damping_scale
        m.actuator_biasprm[plant.aids[self.index], 2] *= waist_damping_scale
        self.smoothing = 1. - np.exp(-dt / force_tau)

    def grip_engaged(self, gpos, grot):
        # The wide-angle transient in the grip does not erase a valid load signal
        # while both welds remain engaged and neither grip has pulled away.
        return np.max(gpos) < .025

    def waist_target(self, measured):
        return self.waist_neutral + self.target_pitch - self.filtered / self.kp

    def arm_targets(self, holder, mass):
        m, data = holder.m, holder.data
        data.qpos[:] = holder.d.qpos
        mujoco.mj_fwdPosition(m, data)
        mujoco.mj_rne(m, data, 0, holder.bias)
        gravity = holder.bias[holder.va]
        support = np.empty_like(gravity)
        for side, site in enumerate(holder.grip):
            mujoco.mj_jacSite(m, data, holder.jac, None, site)
            support[side] = holder.jac[2, holder.va[side]] * .5 * self.g * mass
        base = holder.arm + np.clip(gravity, *self.arm_force_limits) / holder.kp
        loaded = holder.arm + np.clip(gravity + support, *self.arm_force_limits) / holder.kp
        actual = np.clip(loaded, *self.arm_control_limits)
        return actual, actual - np.clip(base, *self.arm_control_limits)


class BracedCarryPreview(CarryPayloadPreview):
    def park_center(self, feet, yaw, cfg, cmd):
        center = super().park_center(feet, yaw, cfg, cmd)
        offsets = self.values(cmd, 'park_center')
        return center + yaw_rot(yaw)[:2, :2] @ [offsets.get('x', 0.), offsets.get('y', 0.)]

    def park_trim(self, cmd, side, cfg):
        # Use the calibrated parking profile. The legacy trim extrapolated a
        # linear correction from 1 kg, which overshot the gather at high loads.
        return CarryOmniPreview.park_trim(self, cmd, side, cfg)


def walking_ladder(first, omni=False):
    trials = make_trials()
    selected = [trial for trial in trials if trial['label'] != 'Backward'] if omni else trials[:4]
    output = []
    for trial in selected:
        delay = first - trial['start_s']
        output.append(trial | {key: trial[key] + delay for key in ('start_s', 'zero_s', 'end_s')})
        first = output[-1]['end_s']
    return output


def load_profile(directory, omni, walking, override):
    if not omni:
        default = 'carry_loaded_forward_calibration.json' if walking else 'carry_omni_calibration.json'
        return json.loads((override or directory / default).read_text())
    profile = json.loads((directory / 'carry_omni_calibration.json').read_text())
    forward = json.loads((directory / 'carry_loaded_forward_calibration.json').read_text())
    paths = [directory / 'carry_loaded_omni_calibration.json']
    if override is not None:
        paths.append(override)
    for config in [forward, *(json.loads(path.read_text()) for path in paths)]:
        for section, values in config.items():
            profile.setdefault(section, {}).update(values)
    return profile


def main():
    parser = argument_parser(__doc__)
    parser.set_defaults(stand_only=True, stand_seconds=20., grip_softness=.04, reach=.29)
    parser.add_argument('--added-mass', type=float, default=10.)
    parser.add_argument('--transfer-start', type=float, default=5.)
    parser.add_argument('--transfer-ramp', type=float, default=.24)
    parser.add_argument('--lean-gain', type=float, default=.20)
    parser.add_argument('--com-gain', type=float, default=1.)
    parser.add_argument('--force-tau', type=float, default=.20)
    parser.add_argument('--arm-damping-scale', type=float, default=5.)
    parser.add_argument('--waist-damping-scale', type=float, default=7.)
    parser.add_argument('--reference-control', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--walk', action='store_true', help='Run four forward start/stop commands after transfer')
    mode.add_argument('--omni', action='store_true', help='Include lateral motion and turns after the forward trials')
    parser.add_argument('--walk-start', type=float, default=10.)
    parser.add_argument('--profile', type=Path)
    args = parser.parse_args()
    options = (args.added_mass, args.transfer_start, args.transfer_ramp, args.stand_seconds,
               args.force_tau, args.arm_damping_scale, args.waist_damping_scale,
               args.lean_gain, args.com_gain, args.grip_softness, args.walk_start)
    if (not np.all(np.isfinite(options)) or min(options) < 0
            or args.transfer_start < 2 or args.transfer_ramp < dt or args.force_tau < dt
            or args.stand_seconds < args.transfer_start + args.transfer_ramp + 5
            or not .1 <= args.mass < 1.):
        parser.error('Provide finite settings, an empty shell in [0.1, 1.0) kg, and at least '
                     '2 s before and 5 s after the transfer')
    walking = args.walk or args.omni
    if walking and args.walk_start < args.transfer_start + args.transfer_ramp + 2.:
        parser.error('Allow at least 2 s for the load to settle before the first walking command')
    args.stand_only = not walking
    profile = load_profile(Path(__file__).parent, args.omni, walking, args.profile)
    trials = walking_ladder(args.walk_start, args.omni) if walking else []
    preview_type = BracedCarryPreview if walking else CarryPayloadPreview

    def balance(plant, sensors, preview):
        if args.reference_control:
            return WrenchLean(plant, sensors, .26, 16., preview, 1., False)
        return BracedStanding(plant, sensors, preview, lean_gain=args.lean_gain, com_gain=args.com_gain,
                              force_tau=args.force_tau, arm_damping_scale=args.arm_damping_scale,
                              waist_damping_scale=args.waist_damping_scale)

    run(args, trials=trials, build_preview=lambda c, a: preview_type(c, a, profile),
        added_mass=args.added_mass, balance_factory=balance,
        fixture_factory=lambda p, mass: PayloadSupport(p, mass, args.transfer_start, args.transfer_ramp))


if __name__ == '__main__':
    main()
