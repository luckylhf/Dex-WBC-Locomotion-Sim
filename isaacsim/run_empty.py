"""Run the frozen v7 ONNX on live 29-axis Isaac Sim / PhysX feedback."""
import argparse
import json
from itertools import count
import math
import os
import signal
import sys
from pathlib import Path
import time
import traceback
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--seconds", type=float)
parser.add_argument("--viewer", action="store_true", help="Watch the live PhysX run in the Isaac Sim viewport")
parser.add_argument("--carry", action="store_true", help="Run the 16-slot wrist-load transfer protocol")
parser.add_argument("--grip-k", type=float, default=10500.)
parser.add_argument("--grip-d", type=float, default=32.)
parser.add_argument("--grip-rot-k", type=float, default=32.)
parser.add_argument("--grip-rot-d", type=float, default=.8)
parser.add_argument("--friction", type=float, default=1.)
parser.add_argument("--implicit", action="store_true")
parser.add_argument("--pos-iters", type=int, default=16)
parser.add_argument("--margin", type=float, default=.002)
parser.add_argument("--ground-thickness", type=float, default=2.,
                    help="Ground box thickness in metres, with the top fixed at z=0")
parser.add_argument("--disable-unmatched-colliders", action="store_true")
parser.add_argument("--filter-self", action="store_true")
parser.add_argument("--custom-cylinders", action="store_true")
parser.add_argument("--contact-stiffness", type=float, default=0.)
parser.add_argument("--contact-damping", type=float, default=0.)
parser.add_argument("--substeps", type=int, default=4)
parser.add_argument("--options", type=Path)
parser.add_argument("--start-pose", type=Path)
parser.add_argument("--record-contact-failures", action="store_true")
parser.add_argument("--teleop", action="store_true", help="Accept live empty-hand ROS Joy commands")
args = parser.parse_args()
if args.teleop and args.carry:
    parser.error("--teleop is available for the empty-hand scene")
if not math.isfinite(args.ground_thickness) or args.ground_thickness <= 0:
    parser.error("--ground-thickness must be positive and finite")

app = AppLauncher(headless=not args.viewer, device="cpu").app

import carb
import mujoco
import numpy as np
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdPhysics
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from dex_mimic.deployment.runtime import quat_to_mat
from gait import bundle, contract, dt
from realtime.omni import FootMeasurements, PASSES, TRIAL_SECONDS, MOVE_SECONDS
from realtime.omni_preview import OmniPreview, ZERO, yaw_rot
from realtime.omni_report import evaluate
from realtime.sim import CALIBRATION, initialize

SUBSTEPS = args.substeps
PHYSICS_DT = dt / SUBSTEPS
ROBOT = "/World/Dex"
GROUND = "/World/Ground"
FEET = [f"ankle_roll_{side}_link" for side in "lr"]
SHINS = [{f"{joint}_{side}_link" for joint in ("knee_pitch", "ankle_pitch")} for side in "lr"]
NAMES = contract["joint_names"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare_urdf(out, model):
    asset = Path(os.environ.get("DEX_ISAAC_ROBOT_ASSETS", bundle.parent / "assets" / "dex_evt"))
    source = asset / "urdf" / "dex_29.urdf"
    document = ET.parse(source)
    links = document.findall("link")
    if args.disable_unmatched_colliders:
        for link in links:
            body = model.body("robot/" + link.attrib["name"]).id
            enabled = (model.geom_contype | model.geom_conaffinity) != 0
            if not np.any((model.geom_bodyid == body) & enabled):
                for collision in link.findall("collision"):
                    link.remove(collision)
    for kind, folder in (("visual", "meshes_visual"), ("collision", "meshes_convex")):
        for mesh in document.findall(f"./link/{kind}/geometry/mesh"):
            name = Path(mesh.attrib["filename"]).name
            path = (asset / folder / name).resolve(strict=True)
            mesh.set("filename", str(path))
    effective = out / "dex_29_isaac.urdf"
    document.write(effective, encoding="utf-8", xml_declaration=True)
    urdf_masses = {l.attrib["name"]: float(l.find("inertial/mass").attrib["value"])
                   for l in links if l.find("inertial/mass") is not None}
    model_masses = {name: float(model.body_mass[model.body("robot/" + name).id]) for name in urdf_masses}
    return effective, dict(urdf_mass_kg=sum(urdf_masses.values()),
                           frozen_mass_kg=mujoco.mj_getTotalmass(model),
                           maximum_link_mass_error_kg=max(abs(urdf_masses[k] - model_masses[k])
                                                          for k in urdf_masses))


class Contacts:
    def __init__(self, bodies, weight):
        self.weight = weight
        self.rub_count = 0
        self.body_count = 0
        self.pair_counts = {}
        self.names = sorted(bodies)
        view = SimulationManager.get_physics_sim_view()
        self.ground_view = view.create_rigid_contact_view(
            [bodies[name] for name in self.names],
            filter_patterns=[[GROUND + "/geometry/mesh"]] * len(self.names),
            max_contact_data_count=512,
        )
        self.names = [path.rsplit("/", 1)[-1] for path in self.ground_view.sensor_paths]
        self.foot_indices = [self.names.index(foot) for foot in FEET]
        self.other_indices = [i for i in range(len(self.names)) if i not in self.foot_indices]
        self.ground_peak = 0.
        self.rub_views = []
        for i, foot in enumerate(FEET):
            opposite = sorted(SHINS[1-i] | ({FEET[1-i]} if i == 0 else set()))
            contact_view = view.create_rigid_contact_view(
                bodies[foot], filter_patterns=[bodies[name] for name in opposite],
                max_contact_data_count=256,
            )
            filters = contact_view.filter_paths[0]
            resolved = [path.rsplit("/", 1)[-1] for path in filters]
            if set(resolved) != set(opposite):
                raise RuntimeError(f"Missing contact filters for {foot}: {filters}")
            self.rub_views.append((foot, resolved, contact_view))
        if set(self.names) != set(bodies):
            raise RuntimeError("Contact view does not cover all articulation bodies")
        self.reset_frame()

    def reset_frame(self):
        self.impulse = np.zeros(2)
        self.rub = False
        self.body = False

    def sample(self):
        forces = self.ground_view.get_contact_force_matrix(dt=1.).numpy().sum(axis=1)
        self.ground_peak = max(self.ground_peak, float(np.max(np.abs(forces))))
        self.impulse += np.abs(forces[self.foot_indices, 2])
        nonfoot = np.linalg.norm(forces[self.other_indices], axis=1) > 1e-7
        hit = bool(nonfoot.any())
        self.body |= hit
        self.body_count += hit
        for index in np.flatnonzero(nonfoot):
            key = f"{self.names[self.other_indices[index]]} / ground"
            self.pair_counts[key] = self.pair_counts.get(key, 0) + 1
        rub = False
        for foot, opposite, view in self.rub_views:
            impulses = view.get_contact_force_matrix(dt=1.).numpy()[0]
            for name, impulse in zip(opposite, impulses):
                if np.linalg.norm(impulse) > 1e-7:
                    key = f"{foot} / {name}"
                    self.pair_counts[key] = self.pair_counts.get(key, 0) + 1
                    rub = True
        self.rub |= rub
        self.rub_count += rub

    def loads(self):
        return self.impulse / (self.weight * dt)


def filter_self_collisions(bodies, stage):
    """Mirror the frozen self-collision mask, with foot and opposite shin sensing enabled."""
    allowed = {frozenset(FEET)}
    for i, foot in enumerate(FEET):
        allowed.update(frozenset((foot, shin)) for shin in SHINS[1-i])
    names = sorted(bodies)
    for i, first in enumerate(names):
        targets = [bodies[second] for second in names[i+1:] if frozenset((first, second)) not in allowed]
        if targets:
            api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(bodies[first]))
            api.CreateFilteredPairsRel().SetTargets(targets)


def run(out, joy=None):
    out.mkdir(parents=True)
    start_clock = time.monotonic()
    switches = {}
    trials = []
    for i, (label, command) in enumerate(PASSES if joy is None else ()):
        start = 3 + i * TRIAL_SECONDS
        stop = start + MOVE_SECONDS
        switches[round(start / dt)] = command
        switches[round(stop / dt)] = ZERO
        trials.append(dict(label=label, command=command, start_s=start, zero_s=stop,
                           end_s=start + TRIAL_SECONDS))
    carry = None
    if args.carry:
        from carry_bridge import Carry
        if args.implicit or args.start_pose:
            raise ValueError("Carry uses explicit joint torque and the carry protocol initial pose")
        carry = Carry(args, out)
        trials = carry.trials
        switches = {round(trial[key] / dt): command
                    for trial in trials
                    for key, command in (("start_s", tuple(trial["command"])), ("zero_s", ZERO))}
    duration = args.seconds if joy is not None or args.seconds is not None else (trials[-1]["end_s"] if carry else 75.)
    ticks = None if duration is None else round(duration / dt)
    options = json.loads((args.options or Path("realtime/omni_calibration.json")).read_text())
    planner = carry.planner if carry else OmniPreview(json.loads(CALIBRATION.read_text()), options)
    ctl, witness = initialize(planner, out, 400 if ticks is None else ticks)
    if ticks is None:
        from realtime.viewer_common import grow_reference
    ctl.reference["root_R"] = ctl.reference["root_R"].copy()
    assert not ctl.root_feedback
    m, d = witness.model, witness.data
    feet = FootMeasurements(witness)
    q0 = d.qpos.copy()
    if args.start_pose:
        q0 = np.load(args.start_pose)["qpos"][100].copy()
        q0[:2] = 0.
    urdf, audit = prepare_urdf(out, m)
    aids = witness.aids
    kp, kd = m.actuator_gainprm[aids, 0], -m.actuator_biasprm[aids, 2]
    effort = m.actuator_forcerange[aids, 1]
    assert np.all(effort > 0) and np.array_equal(m.actuator_forcerange[aids, 0], -effort)
    assert np.allclose(kp, -m.actuator_biasprm[aids, 1])
    print("V7_MODEL", audit, flush=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device="cpu",
                                                             use_fabric=False))
    carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    carb.settings.get_settings().set_bool("/physics/collisionCylinderCustomGeometry", args.custom_cylinders)
    material = sim_utils.RigidBodyMaterialCfg(static_friction=args.friction,
                                             dynamic_friction=args.friction, restitution=0.,
                                             friction_combine_mode="max",
                                             compliant_contact_stiffness=args.contact_stiffness,
                                             compliant_contact_damping=args.contact_damping)
    ground = sim_utils.CuboidCfg(size=(100., 100., args.ground_thickness),
                                 collision_props=sim_utils.CollisionPropertiesCfg(
                                     contact_offset=args.margin, rest_offset=0.))
    ground.func(GROUND, ground, translation=(0., 0., -.5 * args.ground_thickness))
    mat_path = "/World/DexContactMaterial"
    material.func(mat_path, material)
    sim_utils.bind_physics_material(GROUND, mat_path)
    if carry:
        carry.prepare(ctl, args.margin, mat_path)
        from realtime.carry_model import ARM_ORDER
        kd = kd.copy()
        kd[ARM_ORDER] *= 5.
        kd[NAMES.index("waist_pitch_joint")] *= 7.
    spawn = sim_utils.UrdfFileCfg(asset_path=str(urdf.resolve()), usd_dir=str((out / "usd").resolve()),
                                 collision_from_visuals=False, replace_cylinders_with_capsules=True,
                                 fix_base=False, merge_fixed_joints=False, joint_drive=None,
                                 self_collision=True, activate_contact_sensors=True,
                                 articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                                     enabled_self_collisions=True,
                                     solver_position_iteration_count=args.pos_iters,
                                     solver_velocity_iteration_count=4))
    motors = ImplicitActuatorCfg(
        joint_names_expr=[".*"], stiffness=dict(zip(NAMES, map(float, kp))) if args.implicit else 0.,
        damping=dict(zip(NAMES, map(float, kd))) if args.implicit else 0.,
        effort_limit_sim=dict(zip(NAMES, map(float, effort))),
        armature=dict(zip(NAMES, map(float, m.dof_armature[witness.vadr]))),
        friction=dict(zip(NAMES, map(float, m.dof_frictionloss[witness.vadr]))),
        velocity_limit_sim=100.)
    cfg = ArticulationCfg(prim_path=ROBOT, spawn=spawn, actuators={"motors": motors},
                          init_state=ArticulationCfg.InitialStateCfg(
                              pos=tuple(q0[:3]), rot=tuple(q0[3:7]),
                              joint_pos=dict(zip(NAMES, map(float, q0[witness.qadr]))), joint_vel={".*": 0.}))
    bot = Articulation(cfg)
    # Apply matching friction explicitly to each articulation collider.
    from isaacsim.core.utils.stage import get_current_stage
    stage = get_current_stage()
    for prim in list(stage.Traverse()):
        if str(prim.GetPath()).startswith(ROBOT) and prim.GetName() == "collisions":
            prim.SetInstanceable(False)
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(ROBOT) and prim.HasAPI(UsdPhysics.CollisionAPI):
            sim_utils.modify_collision_properties(
                str(prim.GetPath()), sim_utils.CollisionPropertiesCfg(contact_offset=args.margin, rest_offset=0.))
            sim_utils.bind_physics_material(str(prim.GetPath()), mat_path)
    bodies = {prim.GetName(): str(prim.GetPath()) for prim in stage.Traverse()
              if str(prim.GetPath()).startswith(ROBOT) and prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    print("V7_GROUND", [str(prim.GetPath()) for prim in stage.Traverse()
          if str(prim.GetPath()).startswith(GROUND)], flush=True)
    if args.filter_self:
        filter_self_collisions(bodies, stage)
    print("V7_USD_READY", flush=True)
    sim.reset()
    print("V7_PHYSX_READY", flush=True)
    if args.viewer:
        light = sim_utils.DomeLightCfg(intensity=950., color=(.9, .94, 1.))
        light.func("/World/ViewportLight", light)
        sim.set_camera_view((3.9, -4.4, 2.6), (0.8, 0., .8))
    joint_ids, mapped = bot.find_joints(NAMES, preserve_order=True)
    if mapped != NAMES or len(bot.joint_names) != 29:
        raise RuntimeError(f"Joint name mismatch: {mapped}, {bot.joint_names}")
    tensor = lambda v: torch.tensor(np.asarray(v), dtype=torch.float32).reshape(1, -1)
    bot.write_root_link_pose_to_sim(tensor(q0[:7]))
    bot.write_root_link_velocity_to_sim(tensor(np.zeros(6)))
    bot.write_joint_state_to_sim(tensor(q0[witness.qadr]), tensor(np.zeros(29)), joint_ids=joint_ids)
    bot.set_joint_position_target(tensor(q0[witness.qadr]), joint_ids=joint_ids)
    bot.write_data_to_sim()
    robot_bodies = {name: path for name,path in bodies.items() if path.startswith(ROBOT)}
    contacts = Contacts(robot_bodies, audit["frozen_mass_kg"] * 9.81)
    if carry:
        carry.attach(bot, SimulationManager.get_physics_sim_view())
    foot_body_ids = [bot.body_names.index(name) for name in FEET]
    fk_max = 0.
    audit["isaac_joint_order"] = bot.joint_names
    audit["physx_link_mass_kg"] = float(bot.root_physx_view.get_masses().sum())
    audit["initial_root_z_m"] = float(q0[2])
    audit["joint_armature_error"] = float(np.max(np.abs(bot.root_physx_view.get_dof_armatures()[0, joint_ids].numpy() - m.dof_armature[witness.vadr])))
    write_json(out / "model_audit.json", audit)
    rows, distances, poses, req, plan, stages, commands = [], [], [], [], [], [], []
    reason = ""
    if args.viewer:
        sim.render()
        print("V7_VIEWER_RUNNING", flush=True)
    stop_requested = False
    if joy is not None:
        def request_stop(*_):
            nonlocal stop_requested
            stop_requested = True

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
    wall_clock = time.monotonic()
    if joy is not None:
        print("V7_JOY_READY", flush=True)
    frames = count() if ticks is None else range(ticks + 1)
    for frame in frames:
        if stop_requested or (args.viewer and not app.is_running()):
            break
        if args.viewer or joy is not None:
            time.sleep(max(0., wall_clock + frame * dt - time.monotonic()))
        if args.viewer and (frame % 2 == 0 or frame == ticks):
            sim.render()
        if ticks is None and frame + 2 >= ctl.frames:
            grow_reference(ctl)
        if joy is not None:
            command = joy.command()
            if command != planner.intent:
                planner.command(frame, command)
                print("V7_JOY_COMMAND", round(frame * dt, 2), command, flush=True)
        elif frame in switches:
            planner.command(frame, switches[frame])
        cmd = planner.phase(frame)
        goal = planner.goal(frame)
        root = bot.data.root_link_state_w[0].numpy().copy()
        j = bot.data.joint_pos[0, joint_ids].numpy().copy()
        v = bot.data.joint_vel[0, joint_ids].numpy().copy()
        if not np.isfinite(np.r_[root, j, v]).all():
            reason = "Non-finite PhysX state"
            break
        d.qpos[:7], d.qpos[witness.qadr] = root[:7], j
        mujoco.mj_kinematics(m, d)
        R = quat_to_mat(root[3:7])
        chest = d.xmat[feet.chest].reshape(3, 3)
        pitch = np.rad2deg(np.arcsin(np.clip(-chest[2,0], -1., 1.)))
        roll = np.rad2deg(np.arctan2(chest[2,1], chest[2,2]))
        f = bot.data.body_link_pos_w[0, foot_body_ids].numpy()
        fk_max = max(fk_max, float(np.max(np.linalg.norm(f - d.xpos[feet.feet], axis=1))))
        transverse = (f[0] - f[1]) @ R[:,1]
        rows.append([frame * dt, *root[:3], *root[7:9], pitch, roll, 0., 0.,
                     *contacts.loads(), *f.ravel(), int(contacts.rub), int(contacts.body)])
        distances.append([feet.distance(feet.foot_pairs), feet.distance(feet.shin_pairs), transverse,
                          contacts.rub_count, contacts.body_count])
        poses.append(d.qpos.copy())
        if carry:
            carry.capture(d.qpos, cmd.kind, SUBSTEPS)
        req.append(planner.intent)
        plan.append(cmd.cmd)
        stages.append(cmd.kind)
        if frame % 100 == 0 or (contacts.body and not args.record_contact_failures):
            print("V7_TRACE", round(frame * dt, 2), root[:3], round(float(pitch), 2),
                  round(float(roll), 2), contacts.loads(), str(planner.intent),
                  contacts.body_count, contacts.rub_count,
                  (np.round(carry.gap[:,0], 4).tolist(),
                   np.round(carry.wrench[:,0,2], 2).tolist(),
                   round(carry.balance.vertical / 9.81, 3),
                   carry.fixture.box_steps, carry.fixture.ground_steps) if carry else "",
                  flush=True)
        if root[2] < .55 or R[2,2] < .65:
            reason = "Fall safety threshold"
            break
        if contacts.body and not args.record_contact_failures:
            reason = "Non-foot ground contact"
            break
        if carry and (carry.fixture.ground_steps or (contacts.body and not args.record_contact_failures)
                      or (frame > 1 and np.max(carry.gap[:,0]) > .12)):
            reason = "Box ground contact or grip interface separation"
            break
        if frame == ticks:
            break
        ctl.reference["goal"][frame] = goal
        ctl.reference["root_R"][frame] = yaw_rot(planner.angles[frame])
        target = ctl.step(j, v, root[3:7], R.T @ root[10:13])
        if carry:
            target = carry.target(ctl, target)
        commands.append(target)
        contacts.reset_frame()
        if args.implicit:
            bot.set_joint_position_target(tensor(target), joint_ids=joint_ids)
            bot.write_data_to_sim()
        for substep in range(SUBSTEPS):
            if carry:
                carry.calculate_and_apply(frame * dt + substep * PHYSICS_DT)
            if not args.implicit:
                j_sim = bot.data.joint_pos[0, joint_ids].numpy()
                v_sim = bot.data.joint_vel[0, joint_ids].numpy()
                torque = np.clip(kp * (target - j_sim) - kd * v_sim, -effort, effort)
                bot.set_joint_effort_target(tensor(torque), joint_ids=joint_ids)
                bot.write_data_to_sim()
            sim.step(render=False)
            bot.update(PHYSICS_DT)
            contacts.sample()
            if carry:
                carry.after_step(PHYSICS_DT)

    if not poses:
        return
    audit["max_foot_position_fk_disagreement_m"] = fk_max
    write_json(out / "model_audit.json", audit)
    assert ctl.frame == len(commands)
    times = np.arange(len(poses)) * dt
    np.save(out / "telemetry.npy", rows)
    np.save(out / "clearance.npy", distances)
    np.savez_compressed(out / "rollout.trajectory.npz", qpos=poses, control_dt=dt,
                        physics_time=times, joint_target=commands, policy_sha256=contract["policy_sha256"],
                        model_sha256=contract["model_sha256"], origin="Isaac Sim PhysX closed loop")
    np.savez_compressed(out / "commands.npz", requested=req, planned=plan, stage=stages,
                        reference_root=np.array(planner.positions[:len(poses)])-planner.offset,
                        ref_yaw=planner.angles[:len(poses)])
    meta = dict(backend="Isaac Sim 5.1 / CPU PhysX", completed=not reason and (ticks is None or len(poses) == ticks + 1),
                stop_reason=reason, grid=f"50 Hz WBC / {1/PHYSICS_DT:g} Hz physics", wall_seconds=time.monotonic()-start_clock,
                physics_resets=0, duration_s=float(times[-1]), frames=len(poses),
                physics_steps=(len(poses)-1)*SUBSTEPS, rub_substeps=contacts.rub_count,
                nonsupport_ground_substeps=contacts.body_count, other_contact_pairs=contacts.pair_counts,
                ground_contact_peak_impulse_ns=contacts.ground_peak,
                options=options, revisions=planner.revisions, trials=trials,
                config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "out"},
                blocks=[dict(kind=b.kind, side="LR"[b.side], start_s=b.start*dt, end_s=b.end*dt, cmd=b.cmd)
                        for b in planner.blocks if b.start < len(poses)])
    if carry:
        carry.save(out, times, stages, commands, meta)
    write_json(out / "run.json", meta)
    if not carry:
        evaluate(out)
    print("V7_RESULT", {k: v for k, v in meta.items() if k not in ("blocks", "options", "revisions", "trials")},
          flush=True)

    if args.viewer and app.is_running() and not stop_requested:
        sim.pause()
        print("V7_VIEWER_HOLD", flush=True)
        while app.is_running():
            sim.render()
            time.sleep(.04)


try:
    if args.teleop:
        from ros_sim.joy_pipe import JoyInput
        with JoyInput() as joy:
            run(args.out, joy)
    else:
        run(args.out)
except Exception:
    traceback.print_exc()
    sys.stderr.flush()
    os._exit(1)
else:
    # Kit 5.1 can hang on framework teardown in this headless environment.
    # Artifacts are fully closed before exiting this isolated subprocess.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
