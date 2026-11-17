"""Isaac PhysX box, compliant grip interfaces, and the v7 wrench-driven controller."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import mujoco
import torch
from pxr import Gf, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaacsim.core.utils.stage import get_current_stage

from dex_mimic.deployment.runtime import quat_to_mat
from gait import contract, dt
from realtime.carry_empty import Holding, LoadEstimate, mass_bin, EMPTY_TARE_KG
from realtime.carry_load_transfer import BracedCarryPreview, BracedStanding, load_profile, walking_ladder
from realtime.carry_model import ARM_ORDER, BOX, GRIP_OFFSETS, SIDES, TCP, WRISTS, BoxPlant, arm_pose
from realtime.sim import CALIBRATION

BOX_PATH = "/World/CarryBox"
SOURCE = Path("baseline/carry_models/braced_ten_kg.mjb")
KINEMATICS = Path("baseline/carry_models/empty.mjb")
G = 9.81
TRANSFER = 5.
RAMP = .24


def tensor(value):
    return torch.tensor(value, dtype=torch.float32)


def rotation_error_vector(current_quaternion, desired_quaternion):
    conjugate = current_quaternion * [1., -1., -1., -1.]
    delta, axis = np.zeros(4), np.zeros(3)
    mujoco.mju_mulQuat(delta, desired_quaternion, conjugate)
    if delta[0] < 0:
        delta *= -1.
    mujoco.mju_quat2Vel(axis, delta, 1.)
    return axis


def bounded_vectors(value, limit):
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value * np.minimum(1., limit / np.maximum(norm, 1e-9))


class LoadFixture:
    """True mass and external unloading support stay on the physics side."""
    def __init__(self, out, initial_pose, margin, material_path):
        m = mujoco.MjModel.from_binary_path(str(SOURCE))
        bid = m.body(BOX).id
        self.mass = float(m.body_mass[bid])
        self.com = m.body_ipos[bid].copy()
        self.added = sum(float(m.geom(f"carry/wrist_load_{s}").size[0] > 0) for s in SIDES) * 5.
        self.per_grip_n = .5 * self.added * G
        stage = get_current_stage()
        root = UsdGeom.Xform.Define(stage, BOX_PATH).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(root)
        mass = UsdPhysics.MassAPI.Apply(root)
        mass.CreateMassAttr().Set(self.mass)
        mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(*self.com))
        mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*m.body_inertia[bid]))
        w, x, y, z = m.body_iquat[bid]
        mass.CreatePrincipalAxesAttr().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        for i in np.flatnonzero((m.geom_bodyid == bid) & (m.geom_contype != 0)):
            wall = sim_utils.CuboidCfg(size=tuple(2 * m.geom_size[i]),
                                        collision_props=sim_utils.CollisionPropertiesCfg(
                                            contact_offset=margin, rest_offset=0.))
            path = f"{BOX_PATH}/{m.geom(i).name.split('/')[-1]}"
            wall.func(path, wall, translation=tuple(m.geom_pos[i]), orientation=tuple(m.geom_quat[i]))
            sim_utils.bind_physics_material(path, material_path)
        filter_targets = [f"/World/Dex/{n.split('/')[-1]}" for n in (*TCP, *WRISTS)]
        UsdPhysics.FilteredPairsAPI.Apply(root).CreateFilteredPairsRel().SetTargets(filter_targets)
        sim_utils.activate_contact_sensors(BOX_PATH)
        self.body = RigidObject(RigidObjectCfg(prim_path=BOX_PATH, spawn=None,
                                              init_state=RigidObjectCfg.InitialStateCfg(
                                                  pos=tuple(initial_pose[:3]), rot=tuple(initial_pose[3:]))))
        self.start_pose = initial_pose
        self.box_steps = 0
        self.ground_steps = 0
        self.last_contact = False

    def initialize(self, view):
        self.body.write_root_link_state_to_sim(tensor(np.r_[self.start_pose, np.zeros(6)])[None])
        self.ground_view = view.create_rigid_contact_view(
            BOX_PATH, filter_patterns=["/World/Ground/geometry/mesh"], max_contact_data_count=128)
        self.net_view = view.create_rigid_contact_view(BOX_PATH)
        if self.ground_view.sensor_count != 1 or self.ground_view.filter_count != 1:
            raise RuntimeError("Missing box-ground sensor")

    def supported_force(self, time):
        a = np.clip((time - TRANSFER) / RAMP, 0., 1.)
        return self.per_grip_n * (1. - a * a * (3. - 2. * a))

    def contact(self):
        # The box has no native palm collisions; contact forces here are external to the grips.
        ground = self.ground_view.get_contact_force_matrix(dt=1.).numpy()
        net = self.net_view.get_net_contact_forces(dt=1.).numpy()
        self.last_contact = bool(np.linalg.norm(net) > 1e-6 or np.linalg.norm(ground) > 1e-6)
        self.box_steps += self.last_contact
        self.ground_steps += bool(np.linalg.norm(ground) > 1e-6)

    def metadata(self):
        return dict(start_s=TRANSFER, fully_transferred_s=TRANSFER+RAMP,
                    initial_upward_force_per_grip_n=self.per_grip_n,
                    method="Independent support withdrawn at the two physical wrist load centres")


class Carry:
    def __init__(self, options, out):
        self.options = options
        self.out = out
        self.trials = walking_ladder(10., True)
        self.profile = load_profile(Path("realtime"), True, True, None)
        self.arms, self.hand_local_quats = arm_pose(contract["default_q"], .29, -.02)
        cfg = json.loads(CALIBRATION.read_text())
        cfg["anchors"][0]["recipe"].update(stride_scale=1.42, heading_trim=-.31)
        self.planner = BracedCarryPreview(cfg, self.arms, self.profile)
        self.estimate = LoadEstimate(G)
        self.samples, self.balance_rows, self.compensation = [], [], []

    def prepare(self, ctl, margin, material):
        # The controller's FK and arm gravity use a separate empty-box model.
        self.kin = BoxPlant(ctl, mujoco.MjModel.from_binary_path(str(KINEMATICS)))
        self.sensors = SimpleNamespace(ids=np.array([self.kin.model.site(f"carry/wrench_{s}").id for s in SIDES]))
        self.holder = Holding(self.kin, self.arms, 1., 1.)
        self.balance = BracedStanding(self.kin, self.sensors, self.planner, lean_gain=.20,
                                     com_gain=1., force_tau=.20, arm_damping_scale=5., waist_damping_scale=7.)
        d = self.kin.data
        self.fixture = LoadFixture(self.out, d.qpos[self.kin.bq:self.kin.bq+7].copy(), margin, material)

    def attach(self, bot, view):
        self.bot = bot
        self.tcp_ids = [bot.body_names.index(n.split("/")[-1]) for n in TCP]
        self.wrist_ids = [bot.body_names.index(n.split("/")[-1]) for n in WRISTS]
        self.fixture.initialize(view)
        self.gap = np.zeros((2, 2))
        self.wrench = np.zeros((2, 2, 3))
        self.total = np.zeros_like(self.wrench)

    def calculate_and_apply(self, t):
        # Both compliant interfaces use only relative pose and twist. The
        # reaction applied to the opposite side is also the virtual sensor reading.
        box = self.fixture.body
        state = box.data.root_link_state_w[0].numpy()
        box_p, box_q, box_v, box_w = state[:3], state[3:7], state[7:10], state[10:13]
        R = quat_to_mat(box_q)
        grip_offset = GRIP_OFFSETS @ R.T
        box_grips = box_p + grip_offset
        tcp = self.bot.data.body_link_state_w[0, self.tcp_ids].numpy()
        hand_rot = []
        for local, s in zip(self.hand_local_quats, tcp):
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, s[3:7].astype(float), local)
            hand_rot.append(rotation_error_vector(box_q, q))
        hand_rot = np.asarray(hand_rot)
        error = tcp[:,:3] - box_grips
        vrel = tcp[:,7:10] - (box_v + np.cross(box_w, grip_offset))
        K = self.options.grip_k
        D = self.options.grip_d
        f = bounded_vectors(K * error + D * vrel, 650.)
        tau = bounded_vectors(self.options.grip_rot_k * hand_rot
                              + self.options.grip_rot_d * (tcp[:,10:13] - box_w), 35.)
        self.gap = np.stack([np.linalg.norm(error, axis=1), np.linalg.norm(hand_rot, axis=1)], axis=1)
        # Shift the paired moments to the actual body centres of mass.
        wrist_com = self.bot.data.body_com_pos_w[0, self.wrist_ids].numpy()
        box_com = box.data.root_com_pos_w[0].numpy()
        self.bot.set_external_force_and_torque(tensor(-f[None]),
            tensor((-tau + np.cross(tcp[:,:3] - wrist_com, -f))[None]),
            body_ids=self.wrist_ids, is_global=True)
        fixture_f = np.array([0., 0., self.fixture.supported_force(t)])
        box_f = f + fixture_f
        torque_on_box = tau + np.cross(box_grips - box_com, fixture_f) + np.cross(tcp[:,:3] - box_com, f)
        box.set_external_force_and_torque(tensor(box_f.sum(0)[None,None]),
                                          tensor(torque_on_box.sum(0)[None,None]), is_global=True)
        box.write_data_to_sim()
        wrists = self.bot.data.body_link_pos_w[0, self.wrist_ids].numpy()
        self.total[:,0] += f
        self.total[:,1] += tau + np.cross(tcp[:,:3] - wrists, f)

    def after_step(self, physics_dt):
        self.fixture.body.update(physics_dt)
        self.fixture.contact()

    def capture(self, robot_pose, stage, substeps):
        self.kin.data.qpos[:self.kin.bq] = robot_pose
        box = self.fixture.body.data.root_link_state_w[0].numpy().copy()
        self.kin.data.qpos[self.kin.bq:self.kin.bq+7] = box[:7]
        mujoco.mj_kinematics(self.kin.model, self.kin.data)
        self.wrench = self.total.copy() / substeps
        self.total.fill(0.)
        attached = bool(np.max(self.gap[:,0]) < .025)
        good = attached and bool(np.max(self.gap[:,0]) < .015 and np.max(self.gap[:,1]) < .15)
        raw, value, valid = self.estimate.update(self.wrench[:,0], stage, good, self.fixture.last_contact)
        self.samples.append(dict(
            qpos=self.kin.data.qpos.copy(), box_velocity=box[7:13],
            estimates=[raw, np.nan if value is None else value, float(valid), self.estimate.feedforward],
            gap=self.gap.copy(), wrench=self.wrench.copy(), active=[attached, attached],
            box_contacts=self.fixture.box_steps, mass_bin=mass_bin(value),
            estimated_added=np.nan if value is None else value-EMPTY_TARE_KG))
        return self.samples[-1]["qpos"]

    def target(self, controller, target):
        arm_mass = self.balance.arm_mass(self.estimate.feedforward)
        target[ARM_ORDER], effect = self.balance.arm_targets(self.holder, arm_mass)
        attached = self.balance.grip_engaged(self.gap[:,0], self.gap[:,1]) and not self.fixture.last_contact
        self.balance_rows.append(self.balance.step(target, self.wrench, attached))
        self.compensation.append(effect)
        controller.previous_action = np.clip((target-controller.q0)/controller.alpha,
                                              -contract["action_clip"], contract["action_clip"])
        return target

    def save(self, out, times, stages, targets, meta):
        def column(name):
            return np.asarray([s[name] for s in self.samples])
        w = column("wrench")
        np.savez_compressed(out/"rollout.npz", qpos=column("qpos"), physics_time=times,
            wrist_compensated_world=w, grip_wrench_on_box=np.concatenate((w[:,:,0], w[:,:,1]), axis=2),
            estimates=column("estimates"), grip_errors=column("gap"), stage=stages,
            grip_active=column("active"), box_contact_counts=column("box_contacts"),
            estimated_mass_bin=column("mass_bin"), estimated_added_mass_kg=column("estimated_added"),
            box_velocity=column("box_velocity"), actuator_targets=targets,
            balance_trace=self.balance_rows, load_compensation_joint_delta_rad=self.compensation)
        (out/"box_model.mjb").symlink_to(KINEMATICS.resolve())
        meta.update(added_mass_kg=self.fixture.added, model_box_mass_kg=self.fixture.mass,
                    box_free_qadr=self.kin.bq, load_transfer=self.fixture.metadata(),
                    box_contact_steps=self.fixture.box_steps, box_ground_steps=self.fixture.ground_steps,
                    planned_blocks=meta["blocks"], zero_s=None, go_s=10., carry_profile=self.profile,
                    controller_load_input="Virtual wrist wrench from paired compliant grip reactions")
        meta["options"]["omni"] = True
