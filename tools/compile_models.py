#!/usr/bin/env python3
"""Compile v7's frozen MuJoCo presets from one mesh tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import mujoco as mj
import numpy as np

MODEL_OPTIONS = dict(timestep=.005, integrator=mj.mjtIntegrator.mjINT_IMPLICITFAST,
                     cone=mj.mjtCone.mjCONE_PYRAMIDAL, solver=mj.mjtSolver.mjSOL_NEWTON,
                     iterations=100, ls_iterations=50, tolerance=1e-6, ccd_iterations=35)
ARMATURE = dict(hip_pitch=.24, hip_roll=.24, hip_yaw=.18, knee_pitch=.37,
                ankle_pitch=.032, ankle_roll=.032, waist_yaw=.17, waist_roll=.17,
                waist_pitch=.17, shoulder_pitch=.1, shoulder_roll=.1, shoulder_yaw=.1,
                elbow_pitch=.1, elbow_yaw=.1, wrist_pitch=.0236, wrist_roll=.0236)
SCENE = """<mujoco model="mjlab scene">
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba force="1 0 0 1" haze="0.15 0.25 0.35 1"/>
    <global azimuth="135" elevation="-25" offwidth="1920" offheight="1080"/>
    <map force="0.005"/>
    <scale forcewidth="0.25" contactwidth="0.4" contactheight="0.15" framelength="5.0" framewidth="0.3"/>
    <quality shadowsize="8192"/>
  </visual>
  <statistic meansize="0.03"/>
</mujoco>"""
FLOOR = """<mujoco>
  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="4 4" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light name="sun" pos="0 0 1.5" dir="0 0 -1" type="directional"/>
    <body name="terrain"><geom name="terrain" size="0 0 0.01" type="plane" material="groundplane"/></body>
  </worldbody>
</mujoco>"""
GRIPS = [[-.157, .182, -.065], [-.157, -.182, -.065]]
SIDES = "lr"


def scene(root, contract):
    assets = Path(os.environ.get("DEX_ISAAC_ROBOT_ASSETS", root/"baseline/assets/dex_evt"))
    robot = mj.MjSpec.from_file(str(assets/"mjcf/dex_29.xml"))
    for sensor in list(robot.sensors):
        if sensor.name.endswith("_joint_torque"):
            robot.delete(sensor)
    for item in list(robot.actuators) + list(robot.worldbody.geoms):
        robot.delete(item)

    names = contract["joint_names"]
    joint_order = [joint.name for joint in robot.joints
                   if joint.type != mj.mjtJoint.mjJNT_FREE]
    if joint_order != names:
        raise RuntimeError("Unexpected compiled joint order")
    for name, kp, kd, n in zip(names, contract["kp"], contract["kd"], contract["effort_limits"]):
        n = float(n)
        j = robot.joint(name)
        motor = robot.add_actuator(name=name, target=name)
        motor.trntype = mj.mjtTrn.mjTRN_JOINT
        motor.dyntype = mj.mjtDyn.mjDYN_NONE
        motor.gaintype = mj.mjtGain.mjGAIN_FIXED
        motor.biastype = mj.mjtBias.mjBIAS_AFFINE
        motor.gainprm[0] = kp
        motor.biasprm[1] = -kp
        motor.biasprm[2] = -kd
        motor.inheritrange = 0.
        motor.ctrllimited = False
        motor.forcelimited = True
        motor.forcerange[:] = [-n, n]
        delta = n / kp
        motor.ctrlrange[:] = [j.range[0]-delta, j.range[1]+delta]
        group = name.replace("_l_joint", "").replace("_r_joint", "").removesuffix("_joint")
        j.armature = ARMATURE[group]
        j.frictionloss = 0.
        j.damping[0] = 0.

    spec = mj.MjSpec.from_string(SCENE)
    spec.attach(mj.MjSpec.from_string(FLOOR), prefix="", frame=spec.worldbody.add_frame())
    spec.attach(robot, prefix="robot/", frame=spec.worldbody.add_frame())
    q = contract["default_q"]
    spec.add_key(name="init_state", qpos=[0.,0.,.97,1.,0.,0.,0.,*q], ctrl=q)
    return spec


def add_box(spec, preset):
    box = spec.worldbody.add_body(name="carry/box")
    box.add_freejoint(name="carry/free")
    hx, hy, hz, h = .145, .175, .095, .007
    walls = [("bottom",[0,0,-hz+h],[hx,hy,h]),
             ("front",[hx-h,0,0],[h,hy,hz]),("back",[-hx+h,0,0],[h,hy,hz]),
             ("left",[0,hy-h,0],[hx-2*h,h,hz]),("right",[0,-hy+h,0],[hx-2*h,h,hz])]
    vol = sum(np.prod(size) for _,_,size in walls)
    for name,pos,size in walls:
        box.add_geom(name=f"carry/{name}", type=mj.mjtGeom.mjGEOM_BOX, pos=pos, size=size,
                     mass=preset["shell_kg"]*np.prod(size)/vol, rgba=[.62,.43,.24,1.],
                     contype=1, conaffinity=1)
    if preset["added_kg"] > 0:
        for side, grip in zip(SIDES, GRIPS):
            box.add_geom(name=f"carry/wrist_load_{side}", type=mj.mjtGeom.mjGEOM_SPHERE,
                         pos=grip, size=[.008], mass=.5*preset["added_kg"], rgba=[0,0,0,0],
                         contype=0, conaffinity=0)
    if preset.get("content_kg", 0.) > 0.:
        inner = np.array([hx-h, hy-h])
        half = min(hz-h, .03)
        box.add_geom(name="carry/content", type=mj.mjtGeom.mjGEOM_BOX,
                     pos=[0., 0., -hz+h+half], size=[*inner, half],
                     mass=preset["content_kg"], rgba=[.32,.34,.38,1.],
                     contype=0, conaffinity=0)
    for i,(side, word) in enumerate(zip(SIDES, ("left", "right"))):
        hand, grip, wrist = (f"carry/{label}_{side}" for label in ("hand", "box", "wrench"))
        tcp = f"robot/{word}_tcp_link"
        roll = f"robot/wrist_roll_{side}_link"
        spec.body(tcp).add_site(name=hand, pos=[0,0,0], quat=preset["orientations"][i],
                                size=[.012], rgba=[.10,.75,.81,1])
        box.add_site(name=grip, pos=GRIPS[i], size=[.010], rgba=[.1,.75,.81,.7])
        spec.body(roll).add_site(name=wrist, size=[.003], rgba=[.1,.75,.81,0.])
        spec.add_equality(name=f"carry/grip_{side}", type=mj.mjtEq.mjEQ_WELD,
                          objtype=mj.mjtObj.mjOBJ_SITE, name1=hand, name2=grip,
                          solref=[preset["solref_s"],1.])
        for kind, sensor_type in (("force",mj.mjtSensor.mjSENS_FORCE),("torque",mj.mjtSensor.mjSENS_TORQUE)):
            spec.add_sensor(name=f"carry/{kind}_{side}", type=sensor_type,
                            objtype=mj.mjtObj.mjOBJ_SITE, objname=wrist)
        for link in (tcp,roll):
            spec.add_exclude(bodyname1="carry/box", bodyname2=link)


def build(root, contract, preset):
    spec = scene(root, contract)
    if preset is not None:
        add_box(spec, preset)
    model = spec.compile()
    for name, value in MODEL_OPTIONS.items():
        setattr(model.opt, name, value)
    model.geom_friction[:, 0] = 1.
    return model


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", choices=["model","empty","one_kg","braced_ten_kg","all"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, help="Override cache directory for comparison")
    args = parser.parse_args()
    root = args.root.resolve()
    bundle = root/"baseline/bundle"
    carry = root/"baseline/carry_models"
    contract = json.loads((bundle/"contract.json").read_text())
    index = json.loads((carry/"index.json").read_text())
    if mj.__version__ != contract["mujoco_version"] or mj.__version__ != index["mujoco_version"]:
        parser.error(f"Requires MuJoCo {contract['mujoco_version']}, active {mj.__version__}")
    presets = {Path(p["file"]).stem:p for p in index["models"]}
    selected = ["model",*presets] if "all" in args.names else args.names
    for name in dict.fromkeys(selected):
        preset = presets.get(name)
        target = (args.out or (bundle if preset is None else carry))/f"{name}.mjb"
        expected = contract["compiled_model_sha256"] if preset is None else preset["sha256"]
        if target.is_file() and digest(target) == expected:
            print(f"{name}: cache verified")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{name}-",dir=target.parent) as tmp:
            generated = Path(tmp)/target.name
            start = time.monotonic()
            mj.mj_saveModel(build(root, contract, preset), str(generated))
            actual = digest(generated)
            print(f"{name}: {time.monotonic()-start:.3f}s {generated.stat().st_size} bytes {actual}",flush=True)
            if actual != expected:
                raise RuntimeError(f"{name}: regenerated model does not match {expected}")
            os.replace(generated, target)


if __name__ == "__main__":
    main()
