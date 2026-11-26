"""Evaluate signed foot-primitive clearance from recorded root and joint poses."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def measure(model, folder):
    """Use the authored foot collision envelope with the scene ground surface at z=0."""
    data = mujoco.MjData(model)
    feet = [model.body(f"robot/ankle_roll_{side}_link").id for side in "lr"]
    cylinder = model.geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER
    capsule = model.geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE
    collidable = (model.geom_contype | model.geom_conaffinity) != 0
    ids = [np.flatnonzero((model.geom_bodyid == foot) & collidable & (cylinder | capsule)) for foot in feet]
    if any(len(foot) != 7 for foot in ids):
        raise ValueError("Expected seven collidable foot primitives per side")
    with np.load(folder / "rollout.trajectory.npz") as archive:
        poses = archive["qpos"]
        times = archive["physics_time"]
    loads = np.load(folder / "telemetry.npy")[:, 10:12]
    counts = np.load(folder / "clearance.npy")[:, 4]
    heights = np.empty_like(loads)
    for i, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        for j, geoms in enumerate(ids):
            axis_z = data.geom_xmat[geoms].reshape(-1, 3, 3)[:, 2, 2]
            radius, half_length = model.geom_size[geoms, :2].T
            radial_z = np.where(capsule[geoms], radius, radius * np.sqrt(np.maximum(0., 1. - axis_z ** 2)))
            extent = radial_z + half_length * abs(axis_z)
            heights[i, j] = np.min(data.geom_xpos[geoms, 2] - extent)

    records = {}
    for label, selection in (("through_15_seconds", times <= 15.), ("entire_record", times >= 0.)):
        z, force = heights[selection], loads[selection]
        lowest, foot = np.unravel_index(z.argmin(), z.shape)
        records[label] = dict(
            minimum_z_m=float(z[lowest, foot]), time_s=float(times[selection][lowest]), foot="LR"[foot],
            contact_load_bw=float(force[lowest, foot]),
            frames_under_ground_by_5mm=int(np.sum(z.min(axis=1) < -.005)),
            unloaded_foot_samples_under_ground_by_10mm=int(np.sum((z < -.01) & (force < .07))),
            nonsupport_ground_substeps=int(counts[selection][-1]),
        )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(measure(mujoco.MjModel.from_binary_path(str(args.model)), args.folder), indent=2))
