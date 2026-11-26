"""Run the frozen WBC and native physics behind the ROS version boundary."""
import hashlib
import json
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from gait import contract, dt
from realtime.omni import FootMeasurements
from realtime.omni_preview import OmniPreview, yaw_rot
from realtime.sim import initialize
from realtime.viewer_common import grow_reference, writable_reference


class Worker:
    def __init__(self, out):
        self.out = out
        self.calibration = Path(__file__).with_name('forward_calibration.json')
        cal = json.loads(self.calibration.read_text())
        options = json.loads((Path(__file__).resolve().parents[1] / 'realtime/omni_calibration.json').read_text())
        self.preview = OmniPreview(cal, options)
        self.ctl, self.plant = initialize(self.preview, out, 400)
        writable_reference(self.ctl)
        if self.ctl.root_feedback:
            raise ValueError('ROS simulation requires the frozen proprioceptive controller')
        self.measures = FootMeasurements(self.plant)
        m, d, aids = self.plant.model, self.plant.data, self.plant.aids
        self.kp = m.actuator_gainprm[aids, 0].copy()
        self.kd = -m.actuator_biasprm[aids, 2].copy()
        if not np.array_equal(self.kp, -m.actuator_biasprm[aids, 1]):
            raise ValueError('Frozen actuator is not a position servo')
        if not np.allclose(self.kp, contract['kp']) or not np.allclose(self.kd, contract['kd']):
            raise ValueError('Frozen actuator gains differ from the contract')
        self.substeps = round(.01 / m.opt.timestep)
        if (self.substeps * 2 != contract['decimation'] or not np.isclose(m.opt.timestep, .005)):
            raise ValueError('Expected two physics substeps per ROS command')
        self.viewer = None
        self.ticks = 0
        self.trace, self.gaps, self.poses, self.times = [], [], [], []
        self.requested, self.planned, self.stages, self.targets, self.compute_ms = [], [], [], [], []
        self.record((0., 0., 0.), (0., 0., 0.), 'stand')

    def record(self, requested, planned, stage):
        row, gap = self.measures.snapshot(self.ticks // 2)
        self.trace.append(row)
        self.gaps.append(gap)
        self.times.append(float(self.plant.data.time))
        self.poses.append(self.plant.data.qpos.copy())
        self.requested.append(requested)
        self.planned.append(planned)
        self.stages.append(stage)

    def feedback(self):
        q, qd, quat, gyro = self.plant.sensors()
        return [v.tolist() for v in (q, qd, quat, gyro)]

    def infer(self, state, command):
        start = time.perf_counter()
        frame = self.ctl.frame
        if frame != self.ticks // 2 or self.ticks % 2:
            raise RuntimeError('Unexpected policy clock')
        if frame + 2 >= self.ctl.frames:
            grow_reference(self.ctl, 400)
        command = tuple(command)
        self.preview.command(frame, command)
        self.ctl.reference['goal'][frame] = self.preview.goal(frame)
        self.ctl.reference['root_R'][frame] = yaw_rot(self.preview.angles[frame])
        block = self.preview.phase(frame)
        self.planned[-1], self.stages[-1], self.requested[-1] = block.cmd, block.kind, command
        target = self.ctl.step(*state)
        self.targets.append(target.copy())
        self.compute_ms.append((time.perf_counter() - start) * 1000)
        return target.tolist()

    def advance(self, target, kp, kd, speed, effort):
        if not (np.array_equal(kp, self.kp) and np.array_equal(kd, self.kd) and not any(speed) and not any(effort)):
            raise ValueError('Unsupported actuator command for the frozen plant')
        target = np.asarray(target, dtype=np.float32)
        if target.shape != self.kp.shape or not np.isfinite(target).all():
            raise ValueError('Invalid joint targets')
        p, m, d = self.plant, self.plant.model, self.plant.data
        d.ctrl[p.aids] = target
        for _ in range(self.substeps):
            mujoco.mj_step(m, d)
            rub, body = self.measures.contacts()[1:]
            self.measures.rub_steps += rub
            self.measures.body_steps += body
        self.ticks += 1
        mujoco.mj_forward(m, d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or abs(d.time - self.ticks*.01) > 1e-8:
            raise FloatingPointError('Invalid MuJoCo state or physics clock reset')
        if self.ticks % 2 == 0:
            self.record(self.requested[-1], self.planned[-1], self.stages[-1])
        if self.viewer is not None:
            self.viewer.publish(d.qpos)
        viewer_code = self.viewer.process.poll() if self.viewer is not None else None
        return dict(state=self.feedback(), body_contacts=self.measures.body_steps, viewer_exit_code=viewer_code)

    def save(self, trials, completed, ros):
        out, preview = self.out, self.preview
        if not np.allclose(np.diff(self.times), dt, atol=1e-9, rtol=0):
            raise ValueError('Non-continuous recorded physics clock')
        if np.any(self.plant.data.xfrc_applied) or np.any(self.plant.data.qfrc_applied):
            raise ValueError('Unexpected applied force')
        np.save(out/'telemetry.npy', self.trace)
        np.save(out/'clearance.npy', self.gaps)
        np.savez_compressed(out/'rollout.trajectory.npz', qpos=self.poses, physics_time=self.times,
                            control_dt=dt, joint_targets=self.targets, policy_sha256=contract['policy_sha256'],
                            model_sha256=contract['model_sha256'])
        np.savez_compressed(out/'commands.npz', requested=self.requested, planned=self.planned,
                            stage=self.stages, ref_yaw=preview.angles[:len(self.poses)],
                            reference_root=np.asarray(preview.positions[:len(self.poses)])-preview.offset)
        run = dict(completed=completed, duration_s=float(self.plant.data.time), physics_resets=0,
                   physics_steps=self.ticks*self.substeps, policy_updates=self.ctl.frame,
                   actuator_commands=self.ticks, trials=trials, ros=ros, no_external_support=True,
                   rub_substeps=self.measures.rub_steps, nonsupport_ground_substeps=self.measures.body_steps,
                   minimum_pelvis_height_m=float(np.min(np.asarray(self.poses)[:, 2])),
                   policy_compute_ms_p95=float(np.percentile(self.compute_ms, 95)) if self.compute_ms else None,
                   calibration_sha256=hashlib.sha256(self.calibration.read_bytes()).hexdigest(),
                   blocks=[dict(kind=b.kind, side='LR'[b.side], start_s=b.start*dt, end_s=b.end*dt, cmd=b.cmd)
                           for b in preview.blocks if b.start <= len(self.poses)-1])
        (out/'run.json').write_text(json.dumps(run, indent=2)+'\n')
        return run


class ViewerProcess:
    """Send pose snapshots without waiting for display refresh or user input."""

    def __init__(self, model_path, out):
        self.connection, display = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.connection.setblocking(False)
        try:
            with display:
                self.process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).with_name('viewer.py')), str(model_path),
                     str(display.fileno()), str(out/'viewer_timing.json')],
                    pass_fds=(display.fileno(),), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        except BaseException:
            self.connection.close()
            raise

    def publish(self, qpos):
        try:
            self.connection.send(struct.pack('=d', time.monotonic()) + qpos.tobytes())
        except (BlockingIOError, BrokenPipeError, ConnectionResetError):
            pass

    def close(self):
        self.connection.close()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait()


def main():
    worker = Worker(Path(sys.argv[1]))
    if '--viewer' in sys.argv[2:]:
        worker.viewer = ViewerProcess(worker.ctl.bundle / 'model.mjb', worker.out)
        worker.viewer.publish(worker.plant.data.qpos)
    print(json.dumps(dict(joints=contract['joint_names'], state=worker.feedback(),
                          kp=worker.kp.tolist(), kd=worker.kd.tolist())), flush=True)
    commands = {'infer': worker.infer, 'advance': worker.advance, 'save': worker.save}
    try:
        for line in sys.stdin:
            method, values = json.loads(line)
            result = commands[method](*values)
            print(json.dumps(result, allow_nan=False), flush=True)
    finally:
        if worker.viewer is not None:
            worker.viewer.close()


if __name__ == '__main__':
    main()
