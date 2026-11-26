"""Connect Dex-RL Joy and bodyctrl_msgs to the record12 MuJoCo simulation."""
import argparse
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import time

import rclpy
from bodyctrl_msgs.msg import CmdMotorCtrl, Imu, MotorCtrl, MotorStatus, MotorStatusMsg
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Joy

from dex_rl.body_id_map import BodyServoIdMap
from joy import BACK_LIMIT, LIMITS, ZERO, StartGate, velocity

TOPIC = '/dex_wbc_sim'
GROUPS = ('leg', 'waist', 'arm')
CONTROL_PERIOD = .01

def set_stamp(msg, ns):
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ns, 1_000_000_000)


def stamp(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


class PhysicsProcess:
    def __init__(self, python, out, viewer):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=f'{root}/baseline/src:{root}', DEX_MIMIC_ROOT=str(root/'baseline'),
                   PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', DEX_EVALUATION_PROFILE='jump_tracking_v2')
        self.log = (out/'worker.log').open('w')
        command = [python, '-u', str(Path(__file__).with_name('worker.py')), str(out)]
        if viewer:
            command.append('--viewer')
        self.process = subprocess.Popen(command, cwd=root, env=env, text=True, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log, start_new_session=True)
        try:
            self.ready = self.read(60.)
        except BaseException:
            self.close()
            raise

    def read(self, timeout=8.):
        if not select.select([self.process.stdout], [], [], timeout)[0]:
            raise TimeoutError('WBC worker timed out; see worker.log')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('WBC worker exited; see worker.log')
        return json.loads(line)

    def call(self, method, *args):
        self.process.stdin.write(json.dumps([method, args], allow_nan=False)+'\n')
        self.process.stdin.flush()
        return self.read()

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()
        self.log.close()


def group_ids():
    mapping = BodyServoIdMap()
    mapping.body_can_id_map_init()
    ids = list(mapping.index_to_id_map.values())
    n = mapping.leg_motor_nums
    m = n + mapping.waist_motor_nums
    return dict(zip(GROUPS, (ids[:n], ids[n:m], ids[m:])))


def check_joint_order(joints):
    mapping = BodyServoIdMap()
    mapping.body_can_id_map_init()
    for i, joint in enumerate(joints):
        name = joint.removesuffix('_joint')
        if not name.startswith('waist_'):
            base, side = name.rsplit('_', 1)
            base = {'knee_pitch': 'knee', 'elbow_pitch': 'elbow', 'elbow_yaw': 'wrist_yaw'}.get(base, base)
            name = f'{side}_{base}'
        if mapping.get_index_by_name(name) != i:
            raise ValueError(f'Inconsistent Dex-RL motor order for {joint}')
    if len(joints) != mapping.whole_motor_nums:
        raise ValueError('Incomplete Dex-RL motor mapping')


class Loop:
    def __init__(self, executor, simulation, controller, *, require_start):
        self.executor, self.simulation, self.controller = executor, simulation, controller
        self.ids = group_ids()
        self.feedback, self.commands = {}, {}
        self.joy, self.joy_time, self.joy_count = ZERO, 0., 0
        self.start_gate = StartGate() if require_start else None
        self.status_pub, self.command_pub = {}, {}
        for group in GROUPS:
            status = f'{TOPIC}/{group}/status'
            command = f'{TOPIC}/{group}/cmd_ctrl'
            self.status_pub[group] = simulation.create_publisher(MotorStatusMsg, status, 10)
            controller.create_subscription(MotorStatusMsg, status, self.receive(self.feedback, group), 10)
            self.command_pub[group] = controller.create_publisher(CmdMotorCtrl, command, 10)
            simulation.create_subscription(CmdMotorCtrl, command, self.receive(self.commands, group), 10)
        imu = f'{TOPIC}/imu/data'
        self.imu_pub = simulation.create_publisher(Imu, imu, 10)
        controller.create_subscription(Imu, imu, self.receive(self.feedback, 'imu'), 10)
        controller.create_subscription(Joy, '/sbus_data', self.receive_joy, 10)

    @staticmethod
    def receive(destination, group):
        def callback(msg):
            destination[group] = msg
        return callback

    def receive_joy(self, msg):
        self.joy = self.start_gate.update(msg.axes) if self.start_gate else velocity(msg.axes)
        self.joy_time = time.monotonic()
        self.joy_count += 1

    def wait_for(self, ready, label):
        end = time.monotonic() + 5.
        while not ready():
            if time.monotonic() >= end or not rclpy.ok():
                raise TimeoutError(f'ROS transport timeout: {label}')
            self.executor.spin_once(timeout_sec=.005)

    def wait_for_topics(self):
        pubs = [*self.status_pub.values(), self.imu_pub, *self.command_pub.values()]
        self.wait_for(lambda: all(p.get_subscription_count() for p in pubs), 'motor topic discovery')

    def read_state(self, state, ns):
        q, qd, quat, gyro = state
        offset = 0
        for group, ids in self.ids.items():
            msg = MotorStatusMsg()
            set_stamp(msg, ns)
            msg.status = [MotorStatus(name=ident, pos=q[i], speed=qd[i]) for i, ident in enumerate(ids, offset)]
            offset += len(ids)
            self.status_pub[group].publish(msg)
        imu = Imu()
        set_stamp(imu, ns)
        imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z = quat
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = gyro
        self.imu_pub.publish(imu)
        self.wait_for(lambda: self.matching(self.feedback, ns, 4), 'joint feedback and IMU')
        status = self.ordered(self.feedback, 'status')
        imu = self.feedback['imu']
        return [[x.pos for x in status], [x.speed for x in status],
                [imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z],
                [imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z]]

    @staticmethod
    def matching(messages, ns, count):
        return len(messages) == count and all(stamp(m) == ns for m in messages.values())

    def ordered(self, received, field):
        result = []
        for group, ids in self.ids.items():
            rows = getattr(received[group], field)
            by_id = {x.name: x for x in rows}
            if len(rows) != len(ids) or by_id.keys() != set(ids):
                raise ValueError(f'Incorrect Dex-RL motor IDs in {group}/{field}')
            result.extend(by_id[i] for i in ids)
        return result

    def motor_command(self, target, kp, kd, ns):
        offset = 0
        for group, ids in self.ids.items():
            msg = CmdMotorCtrl()
            set_stamp(msg, ns)
            msg.cmds = [MotorCtrl(name=ident, kp=kp[i], kd=kd[i], pos=target[i])
                        for i, ident in enumerate(ids, offset)]
            offset += len(ids)
            self.command_pub[group].publish(msg)
        self.wait_for(lambda: self.matching(self.commands, ns, 3), 'joint commands')
        rows = self.ordered(self.commands, 'cmds')
        return [[getattr(x, field) for x in rows] for field in ('pos', 'kp', 'kd', 'spd', 'tor')]


def route(profile):
    if profile == 'smoke':
        segments = [('Forward 0.10', (.1, 0., 0.)), ('Left 0.30', (0., .3, 0.)),
                    ('Turn right 0.45', (0., 0., -.45))]
    elif profile == 'ladder':
        speeds = list(range(1, 13)) + list(range(11, 0, -1))
        trials = [dict(label=f'Forward {s/10:.2f}', command=(s/10, 0., 0.),
                       start_s=3.+5*i, zero_s=8.+5*i, end_s=8.+5*i) for i, s in enumerate(speeds)]
        trials[-1]['end_s'] += 6.
        return trials
    elif profile == 'directions':
        segments = [(f'{side} {speed:.2f}', tuple(sign*speed if k == axis else 0. for k in range(3)))
                    for axis, (labels, speeds) in enumerate([(('Left', 'Right'), (.1, .2, .3)),
                                                             (('Turn left', 'Turn right'), (.15, .3, .45))], 1)
                    for side, sign in zip(labels, (1, -1)) for speed in speeds]
    else:
        segments = [(f'Forward {s/10:.2f}', (s/10, 0., 0.)) for s in range(1, 13)]
    return [dict(label=label, command=cmd, start_s=3.+12*i, zero_s=10.+12*i, end_s=15.+12*i)
            for i, (label, cmd) in enumerate(segments)]


def run(args, loop, worker, teleop):
    trials = route(args.demo) if args.demo else []
    end = trials[-1]['end_s'] if trials else args.seconds
    events = {}
    for tr in trials:
        events[round(tr['zero_s']/CONTROL_PERIOD)] = ZERO
    for tr in trials:
        events[round(tr['start_s']/CONTROL_PERIOD)] = tr['command']
    ready = worker.ready
    check_joint_order(ready['joints'])
    loop.wait_for_topics()
    start = time.monotonic()
    epoch = loop.simulation.get_clock().now().nanoseconds
    deadline, target, state = start, None, ready['state']
    ticks, repeats, max_wall, missed, error = 0, 0, 0., 0, None
    stamps = []
    try:
        while rclpy.ok() and (not end or ticks < round(end/CONTROL_PERIOD)):
            if args.pace:
                time.sleep(max(0., deadline-time.monotonic()))
            t0 = time.monotonic()
            if ticks in events:
                vx, vy, wz = events[ticks]
                teleop.set_axis_override('y1', vx/(LIMITS[0] if vx >= 0 else BACK_LIMIT))
                teleop.set_axis_override('x1', -vy/LIMITS[1])
                teleop.set_axis_override('turn', -wz/LIMITS[2])
                previous_joy_count = loop.joy_count
                teleop._publish()
                loop.wait_for(lambda: loop.joy_count > previous_joy_count and loop.joy == events[ticks], 'scripted Joy velocity')
                print(f'{ticks*CONTROL_PERIOD:6.2f}s {loop.joy}', flush=True)
            ns = epoch + ticks * 10_000_000
            feedback = loop.read_state(state, ns)
            if ticks % 2 == 0:
                command = loop.joy if trials or time.monotonic()-loop.joy_time <= args.joy_timeout else ZERO
                target = worker.call('infer', feedback, command)
            else:
                repeats += 1
            applied = loop.motor_command(target, ready['kp'], ready['kd'], ns)
            if applied[0] != target:
                raise ValueError('Unexpected ROS motor position quantization')
            stamps.append(time.monotonic()-start)
            reply = worker.call('advance', *applied)
            ticks += 1
            state = reply['state']
            if reply['body_contacts']:
                error = 'non-foot ground contact'
                break
            viewer_code = reply['viewer_exit_code']
            if viewer_code is not None:
                if viewer_code:
                    error = f'MuJoCo viewer exited with status {viewer_code}; see worker.log'
                break
            elapsed = time.monotonic()-t0
            max_wall = max(max_wall, elapsed)
            missed += elapsed > CONTROL_PERIOD
            deadline = max(deadline+CONTROL_PERIOD, time.monotonic())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        error = str(exc)
    complete = error is None and (not end or ticks == round(end/CONTROL_PERIOD))
    wall = time.monotonic()-start
    ros = dict(teleop_topic='/sbus_data', motor_topic_prefix=TOPIC, domain_id=int(os.environ['ROS_DOMAIN_ID']),
               simulated_only=True, lockstep=True, paced=args.pace, repeated_targets=repeats,
               joy_messages=loop.joy_count, wall_seconds=wall, command_rate_wall_hz=ticks/wall,
               cycles_over_10ms=missed, max_cycle_ms=max_wall*1000, error=error)
    (args.out/'ros_command_wall_times.json').write_text(json.dumps(stamps)+'\n')
    result = worker.call('save', trials, complete, ros)
    print(json.dumps({k: v for k, v in result.items() if k not in ('blocks', 'trials')}, indent=2), flush=True)
    return 0 if complete else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wbc-python', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--demo', choices=['smoke', 'ladder', 'directions', 'forward-stops'])
    p.add_argument('--seconds', type=float, default=0., help='Interactive simulation duration; 0 runs until stopped')
    p.add_argument('--viewer', action='store_true', help='Open a passive MuJoCo window on the current display')
    p.add_argument('--joy-timeout', type=float, default=.3)
    p.add_argument('--unpaced', action='store_true', help='Use deterministic accelerated replay for a scripted route')
    args = p.parse_args()
    if args.unpaced and (not args.demo or args.viewer):
        p.error('--unpaced requires a headless demo')
    if not math.isfinite(args.seconds) or args.seconds < 0 or not math.isfinite(args.joy_timeout) or args.joy_timeout <= 0:
        p.error('Duration and Joy timeout must be finite and valid')
    args.pace = not args.unpaced
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    rclpy.init(args=[])
    executor = SingleThreadedExecutor()
    simulation = rclpy.create_node('dex_wbc_mujoco')
    controller = rclpy.create_node('dex_record12_controller')
    nodes = [simulation, controller]
    teleop, worker = None, None
    try:
        if args.demo:
            from dex_rl.gui_monitor import TeleopJoyNode
            teleop = TeleopJoyNode(enable_gamepad=False, enable_keyboard=False, enable_console=False)
            nodes.append(teleop)
        for node in nodes:
            executor.add_node(node)
        loop = Loop(executor, simulation, controller, require_start=not args.demo)
        worker = PhysicsProcess(args.wbc_python, args.out, args.viewer)
        return run(args, loop, worker, teleop)
    finally:
        if worker is not None:
            worker.close()
        for node in nodes:
            node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
