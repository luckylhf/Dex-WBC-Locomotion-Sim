"""Receive ROS Joy in an Isaac process with a different Python ABI."""
import os
from pathlib import Path
import socket
import struct
import subprocess
import time

from ros_sim.joy import ISAAC_FORWARD_LIMIT, StartGate, ZERO

READY = b'ready'
STOP = b'stop'
JOY_SAMPLE = struct.Struct('=13d')
JOY_TIMEOUT = .3


class JoyInput:
    def __enter__(self):
        self.rx, tx = socket.socketpair(type=socket.SOCK_DGRAM)
        self.process = None
        self.gate = StartGate()
        self.last_stamp = None
        self.velocity = ZERO
        env = os.environ.copy()
        # Load ROS Humble against system Python and its own shared libraries.
        for name in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'LD_PRELOAD',
                     'AMENT_PREFIX_PATH', 'COLCON_PREFIX_PATH', 'CMAKE_PREFIX_PATH',
                     'VIRTUAL_ENV', 'CONDA_PREFIX', 'ROS_DISTRO', 'ROS_VERSION', 'ROS_PYTHON_VERSION'):
            env.pop(name, None)
        try:
            with tx:
                script = Path(__file__).with_name('run.sh')
                self.process = subprocess.Popen(
                    ['bash', str(script), 'joy-relay', str(tx.fileno())],
                    env=env, pass_fds=(tx.fileno(),))
            self.rx.settimeout(5.)
            if self.rx.recv(JOY_SAMPLE.size) != READY:
                raise RuntimeError('Unexpected ROS Joy startup message')
            self.rx.setblocking(False)
        except (OSError, RuntimeError) as exc:
            self.close()
            raise RuntimeError('ROS Joy startup failed. Check ROS setup and ./ros_sim/run.sh build.') from exc
        return self

    def close(self):
        if self.process is not None and self.process.poll() is None:
            try:
                self.rx.send(STOP)
            except OSError:
                pass
        self.rx.close()
        if self.process is not None:
            try:
                self.process.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def __exit__(self, *_):
        self.close()

    def command(self):
        now = time.monotonic()
        while True:
            try:
                data = self.rx.recv(JOY_SAMPLE.size)
            except BlockingIOError:
                break
            if len(data) != JOY_SAMPLE.size:
                break
            stamp, *axes = JOY_SAMPLE.unpack(data)
            if stamp < now - JOY_TIMEOUT:
                continue
            if self.last_stamp is not None and stamp - self.last_stamp > JOY_TIMEOUT:
                self.gate = StartGate()
            self.last_stamp = stamp
            vx, vy, yaw = self.gate.update(axes)
            self.velocity = min(vx, ISAAC_FORWARD_LIMIT), vy, yaw
        if self.last_stamp is not None and now - self.last_stamp > JOY_TIMEOUT:
            self.last_stamp = None
            self.gate = StartGate()
            self.velocity = ZERO
        return self.velocity
