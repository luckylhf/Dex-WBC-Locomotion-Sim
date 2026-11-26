"""Render WBC pose snapshots on a separate wall clock."""
from collections import deque
import json
from pathlib import Path
import socket
import struct
import sys
import time

import glfw
import mujoco
import numpy as np

FRAME_PERIOD = 1 / 60
DISPLAY_DELAY = .015
STAMP = struct.Struct('=d')


class PoseStream:
    def __init__(self, connection, model):
        self.connection = connection
        self.size = STAMP.size + model.nq * np.dtype(np.float64).itemsize
        self.samples = deque(maxlen=8)
        self.velocity = np.empty(model.nv)
        self.model = model

    def read(self, *, wait=False):
        while True:
            try:
                packet = self.connection.recv(self.size)
            except BlockingIOError:
                return True
            if not packet:
                return False
            when = STAMP.unpack_from(packet)[0]
            pose = np.frombuffer(packet, dtype=np.float64, offset=STAMP.size)
            self.samples.append((when, pose))
            if wait:
                return True

    def pose_at(self, when, qpos):
        while len(self.samples) > 2 and self.samples[1][0] <= when:
            self.samples.popleft()
        start, first = self.samples[0]
        end, second = self.samples[-1] if len(self.samples) == 1 else self.samples[1]
        if when <= start or end <= start:
            qpos[:] = first
        elif when >= end:
            qpos[:] = second
        else:
            qpos[:] = first
            mujoco.mj_differentiatePos(self.model, self.velocity, end-start, first, second)
            mujoco.mj_integratePos(self.model, qpos, self.velocity, when-start)


def camera_controls(window, model, camera):
    x, y = glfw.get_cursor_pos(window)
    def cursor(window, next_x, next_y):
        nonlocal x, y
        dx, dy = next_x - x, next_y - y
        x, y = next_x, next_y
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            return
        height = max(glfw.get_window_size(window)[1], 1)
        mujoco.mjv_moveCamera(model, action, dx/height, dy/height, camera)
    def scroll(window, xoffset, yoffset):
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0., -.05*yoffset, camera)
    def key(window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
    glfw.set_cursor_pos_callback(window, cursor)
    glfw.set_scroll_callback(window, scroll)
    glfw.set_key_callback(window, key)


def write_timing(path, swapped, draw_ms, latest_age_ms, framebuffer):
    times = np.asarray(swapped)
    interval_ms = np.diff(times)*1000
    summary = dict(frames=len(times), swap_wall_hz=float(1000/interval_ms.mean()),
                   frame_interval_ms_p95=float(np.percentile(interval_ms,95)),
                   frame_interval_ms_max=float(interval_ms.max()),
                   draw_ms_p95=float(np.percentile(draw_ms,95)),
                   latest_pose_age_ms_p95=float(np.percentile(latest_age_ms,95)),
                   framebuffer=list(framebuffer), display_delay_ms=DISPLAY_DELAY*1000)
    tmp = path.with_suffix('.part')
    tmp.write_text(json.dumps(summary,indent=2)+'\n')
    tmp.replace(path)


def render(window, model, stream, timing_path):
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = model.body('robot/pelvis').id
    camera.distance, camera.azimuth, camera.elevation = 3., 135., -14.
    scene = mujoco.MjvScene(model, maxgeom=model.ngeom+200)
    for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION):
        scene.flags[flag] = 0
    options = mujoco.MjvOption()
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
    camera_controls(window, model, camera)
    swapped, draw_ms, pose_age_ms = deque(maxlen=600), deque(maxlen=600), deque(maxlen=600)
    next_frame = time.monotonic()
    next_report = next_frame+5.
    try:
        while not glfw.window_should_close(window):
            remaining = next_frame-time.monotonic()
            if remaining > 0:
                glfw.wait_events_timeout(remaining)
                continue
            started = time.monotonic()
            next_frame += FRAME_PERIOD
            if next_frame < started:
                next_frame = started+FRAME_PERIOD
            glfw.poll_events()
            if not stream.read():
                break
            framebuffer = glfw.get_framebuffer_size(window)
            if not all(framebuffer):
                continue
            stream.pose_at(started-DISPLAY_DELAY, data.qpos)
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_camlight(model, data)
            mujoco.mjv_updateScene(model, data, options, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            rect = mujoco.MjrRect(0, 0, *framebuffer)
            mujoco.mjr_render(rect, scene, context)
            glfw.swap_buffers(window)
            finished = time.monotonic()
            swapped.append(finished)
            draw_ms.append((finished-started)*1000)
            pose_age_ms.append(max(0., started-stream.samples[-1][0])*1000)
            if finished >= next_report and len(swapped) >= 2:
                write_timing(timing_path, swapped, draw_ms, pose_age_ms, framebuffer)
                next_report = finished+5.
    finally:
        context.free()


def show(model_path, descriptor, timing_path):
    model = mujoco.MjModel.from_binary_path(model_path)
    # These graphics settings apply only to the display model.
    model.vis.quality.shadowsize = 1024
    with socket.socket(fileno=descriptor) as connection:
        stream = PoseStream(connection, model)
        if not stream.read(wait=True):
            return
        connection.setblocking(False)
        if not glfw.init():
            raise RuntimeError('GLFW could not open the current display')
        try:
            glfw.window_hint(glfw.SAMPLES, 0)
            window = glfw.create_window(1360, 850, 'MuJoCo - Dex WBC ROS', None, None)
            if window is None:
                raise RuntimeError('GLFW could not create the MuJoCo window')
            glfw.make_context_current(window)
            glfw.swap_interval(0)
            render(window, model, stream, timing_path)
        finally:
            glfw.terminate()


if __name__ == '__main__':
    show(sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]))
