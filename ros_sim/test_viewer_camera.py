"""Exercise the input callbacks with the installed MuJoCo camera binding."""
import unittest
from unittest.mock import patch

import glfw
import mujoco
import numpy as np

from viewer import camera_controls


class CameraControlsTest(unittest.TestCase):
    def test_drag_and_scroll_use_the_mujoco_binding(self):
        model = mujoco.MjModel.from_xml_string('<mujoco/>')
        camera = mujoco.MjvCamera()
        camera.distance = 3.
        window = object()
        with (patch.object(glfw, 'get_cursor_pos', return_value=(0, 0)),
              patch.object(glfw, 'get_window_size', return_value=(1360, 850)),
              patch.object(glfw, 'set_cursor_pos_callback') as cursor,
              patch.object(glfw, 'set_scroll_callback') as scroll,
              patch.object(glfw, 'set_key_callback'),
              patch.object(glfw, 'get_mouse_button') as button):
            camera_controls(window, model, camera)
            drag_callback = cursor.call_args.args[1]
            scroll_callback = scroll.call_args.args[1]

            button.side_effect = lambda _, which: glfw.PRESS if which == glfw.MOUSE_BUTTON_LEFT else glfw.RELEASE
            angle = camera.azimuth
            drag_callback(window, 40, 20)
            self.assertNotEqual(camera.azimuth, angle)

            button.side_effect = lambda _, which: glfw.PRESS if which == glfw.MOUSE_BUTTON_RIGHT else glfw.RELEASE
            center = camera.lookat.copy()
            drag_callback(window, 65, 42)
            self.assertFalse(np.array_equal(camera.lookat, center))

            distance = camera.distance
            scroll_callback(window, 0, 1)
            self.assertNotEqual(camera.distance, distance)


if __name__ == '__main__':
    unittest.main()
