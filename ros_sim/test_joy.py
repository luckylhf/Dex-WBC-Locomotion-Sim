import struct
import unittest

from joy import ZERO, StartGate, speed_levels, speed_to_stick, velocity


class JoyTest(unittest.TestCase):
    def test_sliders_contain_only_selectable_commands(self):
        self.assertEqual(speed_levels(0)[6:10], (0., .10, .15, .20))
        axes = [0.] * 12
        for i, topic_axis in enumerate((2, 3, 0)):
            for speed in speed_levels(i):
                stick = speed_to_stick(i, speed)
                axes[topic_axis] = struct.unpack('=f', struct.pack('=f', stick))[0]
                self.assertEqual(velocity(axes)[i], speed)
            axes[topic_axis] = 0.

    def test_start_order_and_stop(self):
        axes = [0.] * 12
        axes[2] = .25
        axes[3] = -1/3
        gate = StartGate()
        self.assertEqual(gate.update(axes), ZERO)
        axes[8] = 1.
        self.assertEqual(gate.update(axes), ZERO)  # A alone.
        axes[8], axes[11] = 0., 1.
        self.assertEqual(gate.update(axes), ZERO)  # D prepares.
        axes[8] = 1.
        self.assertEqual(gate.update(axes), ZERO)  # Pressing A while holding D.
        axes[11] = 0.
        self.assertEqual(gate.update(axes), ZERO)  # The old A press cannot arm motion.
        axes[8] = 0.
        self.assertEqual(gate.update(axes), ZERO)
        axes[8] = 1.
        self.assertEqual(gate.update(axes), (.30, .10, 0.))
        axes[8] = 0.
        self.assertEqual(gate.update(axes), (.30, .10, 0.))
        axes[7] = -1.
        self.assertEqual(gate.update(axes), ZERO)  # F retains its zero command.
        axes[7] = 0.
        self.assertEqual(gate.update(axes), (.30, .10, 0.))
        axes[10] = 1.
        self.assertEqual(gate.update(axes), ZERO)  # C returns to the initial gate.
        axes[10], axes[8] = 0., 1.
        self.assertEqual(gate.update(axes), ZERO)


if __name__ == '__main__':
    unittest.main()
