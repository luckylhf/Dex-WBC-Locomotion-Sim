"""Dex-RL Joy mapping, valid slider speeds and interactive D/A start gate."""
import math

ZERO = (0., 0., 0.)
LIMITS = (1.2, .3, .45)
BACK_LIMIT = .3
ISAAC_FORWARD_LIMIT = .4


def valid_joy(axes):
    return len(axes) >= 12 and all(math.isfinite(x) for x in axes[:12])


def velocity(axes):
    if not valid_joy(axes):
        return ZERO
    if axes[10] > .5 or axes[7] < -.5:
        return ZERO
    # Decode each Dex-RL motion axis independently. Positive vx is forward,
    # positive vy is left and positive yaw is a left turn. Y2 is unused.
    sticks = (axes[2], -axes[3], -axes[0])
    result = []
    for i, raw_stick in enumerate(sticks):
        stick = min(1., max(-1., raw_stick))
        scale = BACK_LIMIT if i == 0 and stick < 0 else LIMITS[i]
        result.append(0. if abs(stick) < .05 else round(stick * scale, 2))
    # Suppress sub-minimum forward commands without clearing the other axes.
    if 0 < result[0] < .10:
        result[0] = 0.
    return tuple(result)


AXES = ('y1', 'x1', 'turn')
STEP_SIZE = .05


def speed_levels(axis, *, forward_limit=LIMITS[0]):
    """The GUI fills each axis only with speeds the controller can request."""
    reverse = BACK_LIMIT if axis == 0 else LIMITS[axis]
    first = -round(reverse / STEP_SIZE)
    upper = forward_limit if axis == 0 else LIMITS[axis]
    last = round(upper / STEP_SIZE)
    return tuple(round(step * STEP_SIZE, 2) for step in range(first, last + 1)
                 if axis != 0 or step != 1)


def speed_to_stick(axis, speed):
    limit = BACK_LIMIT if axis == 0 and speed < 0 else LIMITS[axis]
    return speed / limit if axis == 0 else -speed / limit


class StartGate:
    """Arm motion only after D, release D, then a new press of A."""

    def __init__(self):
        self.prepared = False
        self.armed = False
        self.a_pressed = False

    def update(self, axes):
        if not valid_joy(axes):
            return ZERO
        a_pressed = axes[8] > .5
        if axes[10] > .5:
            self.prepared = self.armed = False
        elif axes[11] > .5:
            self.armed = False
            if not a_pressed:
                self.prepared = True
        elif a_pressed and not self.a_pressed and self.prepared and axes[7] >= -.5:
            self.armed = True
            self.prepared = False
        self.a_pressed = a_pressed
        return velocity(axes) if self.armed else ZERO
