"""Compact physical-unit controls for the Dex WBC simulator."""
import argparse
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont

from rclpy.executors import SingleThreadedExecutor
from rclpy.utilities import remove_ros_args
from dex_rl import gui_monitor

from joy import AXES, ISAAC_FORWARD_LIMIT, LIMITS, speed_levels, speed_to_stick


class WbcTeleopGui(gui_monitor.TeleopGuiApp):
    AXIS_WIDGETS = tuple(zip(AXES, ('前后 Y1', '横移 X1', '转向 X2')))
    UNITS = ('m/s', 'm/s', 'rad/s')

    def __init__(self, *, isaac=False):
        self.isaac = isaac
        super().__init__()

    def _build_layout(self):
        self.root.title('Dex WBC Teleop · Isaac Sim' if self.isaac else 'Dex WBC Teleop')
        self.root.resizable(False, False)
        style = ttk.Style(self.root)
        style.theme_use('clam')
        # Xft can apply the desktop DPI multiplier even to pixel-sized Tk fonts.
        probe = tkfont.Font(root=self.root, family='sans-serif', size=-16)
        body_size = min(16, max(1, round(16 * 19 / probe.metrics('linespace'))))
        body_font = ('sans-serif', -body_size)
        style.configure('TLabel', font=body_font)
        style.configure('TCheckbutton', font=body_font)
        style.configure('TButton', font=body_font, padding=(8, 4))
        style.configure('Card.TFrame', background='#ffffff')
        style.configure('Axis.TLabel', background='#ffffff', foreground='#17283a')
        style.configure('Range.TLabel', background='#ffffff', foreground='#43556b')
        style.configure('Value.TLabel', background='#ffffff', foreground='#154d9e',
                        font=('DejaVu Sans Mono', -body_size, 'bold'))

        main = ttk.Frame(self.root, padding=(12, 10))
        main.pack()
        ttk.Label(main, text='速度指令', font=('sans-serif', -round(body_size*1.125), 'bold')).grid(
            row=0, column=0, sticky='w')
        ttk.Label(main, text='每档 0.05').grid(row=0, column=1, sticky='e')

        self.axis_scales = {}
        self.speed_ticks = {}
        for axis in range(len(self.AXIS_WIDGETS)):
            self._build_axis(main, axis)

        controls = ttk.Frame(main)
        controls.grid(row=4, column=0, columnspan=2, sticky='ew', pady=(8, 4))
        self.button_vars = {}
        for column, (key, label) in enumerate((('d', 'D 准备'), ('a', 'A 行走'), ('c', 'C'))):
            var = tk.BooleanVar(master=self.root)
            self.button_vars[key] = var
            ttk.Checkbutton(controls, text=label, variable=var,
                            command=lambda name=key, state=var:
                            self.teleop_node.set_button_state(name, state.get())).grid(
                                row=0, column=column, sticky='w', padx=(0, 10))
        self.pause_var = tk.BooleanVar(master=self.root)
        ttk.Checkbutton(controls, text='F 暂停', variable=self.pause_var,
                        command=lambda: self.teleop_node.set_switch_state(
                            'f', 'up' if self.pause_var.get() else 'mid')).grid(
                                row=0, column=3, sticky='w')
        ttk.Label(main, text='勾选 D → 取消 D → 勾选 A').grid(row=5, column=0, sticky='w', padx=(0, 10), pady=(4, 0))
        actions = ttk.Frame(main)
        actions.grid(row=5, column=1, sticky='e', pady=(4, 0))
        ttk.Button(actions, text='松开按键', width=0, command=self._release_buttons).pack(side='left', padx=(0, 6))
        ttk.Button(actions, text='滑杆回零', width=0, command=self._center_axes).pack(side='left')

    def _build_axis(self, parent, axis):
        key, name = self.AXIS_WIDGETS[axis]
        unit = self.UNITS[axis]
        forward_limit = ISAAC_FORWARD_LIMIT if self.isaac else LIMITS[0]
        levels = speed_levels(axis, forward_limit=forward_limit)
        zero = levels.index(0.)
        card = ttk.Frame(parent, padding=(10, 7), style='Card.TFrame')
        card.grid(row=axis+1, column=0, columnspan=2, sticky='ew', pady=(6, 0))
        card.columnconfigure(1, weight=1)
        ttk.Label(card, text=name, style='Axis.TLabel').grid(row=0, column=0, columnspan=2, sticky='w')
        readout = ttk.Label(card, style='Value.TLabel')
        readout.grid(row=0, column=2, sticky='e')
        ttk.Label(card, text=f'{levels[0]:+.2f}', style='Range.TLabel').grid(row=1, column=0, sticky='w')
        ttk.Label(card, text=f'{levels[-1]:+.2f}', style='Range.TLabel').grid(row=1, column=2, sticky='e')
        tick = tk.IntVar(master=self.root, value=0)
        scale = tk.Scale(card, from_=-zero, to=len(levels)-zero-1, orient=tk.HORIZONTAL,
                         resolution=1, showvalue=False, variable=tick,
                         length=268, width=10, sliderlength=18, bd=1,
                         background='#ffffff', activebackground='#d6e8ff',
                         troughcolor='#e1e8ef', highlightbackground='#ffffff',
                         highlightcolor='#154d9e', highlightthickness=1)
        scale.grid(row=1, column=1, sticky='ew', padx=8)
        self.speed_ticks[key] = tick
        self.axis_scales[key] = scale

        def update_speed(*_args):
            speed = levels[zero + tick.get()]
            readout.configure(text=f'{speed:+.2f} {unit}')
            self.teleop_node.set_axis_override(key, speed_to_stick(axis, speed))

        tick.trace_add('write', update_speed)
        update_speed()

    def _refresh(self):
        """The inherited timer still closes the GUI when ROS shuts down."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", action="store_true", help="Use the Isaac Sim speed range")
    args = parser.parse_args(remove_ros_args()[1:])
    # The Humble multithreaded GUI executor can stall the Joy publication timer.
    gui_monitor.MultiThreadedExecutor = SingleThreadedExecutor
    WbcTeleopGui(isaac=args.isaac).run()


if __name__ == '__main__':
    main()
