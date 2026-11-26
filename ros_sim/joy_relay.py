"""Forward the Dex-RL /sbus_data Joy messages across the local ABI boundary."""
import os
import socket
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Joy

from ros_sim.joy import valid_joy
from ros_sim.joy_pipe import JOY_SAMPLE, READY, STOP


def main():
    parent = os.getppid()
    with socket.socket(fileno=int(sys.argv[1])) as sock:
        sock.setblocking(False)
        rclpy.init(args=[])
        node = rclpy.create_node('dex_isaac_joy_relay')

        def publish(msg):
            if valid_joy(msg.axes):
                try:
                    sock.send(JOY_SAMPLE.pack(time.monotonic(), *msg.axes[:12]))
                except (BlockingIOError, ConnectionError):
                    pass

        node.create_subscription(Joy, '/sbus_data', publish, 10)
        sock.send(READY)
        try:
            while rclpy.ok() and os.getppid() == parent:
                rclpy.spin_once(node, timeout_sec=.1)
                try:
                    if sock.recv(len(STOP)) == STOP:
                        break
                except BlockingIOError:
                    pass
        except ExternalShutdownException:
            pass
        finally:
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == '__main__':
    main()
