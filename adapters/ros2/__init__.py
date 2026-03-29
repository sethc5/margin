"""
ROS2 adapter for margin.

Typed health classification for robot sensor streams.
Motor temperature, battery voltage, joint torque, lidar signal,
IMU drift — all classified with correct polarity, drift detection,
and anomaly detection.

Usage without ROS2 (threshold tables + manual readings):
    from adapters.ros2 import parse_robot, robot_expression, ROBOT_PROFILES

Usage with ROS2 (requires rclpy):
    from adapters.ros2.node import MarginNode
"""
from .sensors import (
    ROBOT_PROFILES, RobotSensorProfile, RobotSensor,
    parse_robot, robot_expression,
)
