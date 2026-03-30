"""
ROS2 margin node — subscribes to sensor topics, publishes typed health.

Requires rclpy (ROS2 Python client). Not imported at adapter level
so the sensor profiles work without ROS2 installed.

Usage:
    # In a ROS2 workspace:
    from adapters.ros2.node import MarginNode
    import rclpy

    rclpy.init()
    node = MarginNode(profile="mobile", robot_id="robot_01")
    rclpy.spin(node)

The node:
  - Subscribes to /margin/sensors (JSON string with sensor readings)
  - Publishes /margin/health (JSON string with Expression)
  - Publishes /margin/diagnostics (diagnostic_msgs/DiagnosticArray)
  - Runs drift + anomaly tracking internally
  - Logs state changes
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

# These imports are deferred — only fail when the node is actually instantiated
# so the rest of the adapter works without rclpy
_ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    _ROS2_AVAILABLE = True
except ImportError:
    Node = object  # type: ignore


from margin.streaming import Monitor
from margin.observation import Parser
from margin.health import Thresholds, Health
from margin.confidence import Confidence
from .sensors import ROBOT_PROFILES, RobotSensor


def _build_parser(sensors: dict[str, RobotSensor]) -> Parser:
    """Build a margin Parser from a sensor profile."""
    baselines = {}
    component_thresholds = {}
    for name, sensor in sensors.items():
        baselines[name] = sensor.baseline
        component_thresholds[name] = sensor.thresholds
    # Use first sensor's thresholds as default
    first = next(iter(sensors.values()))
    return Parser(
        baselines=baselines,
        thresholds=first.thresholds,
        component_thresholds=component_thresholds,
    )


# Health → DiagnosticStatus level mapping
_HEALTH_TO_DIAG_LEVEL = {
    "INTACT": 0,       # OK
    "RECOVERING": 1,   # WARN
    "DEGRADED": 1,     # WARN
    "ABLATED": 2,      # ERROR
    "OOD": 3,          # STALE
}


class MarginNode(Node):
    """
    ROS2 node that monitors robot health using margin.

    Subscribes to sensor readings, runs a streaming Monitor,
    publishes health classifications and ROS2 diagnostics.

    Parameters:
        profile:    robot profile name ("mobile", "manipulator", "drone", "agv")
        robot_id:   robot identifier for labeling
        window:     tracker window size (default 100)
        sensors:    override sensor definitions (default: from profile)
        timer_hz:   status publish rate (default 1.0 Hz)
    """

    def __init__(
        self,
        profile: str = "mobile",
        robot_id: str = "robot",
        window: int = 100,
        sensors: Optional[dict[str, RobotSensor]] = None,
        timer_hz: float = 1.0,
    ):
        if not _ROS2_AVAILABLE:
            raise ImportError(
                "rclpy not available. Install ROS2 or run: "
                "pip install rclpy\n"
                "For sensor profiles without ROS2, use: "
                "from adapters.ros2 import parse_robot, robot_expression"
            )

        super().__init__(f"margin_{robot_id}")
        self.robot_id = robot_id
        self.profile_name = profile

        # Resolve sensors
        if sensors is None:
            p = ROBOT_PROFILES.get(profile, ROBOT_PROFILES["mobile"])
            sensors = p.sensors
        self._sensors = sensors

        # Build parser and monitor
        parser = _build_parser(sensors)
        self.monitor = Monitor(parser, window=window)

        # Last known readings (updated by subscriber)
        self._latest_readings: dict[str, float] = {}

        # ── ROS2 interfaces ──

        # Subscriber: sensor readings as JSON
        self._sub_sensors = self.create_subscription(
            String, "/margin/sensors", self._on_sensors, 10,
        )

        # Publisher: health expression as JSON
        self._pub_health = self.create_publisher(String, "/margin/health", 10)

        # Publisher: ROS2 diagnostics (optional — only if diagnostic_msgs available)
        self._pub_diag = None
        try:
            from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
            self._pub_diag = self.create_publisher(DiagnosticArray, "/margin/diagnostics", 10)
            self._DiagnosticArray = DiagnosticArray
            self._DiagnosticStatus = DiagnosticStatus
            self._KeyValue = KeyValue
        except ImportError:
            self.get_logger().info("diagnostic_msgs not available, skipping diagnostics publisher")

        # Timer for periodic status publish
        if timer_hz > 0:
            self._timer = self.create_timer(1.0 / timer_hz, self._publish_status)

        self.get_logger().info(
            f"margin node started: profile={profile}, "
            f"sensors={len(sensors)}, window={window}"
        )

    def _on_sensors(self, msg: 'String') -> None:
        """Handle incoming sensor readings."""
        try:
            readings = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"Invalid JSON on /margin/sensors: {msg.data[:100]}")
            return

        if not isinstance(readings, dict):
            return

        # Update latest readings
        for k, v in readings.items():
            if isinstance(v, (int, float)):
                self._latest_readings[k] = float(v)

        # Run through monitor
        now = datetime.now()
        expr = self.monitor.update(self._latest_readings, now=now, label=self.robot_id)

        # Log health changes
        for obs in expr.observations:
            if obs.health in (Health.DEGRADED, Health.ABLATED):
                self.get_logger().warn(
                    f"{obs.name}: {obs.health.value} (σ={obs.sigma:+.2f})"
                )

        # Log drift warnings
        for name in self._sensors:
            dc = self.monitor.drift(name)
            if dc and dc.worsening:
                self.get_logger().warn(
                    f"drift({name}): {dc.state.value}({dc.direction.value})"
                )

        # Log anomalies
        for name in self._sensors:
            ac = self.monitor.anomaly(name)
            if ac and ac.anomalous:
                self.get_logger().warn(
                    f"anomaly({name}): {ac.state.value} (z={ac.z_score:.2f})"
                )

    def _publish_status(self) -> None:
        """Periodic health publish."""
        if self.monitor.expression is None:
            return

        # Publish health as JSON
        health_msg = String()
        health_data = self.monitor.status()
        health_data["robot_id"] = self.robot_id
        health_data["profile"] = self.profile_name
        health_msg.data = json.dumps(health_data)
        self._pub_health.publish(health_msg)

        # Publish diagnostics
        if self._pub_diag is not None:
            self._publish_diagnostics()

    def _publish_diagnostics(self) -> None:
        """Publish ROS2 DiagnosticArray from current health."""
        expr = self.monitor.expression
        if expr is None:
            return

        diag = self._DiagnosticArray()
        diag.header.stamp = self.get_clock().now().to_msg()

        for obs in expr.observations:
            status = self._DiagnosticStatus()
            status.name = f"margin/{self.robot_id}/{obs.name}"
            status.hardware_id = self.robot_id
            status.level = _HEALTH_TO_DIAG_LEVEL.get(obs.health.value, 3)
            status.message = f"{obs.health.value} (σ={obs.sigma:+.2f})"
            status.values = [
                self._KeyValue(key="health", value=obs.health.value),
                self._KeyValue(key="value", value=str(obs.value)),
                self._KeyValue(key="baseline", value=str(obs.baseline)),
                self._KeyValue(key="sigma", value=f"{obs.sigma:.4f}"),
                self._KeyValue(key="confidence", value=obs.confidence.value),
            ]

            # Add drift info
            dc = self.monitor.drift(obs.name)
            if dc:
                status.values.extend([
                    self._KeyValue(key="drift_state", value=dc.state.value),
                    self._KeyValue(key="drift_direction", value=dc.direction.value),
                ])

            # Add anomaly info
            ac = self.monitor.anomaly(obs.name)
            if ac:
                status.values.extend([
                    self._KeyValue(key="anomaly_state", value=ac.state.value),
                    self._KeyValue(key="anomaly_z", value=f"{ac.z_score:.4f}"),
                ])

            diag.status.append(status)

        self._pub_diag.publish(diag)
