#!/usr/bin/env python3
"""D5: take the YOLO detection, use the lidar for distance, send a Nav2 goal.

Chain:  /command arms the node
        -> /camera/yolo/detections gives center_x of the target
        -> bearing from camera_info (fx, cx)
        -> bearing is turned into a lidar angle through TF
        -> /scan beams at that angle give the range
        -> point is transformed to the map frame
        -> goal = standoff metres before the object, facing it
        -> NavigateToPose

  ros2 run amr goal_from_lidar --ros-args -p use_sim_time:=true
"""
import json
import math

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, LaserScan
from std_msgs.msg import String
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transforms)
from tf2_ros import Buffer, TransformListener


class GoalFromLidar(Node):
    def __init__(self):
        super().__init__('goal_from_lidar')
        self.declare_parameter('detections_topic', '/camera/yolo/detections')
        self.declare_parameter('command_topic', '/command')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('standoff', 0.5)     # stop this far from object (m)
        self.declare_parameter('beam_window', 3)    # +/- beams around bearing
        self.declare_parameter('min_conf', 0.5)

        g = self.get_parameter
        self.map_frame = g('map_frame').value
        self.standoff = float(g('standoff').value)
        self.window = int(g('beam_window').value)
        self.min_conf = float(g('min_conf').value)

        self.fx = self.cx = None
        self.scan = None
        self.armed = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.create_subscription(
            CameraInfo, g('camera_info_topic').value, self.on_info, 10)
        self.create_subscription(
            LaserScan, g('scan_topic').value, self.on_scan, qos_profile_sensor_data)
        self.create_subscription(
            String, g('command_topic').value, self.on_command, 10)
        self.create_subscription(
            String, g('detections_topic').value, self.on_det, 10)
        self.get_logger().info('goal_from_lidar ready. Send a sentence to /command')

    def on_info(self, msg):
        self.fx = msg.k[0]
        self.cx = msg.k[2]

    def on_scan(self, msg):
        self.scan = msg

    def on_command(self, msg):
        self.armed = True
        self.get_logger().info(f"Armed by command '{msg.data}'")

    # ------------------------------------------------------------------
    def on_det(self, msg):
        if not self.armed or self.fx is None or self.scan is None:
            return
        data = json.loads(msg.data)
        target = data.get('target')
        dets = [d for d in data.get('detections', [])
                if d['class'] == target and d['conf'] >= self.min_conf]
        if not dets:
            return
        det = max(dets, key=lambda d: d['conf'])
        try:
            goal = self.make_goal(det['center_x'], data['frame_id'])
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'Could not build goal: {e}',
                                   throttle_duration_sec=2.0)
            return
        if goal is None:
            return
        self.armed = False
        self.send_goal(goal, target)

    def make_goal(self, center_x, cam_frame):
        scan = self.scan
        scan_frame = scan.header.frame_id
        timeout = Duration(seconds=0.5)

        # 1) bearing in camera frame (image right = negative, ROS convention)
        b = -math.atan2(center_x - self.cx, self.fx)

        # 2) same direction expressed in the lidar frame
        def cam_pt(x, y):
            p = PointStamped()
            p.header.frame_id = cam_frame
            p.header.stamp = Time().to_msg()   # latest transform
            p.point.x, p.point.y = x, y
            return self.tf_buffer.transform(p, scan_frame, timeout)

        o = cam_pt(0.0, 0.0)
        d = cam_pt(math.cos(b), math.sin(b))
        angle = math.atan2(d.point.y - o.point.y, d.point.x - o.point.x)

        # 3) lidar beams at that angle, nearest valid one
        i0 = round((angle - scan.angle_min) / scan.angle_increment)
        valid = []
        for i in range(i0 - self.window, i0 + self.window + 1):
            if 0 <= i < len(scan.ranges):
                r = scan.ranges[i]
                if math.isfinite(r) and scan.range_min < r < scan.range_max:
                    valid.append(r)
        if not valid:
            self.get_logger().warn('No valid lidar beam at the object bearing',
                                   throttle_duration_sec=2.0)
            return None
        r = min(valid)

        # 4) object and robot positions in the map frame
        def scan_pt(x, y):
            p = PointStamped()
            p.header.frame_id = scan_frame
            p.header.stamp = Time().to_msg()
            p.point.x, p.point.y = x, y
            return self.tf_buffer.transform(p, self.map_frame, timeout).point

        obj = scan_pt(r * math.cos(angle), r * math.sin(angle))
        rob = scan_pt(0.0, 0.0)

        dx, dy = obj.x - rob.x, obj.y - rob.y
        dist = math.hypot(dx, dy)
        yaw = math.atan2(dy, dx)
        if dist <= self.standoff + 0.1:
            self.get_logger().info(f'Already near the object ({dist:.2f} m)')
            self.armed = False
            return None

        # 5) stop `standoff` metres before the object, facing it
        gx = obj.x - self.standoff * dx / dist
        gy = obj.y - self.standoff * dy / dist
        goal = PoseStamped()
        goal.header.frame_id = self.map_frame
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = gx
        goal.pose.position.y = gy
        goal.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.orientation.w = math.cos(yaw / 2)
        self.get_logger().info(
            f'Object range {r:.2f} m at map ({obj.x:.2f}, {obj.y:.2f}) '
            f'-> goal ({gx:.2f}, {gy:.2f})')
        return goal

    # ------------------------------------------------------------------
    def send_goal(self, pose, target):
        if not self.nav.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('navigate_to_pose action server not available')
            return
        goal = NavigateToPose.Goal()
        goal.pose = pose
        self.get_logger().info(f"Sending goal near '{target}' to Nav2")
        self.nav.send_goal_async(goal).add_done_callback(self.on_accept)

    def on_accept(self, fut):
        handle = fut.result()
        if not handle.accepted:
            self.get_logger().error('Nav2 rejected the goal')
            return
        handle.get_result_async().add_done_callback(
            lambda f: self.get_logger().info(f'Nav2 finished, status {f.result().status}'))


def main(args=None):
    rclpy.init(args=args)
    node = GoalFromLidar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()