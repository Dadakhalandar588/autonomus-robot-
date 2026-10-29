#!/usr/bin/env python3
"""YOLO detector node for the AMR Gazebo simulation.

Uses YOUR fine-tuned model (classes: box, chair, table), draws the bounding
boxes, and shows the detections in three places:
  1. the terminal (class, confidence, box pixels, FPS)
  2. a live window on your screen (show_window:=true)
  3. a ROS image topic /yolo/results (open it in rqt_image_view or RViz)

  ros2 run amr yolo --ros-args -p use_sim_time:=true -p conf:=0.4

Default model: ~/Desktop/amr_gazebo_detector_best.pt
"""
import json
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        default_model = os.path.expanduser('~/Desktop/amr_gazebo_detector_best.pt')
        self.declare_parameter('model', default_model)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/yolo/results')
        self.declare_parameter('detections_topic', '/yolo/detections')
        self.declare_parameter('conf', 0.4)
        self.declare_parameter('imgsz', 960)          # same size the model was trained at
        self.declare_parameter('show_window', True)   # live window on your screen

        model_path = os.path.expanduser(self.get_parameter('model').value)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model not found: {model_path}')

        self.conf = float(self.get_parameter('conf').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.show_window = bool(self.get_parameter('show_window').value)
        image_topic = self.get_parameter('image_topic').value

        self.model = YOLO(model_path)
        self.get_logger().info(f'Loaded model: {model_path}')
        self.get_logger().info(f'Classes: {list(self.model.names.values())}')
        self.get_logger().info(f'Listening on: {image_topic}, conf={self.conf}')

        self.sub = self.create_subscription(
            Image, image_topic, self.on_image, qos_profile_sensor_data)
        self.pub_img = self.create_publisher(
            Image, self.get_parameter('output_topic').value, 10)
        self.pub_det = self.create_publisher(
            String, self.get_parameter('detections_topic').value, 10)

        self.last_time = time.monotonic()
        self.fps = 0.0

    # ---- image conversion without cv_bridge (it crashed inside mujoco_env) ----
    @staticmethod
    def to_bgr(msg: Image):
        enc = msg.encoding
        channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(enc)
        if channels is None:
            return None
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        arr = arr[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
        if enc == 'rgb8':
            arr = arr[:, :, ::-1]
        elif enc == 'rgba8':
            arr = arr[:, :, 2::-1]
        elif enc == 'bgra8':
            arr = arr[:, :, :3]
        elif enc == 'mono8':
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(arr)

    def on_image(self, msg: Image):
        frame = self.to_bgr(msg)
        if frame is None:
            self.get_logger().warn(f'Unsupported image encoding: {msg.encoding}',
                                   throttle_duration_sec=5.0)
            return

        result = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]

        # Draw the bounding boxes and labels
        annotated = np.ascontiguousarray(result.plot())

        # 1) publish the annotated image
        out = Image()
        out.header = msg.header
        out.height, out.width = annotated.shape[:2]
        out.encoding = 'bgr8'
        out.is_bigendian = 0
        out.step = out.width * 3
        out.data = annotated.tobytes()
        self.pub_img.publish(out)

        # 2) machine-readable detections (with the image time, for the chair goal node)
        detections = []
        for b in result.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            detections.append({
                'class': self.model.names[int(b.cls[0])],
                'conf': round(float(b.conf[0]), 3),
                'bbox': [round(x1), round(y1), round(x2), round(y2)],
                'center_x': round((x1 + x2) / 2),
            })
        payload = {
            'stamp': {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec},
            'frame_id': msg.header.frame_id,
            'detections': detections,
        }
        self.pub_det.publish(String(data=json.dumps(payload)))

        # 3) terminal output
        now = time.monotonic()
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - self.last_time, 1e-6))
        self.last_time = now
        if detections:
            lines = ', '.join(
                f"{d['class']} {d['conf']:.2f} {d['bbox']}" for d in detections)
            self.get_logger().info(f'[{self.fps:.1f} FPS] Detected: {lines}')
        else:
            self.get_logger().info(f'[{self.fps:.1f} FPS] No detections',
                                   throttle_duration_sec=2.0)

        # 4) live window
        if self.show_window:
            try:
                cv2.imshow('YOLO detections', annotated)
                cv2.waitKey(1)
            except cv2.error as e:
                self.get_logger().warn(
                    f'Could not open a window ({e}). Turning it off, use /yolo/results instead.')
                self.show_window = False


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()