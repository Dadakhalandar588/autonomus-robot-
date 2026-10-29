
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image


class ImageCapture(Node):

    def __init__(self):
        super().__init__('image_capture')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('output_dir', os.path.expanduser('~/amr/captures'))
        self.declare_parameter('interval', 0.5)      # seconds between pictures
        self.declare_parameter('max_images', 0)      # 0 = unlimited
        self.declare_parameter('jpg_quality', 95)

        self.topic = self.get_parameter('image_topic').value
        self.out = self.get_parameter('output_dir').value
        self.interval = float(self.get_parameter('interval').value)
        self.max_images = int(self.get_parameter('max_images').value)
        self.quality = int(self.get_parameter('jpg_quality').value)

        os.makedirs(self.out, exist_ok=True)

        # Continue numbering if the folder already has pictures
        existing = [f for f in os.listdir(self.out) if f.endswith('.jpg')]
        self.count = len(existing)
        self.start_count = self.count
        self.last_save = 0.0

        # Camera publishes best effort, so use the sensor-data QoS
        self.create_subscription(
            Image, self.topic, self.image_cb, qos_profile_sensor_data)

        self.get_logger().info(
            f'Listening on {self.topic} | saving to {self.out} | '
            f'every {self.interval}s | max_images={self.max_images or "unlimited"}')

    def image_cb(self, msg):
        now = time.time()
        if now - self.last_save < self.interval:
            return

        channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}
        if msg.encoding not in channels:
            self.get_logger().warning(
                f'Unsupported encoding: {msg.encoding}',
                throttle_duration_sec=5.0)
            return

        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, channels[msg.encoding])

        # Convert to BGR, which OpenCV expects when saving
        if msg.encoding == 'rgb8':
            frame = frame[:, :, ::-1]
        elif msg.encoding == 'rgba8':
            frame = frame[:, :, [2, 1, 0]]
        elif msg.encoding == 'bgra8':
            frame = frame[:, :, :3]
        elif msg.encoding == 'mono8':
            frame = np.repeat(frame, 3, axis=2)
        frame = np.ascontiguousarray(frame)

        path = os.path.join(self.out, f'img_{int(now * 1000)}.jpg')
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])

        self.last_save = now
        self.count += 1
        saved = self.count - self.start_count
        if saved % 25 == 0:
            self.get_logger().info(f'Saved {saved} pictures (total {self.count})')

        if self.max_images and saved >= self.max_images:
            self.get_logger().info(f'Reached {self.max_images} pictures. Done.')
            raise SystemExit


def main():
    rclpy.init()
    node = ImageCapture()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.get_logger().info(
            f'Stopped. {node.count - node.start_count} pictures saved to {node.out}')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()