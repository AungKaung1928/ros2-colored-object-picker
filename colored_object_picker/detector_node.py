#!/usr/bin/env python3
"""
ROS2 node wrapper for color detection.
Handles camera capture, ROS publishing, and visualization.
"""

from typing import Dict, Optional
import cv2
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from .color_detector import (
    ColorDetector,
    ColorConfig,
    DetectedObject,
    load_color_configs_from_dict
)


class ColorDetectorNode(Node):
    """ROS2 node for real-time color object detection."""
    
    def __init__(self) -> None:
        super().__init__('color_detector_node')
        
        # Declare parameters
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('update_rate', 10.0)
        self.declare_parameter('show_visualization', True)
        self.declare_parameter('show_masks', False)
        self.declare_parameter('config_file', '')
        # Non-empty: subscribe to this image topic instead of opening a camera device.
        # Lets the node run on a sim camera or a bag, and be tested without hardware.
        self.declare_parameter('image_topic', '')
        self.declare_parameter('camera_frame', 'camera_link')
        
        # Get parameters
        camera_id = self.get_parameter('camera_id').value
        self.frame_width = self.get_parameter('frame_width').value
        self.frame_height = self.get_parameter('frame_height').value
        update_rate = self.get_parameter('update_rate').value
        self.show_visualization = self.get_parameter('show_visualization').value
        self.show_masks = self.get_parameter('show_masks').value
        config_file = self.get_parameter('config_file').value
        image_topic = self.get_parameter('image_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        
        # Load color configuration
        color_configs = self._load_config(config_file)
        if not color_configs:
            self.get_logger().error('Failed to load color configuration')
            return
        
        # Initialize detector
        self.detector = ColorDetector(color_configs)
        self.color_configs = color_configs
        
        self.bridge = CvBridge()
        self.cap = None
        self.image_sub = None
        self.frame_count = 0
        self.pose_pub = self.create_publisher(PoseStamped, '/detected_object_pose', 10)

        if image_topic:
            sensor_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
            self.image_sub = self.create_subscription(
                Image, image_topic, self._on_image, sensor_qos)
            self.image_pub = None
            source = f'topic {image_topic}'
        else:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                self.get_logger().error(f'Cannot open camera {camera_id}')
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
            self.timer = self.create_timer(1.0 / update_rate, self.process_frame)
            source = f'camera {camera_id} ({self.frame_width}x{self.frame_height}) at {update_rate} Hz'

        color_names = ', '.join(color_configs.keys())
        self.get_logger().info(f'Color detector started: {source}; detecting {color_names}')
    
    def _load_config(self, config_file: str) -> Optional[Dict[str, ColorConfig]]:
        """
        Load color configuration from YAML file.
        
        Args:
            config_file: Path to config file, or empty for default
            
        Returns:
            Dictionary of ColorConfig objects, or None on failure
        """
        if not config_file:
            # Try default location
            try:
                package_share = get_package_share_directory('colored_object_picker')
                config_file = f'{package_share}/config/colors.yaml'
            except Exception:
                self.get_logger().warn(
                    'Could not find package share directory, using default config'
                )
                return self._get_default_config()
        
        try:
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            self.get_logger().info(f'Loaded config from {config_file}')
            return load_color_configs_from_dict(config_dict)
        except Exception as e:
            self.get_logger().warn(f'Failed to load config file: {e}')
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, ColorConfig]:
        """Return hardcoded default configuration."""
        import numpy as np
        
        return {
            'yellow': ColorConfig(
                name='yellow',
                hsv_ranges=[
                    (np.array([18, 120, 180]), np.array([32, 255, 255])),
                    (np.array([15, 100, 200]), np.array([35, 255, 255]))
                ],
                min_area=800,
                display_color=(0, 255, 255),
                message='🌟⚡ YELLOW SUNSHINE DETECTED! ⚡🌟'
            ),
            'pink': ColorConfig(
                name='pink',
                hsv_ranges=[
                    (np.array([140, 50, 180]), np.array([170, 150, 255]))
                ],
                min_area=800,
                display_color=(203, 192, 255),
                message='🌸💖 PINK BLOSSOM SPOTTED! 💖🌸'
            ),
            'blue': ColorConfig(
                name='blue',
                hsv_ranges=[
                    (np.array([100, 150, 150]), np.array([120, 255, 255]))
                ],
                min_area=1200,
                display_color=(255, 0, 0),
                message='🌊💙 OCEAN BLUE FOUND! 💙🌊'
            ),
            'red': ColorConfig(
                name='red',
                hsv_ranges=[
                    (np.array([0, 150, 150]), np.array([10, 255, 255])),
                    (np.array([170, 150, 150]), np.array([180, 255, 255]))
                ],
                min_area=1200,
                display_color=(0, 0, 255),
                message='🔥❤️ FIERY RED DETECTED! ❤️🔥'
            ),
            'green': ColorConfig(
                name='green',
                hsv_ranges=[
                    (np.array([50, 150, 100]), np.array([70, 255, 255]))
                ],
                min_area=1200,
                display_color=(0, 255, 0),
                message='🌿💚 FOREST GREEN LOCKED! 💚🌿'
            ),
            'purple': ColorConfig(
                name='purple',
                hsv_ranges=[
                    (np.array([125, 120, 120]), np.array([140, 255, 255]))
                ],
                min_area=1000,
                display_color=(128, 0, 128),
                message='🔮💜 MYSTIC PURPLE DISCOVERED! 💜🔮'
            )
        }
    
    def _on_image(self, msg: Image) -> None:
        """Image-topic source."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:  # cv_bridge encoding mismatch
            self.get_logger().error(f'Cannot convert image: {e}', throttle_duration_sec=5.0)
            return
        self.frame_height, self.frame_width = frame.shape[:2]
        self._process(frame, msg.header.stamp)

    def process_frame(self) -> None:
        """Camera-device source: capture one frame, republish it, process it."""
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to capture frame')
            return
        stamp = self.get_clock().now().to_msg()
        img_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = self.camera_frame
        self.image_pub.publish(img_msg)
        self._process(frame, stamp)

    def _process(self, frame, stamp) -> None:
        """Detect, publish poses, draw."""
        try:
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

            # Detect objects
            detected_objects, masks = self.detector.detect_objects(
                frame, return_masks=self.show_masks
            )
            
            if detected_objects:
                self._handle_detections(detected_objects, stamp)
                
                if self.show_masks and masks:
                    for color_name, mask in masks.items():
                        cv2.imshow(f'{color_name.upper()} Mask', mask)
            else:
                self.frame_count += 1
                if self.frame_count % 50 == 0:
                    self.get_logger().info('No target colors detected')
            
            # Show visualization
            if self.show_visualization:
                display_frame = self.detector.draw_detections(frame, detected_objects)
                cv2.imshow('Color Detection', display_frame)
                cv2.waitKey(1)
                
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')
    
    def _handle_detections(self, detected_objects: list[DetectedObject], stamp) -> None:
        """Publish one PoseStamped per detection (image-plane offset, see pixel_to_world)."""
        for obj in detected_objects:
            world_pose = self.detector.pixel_to_world(
                obj.centroid, self.frame_width, self.frame_height)
            pose_msg = PoseStamped()
            pose_msg.header.stamp = stamp
            pose_msg.header.frame_id = self.camera_frame
            pose_msg.pose.position = Point(x=world_pose.x, y=world_pose.y, z=world_pose.z)
            pose_msg.pose.orientation = Quaternion(
                w=world_pose.qw, x=world_pose.qx, y=world_pose.qy, z=world_pose.qz)
            self.pose_pub.publish(pose_msg)
            
            # Log detection message
            config = self.color_configs.get(obj.color_name)
            if config:
                self.get_logger().info(config.message)
    
    def destroy_node(self) -> None:
        """Clean up resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None) -> None:
    """Main entry point."""
    rclpy.init(args=args)
    node = ColorDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()