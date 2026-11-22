"""Launch test: the installed detector_node on an image topic (no camera device, no display).

The test process publishes synthetic bgr8 frames and reads /detected_object_pose:
a black frame gives no pose; a 60 x 60 px yellow square centred at pixel (480, 120) in a
640 x 480 frame gives one pose at the image-plane offset pixel_to_world documents
(1 mm per pixel from the centre, fixed z 0.1): (0.16, -0.12, 0.1) in camera_link.
Run: launch_test test/test_launch_detector.py (after sourcing the workspace).
"""
import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

W, H = 640, 480


@pytest.mark.launch_test
def generate_test_description():
    node = launch_ros.actions.Node(
        package='colored_object_picker', executable='detector_node', output='screen',
        parameters=[{'image_topic': '/test/image_raw', 'show_visualization': False}])
    return launch.LaunchDescription([node, launch_testing.actions.ReadyToTest()]), {'node': node}


def frame_msg(square_centre=None):
    img = np.zeros((H, W, 3), np.uint8)
    if square_centre:
        cx, cy = square_centre
        img[cy - 30:cy + 30, cx - 30:cx + 30] = (0, 255, 255)  # yellow, BGR
    msg = Image(height=H, width=W, encoding='bgr8', is_bigendian=0, step=W * 3, data=img.tobytes())
    msg.header.frame_id = 'test_camera'
    return msg


class TestDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.probe = rclpy.create_node('detector_probe')
        self.poses = []
        self.probe.create_subscription(PoseStamped, '/detected_object_pose', self.poses.append, 10)
        self.pub = self.probe.create_publisher(Image, '/test/image_raw', qos_profile_sensor_data)

    def tearDown(self):
        self.probe.destroy_node()

    def publish_for(self, msg, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.pub.publish(msg)
            rclpy.spin_once(self.probe, timeout_sec=0.1)

    def test_black_frame_then_yellow_square(self):
        # Wait for the node to subscribe, then check a black frame yields nothing.
        start = time.monotonic()
        while self.pub.get_subscription_count() == 0 and time.monotonic() - start < 20.0:
            rclpy.spin_once(self.probe, timeout_sec=0.1)
        self.publish_for(frame_msg(), 2.0)
        self.assertEqual(self.poses, [])

        start = time.monotonic()
        while not self.poses and time.monotonic() - start < 10.0:
            self.publish_for(frame_msg((480, 120)), 0.3)
        self.assertTrue(self.poses, 'no /detected_object_pose for the yellow square')
        p = self.poses[-1]
        self.assertEqual(p.header.frame_id, 'camera_link')
        self.assertAlmostEqual(p.pose.position.x, 0.16, delta=0.003)
        self.assertAlmostEqual(p.pose.position.y, -0.12, delta=0.003)
        self.assertAlmostEqual(p.pose.position.z, 0.1, delta=1e-6)
