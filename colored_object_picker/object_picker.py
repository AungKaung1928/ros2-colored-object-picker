#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point, Quaternion
from cv_bridge import CvBridge
import cv2
import numpy as np

class PreciseColorObjectPicker(Node):
    def __init__(self):
        super().__init__('precise_color_object_picker')
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error('Cannot open camera')
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.pose_pub = self.create_publisher(Pose, '/detected_object_pose', 10)
        self.timer = self.create_timer(0.1, self.capture_and_process)
        self.frame_count = 0
        
        # Color ranges, display colors, min areas, and messages
        self.color_ranges = {
            'yellow': [(np.array([18, 120, 180]), np.array([32, 255, 255])), (np.array([15, 100, 200]), np.array([35, 255, 255]))],
            'pink': [(np.array([140, 50, 180]), np.array([170, 150, 255]))],
            'blue': [(np.array([100, 150, 150]), np.array([120, 255, 255]))],
            'red': [(np.array([0, 150, 150]), np.array([10, 255, 255])), (np.array([170, 150, 150]), np.array([180, 255, 255]))],
            'green': [(np.array([50, 150, 100]), np.array([70, 255, 255]))],
            'purple': [(np.array([125, 120, 120]), np.array([140, 255, 255]))]
        }
        self.display_colors = {'yellow': (0, 255, 255), 'pink': (203, 192, 255), 'blue': (255, 0, 0), 'red': (0, 0, 255), 'green': (0, 255, 0), 'purple': (128, 0, 128)}
        self.min_areas = {'yellow': 800, 'pink': 800, 'blue': 1200, 'red': 1200, 'green': 1200, 'purple': 1000}
        self.messages = {'yellow': '🌟⚡ YELLOW SUNSHINE DETECTED! ⚡🌟', 'pink': '🌸💖 PINK BLOSSOM SPOTTED! 💖🌸', 'blue': '🌊💙 OCEAN BLUE FOUND! 💙🌊', 'red': '🔥❤️ FIERY RED DETECTED! ❤️🔥', 'green': '🌿💚 FOREST GREEN LOCKED! 💚🌿', 'purple': '🔮💜 MYSTIC PURPLE DISCOVERED! 💜🔮'}
        
        self.get_logger().info('6-Color Object Picker Node Started\nDetecting: Yellow, Pink, Blue, Red, Green, Purple')

    def capture_and_process(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to capture frame')
            return
        
        try:
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            img_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = 'camera_link'
            self.image_pub.publish(img_msg)
            
            detected_objects = self.detect_colored_objects(frame)
            
            if detected_objects:
                for color_name, center, area in detected_objects:
                    self.pose_pub.publish(self.pixel_to_world(center))
                    self.get_logger().info(self.messages[color_name])
                    
                    # Draw detection
                    display_color = self.display_colors[color_name]
                    cv2.circle(frame, center, 15, display_color, 4)
                    cv2.circle(frame, center, 6, (255, 255, 255), -1)
                    
                    text = f'{color_name.upper()}'
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(frame, (center[0] - text_size[0]//2, center[1] - 35), (center[0] + text_size[0]//2, center[1] - 10), (0, 0, 0), -1)
                    cv2.putText(frame, text, (center[0] - text_size[0]//2, center[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, display_color, 2)
            else:
                self.frame_count += 1
                if self.frame_count % 50 == 0:
                    self.get_logger().info('No target colors detected')
            
            cv2.imshow('6-Color Detection', frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')

    def detect_colored_objects(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detected_objects = []
        
        for color_name, ranges in self.color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
            
            kernel = np.ones((7,7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.medianBlur(mask, 5)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > self.min_areas.get(color_name, 1000):
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        
                        if self.validate_color(hsv, (cx, cy), color_name):
                            detected_objects.append((color_name, (cx, cy), area))
                            cv2.imshow(f'{color_name.upper()} Mask', mask)
        
        return sorted(detected_objects, key=lambda x: x[2], reverse=True)

    def validate_color(self, hsv_image, center, color_name):
        cx, cy = center
        if not (0 <= cx < hsv_image.shape[1] and 0 <= cy < hsv_image.shape[0]):
            return False
        
        # Center pixel validation
        pixel_hsv = hsv_image[cy, cx]
        center_valid = any(np.all(pixel_hsv >= lower) and np.all(pixel_hsv <= upper) for lower, upper in self.color_ranges[color_name])
        
        # Region validation
        region_size = 12
        valid_pixels = total_pixels = 0
        for y in range(max(0, cy-region_size//2), min(hsv_image.shape[0], cy+region_size//2)):
            for x in range(max(0, cx-region_size//2), min(hsv_image.shape[1], cx+region_size//2)):
                pixel_hsv = hsv_image[y, x]
                total_pixels += 1
                if any(np.all(pixel_hsv >= lower) and np.all(pixel_hsv <= upper) for lower, upper in self.color_ranges[color_name]):
                    valid_pixels += 1
        
        region_valid = (valid_pixels / total_pixels) > 0.65 if total_pixels > 0 else False
        
        # Extra validation for similar colors
        if color_name in ['pink', 'blue', 'purple']:
            h, s, v = pixel_hsv
            if color_name == 'pink':
                precise_valid = 140 <= h <= 170 and 50 <= s <= 150 and v > 180
            elif color_name == 'blue':
                precise_valid = 100 <= h <= 120 and s > 150 and v > 150
            else:  # purple
                precise_valid = 125 <= h <= 140 and s > 120 and v > 120
            return center_valid and region_valid and precise_valid
        
        return center_valid and region_valid

    def pixel_to_world(self, pixel_coords):
        x, y = pixel_coords
        pose = Pose()
        pose.position = Point(x=(x - 320) * 0.001, y=(y - 240) * 0.001, z=0.1)
        pose.orientation = Quaternion(w=1.0)
        return pose

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = PreciseColorObjectPicker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()