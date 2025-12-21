#!/usr/bin/env python3
"""
Color detection logic module - Pure Python, no ROS dependencies.
Handles HSV color detection, contour analysis, and validation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class ColorConfig:
    """Configuration for a single color detection."""
    name: str
    hsv_ranges: List[Tuple[np.ndarray, np.ndarray]]
    min_area: int
    display_color: Tuple[int, int, int]
    message: str


@dataclass
class DetectedObject:
    """Represents a detected colored object."""
    color_name: str
    centroid: Tuple[int, int]
    area: float
    contour: np.ndarray


@dataclass
class WorldPose:
    """3D pose in world coordinates."""
    x: float
    y: float
    z: float
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0


class ColorDetector:
    """
    Pure color detection logic without ROS dependencies.
    Handles HSV-based color detection with morphological filtering.
    """
    
    def __init__(self, color_configs: Dict[str, ColorConfig]) -> None:
        """
        Initialize the color detector.
        
        Args:
            color_configs: Dictionary mapping color names to ColorConfig objects
        """
        self.color_configs = color_configs
        self.kernel = np.ones((7, 7), np.uint8)
    
    @staticmethod
    def calculate_centroid(contour: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Calculate the centroid of a contour using image moments.
        
        Args:
            contour: OpenCV contour array
            
        Returns:
            Tuple of (cx, cy) pixel coordinates, or None if invalid
        """
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)
    
    @staticmethod
    def pixel_to_world(
        pixel_coords: Tuple[int, int],
        image_width: int = 640,
        image_height: int = 480,
        scale: float = 0.001,
        z_offset: float = 0.1
    ) -> WorldPose:
        """
        Convert pixel coordinates to world coordinates.
        
        Args:
            pixel_coords: (x, y) pixel coordinates
            image_width: Camera image width
            image_height: Camera image height
            scale: Conversion scale factor (meters per pixel)
            z_offset: Fixed z-coordinate offset
            
        Returns:
            WorldPose object with 3D coordinates
        """
        x, y = pixel_coords
        world_x = (x - image_width / 2) * scale
        world_y = (y - image_height / 2) * scale
        return WorldPose(x=world_x, y=world_y, z=z_offset)
    
    def create_color_mask(
        self,
        hsv_image: np.ndarray,
        color_name: str
    ) -> np.ndarray:
        """
        Create a binary mask for a specific color.
        
        Args:
            hsv_image: Image in HSV color space
            color_name: Name of the color to detect
            
        Returns:
            Binary mask with detected color regions
        """
        config = self.color_configs.get(color_name)
        if config is None:
            return np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for lower, upper in config.hsv_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv_image, lower, upper))
        
        # Morphological operations to clean up mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.medianBlur(mask, 5)
        
        return mask
    
    def validate_color_at_point(
        self,
        hsv_image: np.ndarray,
        center: Tuple[int, int],
        color_name: str,
        region_size: int = 12,
        threshold: float = 0.65
    ) -> bool:
        """
        Validate that a point actually contains the expected color.
        
        Args:
            hsv_image: Image in HSV color space
            center: (cx, cy) point to validate
            color_name: Expected color name
            region_size: Size of region to sample around center
            threshold: Minimum ratio of valid pixels required
            
        Returns:
            True if color is validated, False otherwise
        """
        cx, cy = center
        height, width = hsv_image.shape[:2]
        
        # Bounds check
        if not (0 <= cx < width and 0 <= cy < height):
            return False
        
        config = self.color_configs.get(color_name)
        if config is None:
            return False
        
        # Center pixel validation
        pixel_hsv = hsv_image[cy, cx]
        center_valid = any(
            np.all(pixel_hsv >= lower) and np.all(pixel_hsv <= upper)
            for lower, upper in config.hsv_ranges
        )
        
        # Region validation
        half_size = region_size // 2
        y_start = max(0, cy - half_size)
        y_end = min(height, cy + half_size)
        x_start = max(0, cx - half_size)
        x_end = min(width, cx + half_size)
        
        valid_pixels = 0
        total_pixels = 0
        
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                pixel_hsv = hsv_image[y, x]
                total_pixels += 1
                if any(
                    np.all(pixel_hsv >= lower) and np.all(pixel_hsv <= upper)
                    for lower, upper in config.hsv_ranges
                ):
                    valid_pixels += 1
        
        region_valid = (valid_pixels / total_pixels) > threshold if total_pixels > 0 else False
        
        # Extra validation for similar colors
        if color_name in ['pink', 'blue', 'purple']:
            h, s, v = hsv_image[cy, cx]
            if color_name == 'pink':
                precise_valid = 140 <= h <= 170 and 50 <= s <= 150 and v > 180
            elif color_name == 'blue':
                precise_valid = 100 <= h <= 120 and s > 150 and v > 150
            else:  # purple
                precise_valid = 125 <= h <= 140 and s > 120 and v > 120
            return center_valid and region_valid and precise_valid
        
        return center_valid and region_valid
    
    def detect_objects(
        self,
        image: np.ndarray,
        return_masks: bool = False
    ) -> Tuple[List[DetectedObject], Optional[Dict[str, np.ndarray]]]:
        """
        Detect all colored objects in an image.
        
        Args:
            image: BGR image from camera
            return_masks: Whether to return individual color masks
            
        Returns:
            Tuple of (list of DetectedObject, optional dict of masks)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detected_objects: List[DetectedObject] = []
        masks: Dict[str, np.ndarray] = {} if return_masks else {}
        
        for color_name, config in self.color_configs.items():
            mask = self.create_color_mask(hsv, color_name)
            
            if return_masks:
                masks[color_name] = mask
            
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                continue
            
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area <= config.min_area:
                continue
            
            centroid = self.calculate_centroid(largest_contour)
            if centroid is None:
                continue
            
            if not self.validate_color_at_point(hsv, centroid, color_name):
                continue
            
            detected_objects.append(DetectedObject(
                color_name=color_name,
                centroid=centroid,
                area=area,
                contour=largest_contour
            ))
        
        # Sort by area (largest first)
        detected_objects.sort(key=lambda obj: obj.area, reverse=True)
        
        if return_masks:
            return detected_objects, masks
        return detected_objects, None
    
    def draw_detections(
        self,
        image: np.ndarray,
        detected_objects: List[DetectedObject]
    ) -> np.ndarray:
        """
        Draw detection markers on image.
        
        Args:
            image: BGR image to draw on
            detected_objects: List of detected objects
            
        Returns:
            Image with detection overlays
        """
        output = image.copy()
        
        for obj in detected_objects:
            config = self.color_configs.get(obj.color_name)
            if config is None:
                continue
            
            cx, cy = obj.centroid
            display_color = config.display_color
            
            # Draw detection circle
            cv2.circle(output, (cx, cy), 15, display_color, 4)
            cv2.circle(output, (cx, cy), 6, (255, 255, 255), -1)
            
            # Draw label
            text = obj.color_name.upper()
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            
            # Background rectangle
            cv2.rectangle(
                output,
                (cx - text_size[0] // 2, cy - 35),
                (cx + text_size[0] // 2, cy - 10),
                (0, 0, 0),
                -1
            )
            
            # Text
            cv2.putText(
                output,
                text,
                (cx - text_size[0] // 2, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                display_color,
                2
            )
        
        return output


def load_color_configs_from_dict(config_dict: Dict) -> Dict[str, ColorConfig]:
    """
    Load color configurations from a dictionary (e.g., loaded from YAML).
    
    Args:
        config_dict: Dictionary with color configuration data
        
    Returns:
        Dictionary mapping color names to ColorConfig objects
    """
    colors = {}
    
    for color_name, color_data in config_dict.get('colors', {}).items():
        hsv_ranges = []
        for range_data in color_data.get('hsv_ranges', []):
            lower = np.array(range_data['lower'], dtype=np.uint8)
            upper = np.array(range_data['upper'], dtype=np.uint8)
            hsv_ranges.append((lower, upper))
        
        colors[color_name] = ColorConfig(
            name=color_name,
            hsv_ranges=hsv_ranges,
            min_area=color_data.get('min_area', 1000),
            display_color=tuple(color_data.get('display_color', [255, 255, 255])),
            message=color_data.get('message', f'{color_name} detected!')
        )
    
    return colors