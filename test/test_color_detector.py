#!/usr/bin/env python3
"""
Unit tests for ColorDetector class.
Tests centroid calculation, pixel-to-world conversion, and mask creation.
"""

import numpy as np
import pytest
import cv2

from colored_object_picker.color_detector import (
    ColorDetector,
    ColorConfig,
    DetectedObject,
    WorldPose,
    load_color_configs_from_dict
)


@pytest.fixture
def sample_color_configs():
    """Fixture providing sample color configurations."""
    return {
        'red': ColorConfig(
            name='red',
            hsv_ranges=[
                (np.array([0, 150, 150]), np.array([10, 255, 255])),
                (np.array([170, 150, 150]), np.array([180, 255, 255]))
            ],
            min_area=1000,
            display_color=(0, 0, 255),
            message='Red detected!'
        ),
        'blue': ColorConfig(
            name='blue',
            hsv_ranges=[
                (np.array([100, 150, 150]), np.array([120, 255, 255]))
            ],
            min_area=1000,
            display_color=(255, 0, 0),
            message='Blue detected!'
        )
    }


@pytest.fixture
def detector(sample_color_configs):
    """Fixture providing a ColorDetector instance."""
    return ColorDetector(sample_color_configs)


class TestCentroidCalculation:
    """Tests for centroid calculation functionality."""
    
    def test_centroid_square_contour(self):
        """Test centroid calculation for a square contour."""
        # Create a square contour centered at (50, 50)
        contour = np.array([
            [[25, 25]],
            [[75, 25]],
            [[75, 75]],
            [[25, 75]]
        ], dtype=np.int32)
        
        centroid = ColorDetector.calculate_centroid(contour)
        
        assert centroid is not None
        cx, cy = centroid
        assert cx == 50
        assert cy == 50
    
    def test_centroid_triangle_contour(self):
        """Test centroid calculation for a triangular contour."""
        # Triangle with vertices at (0, 0), (100, 0), (50, 100)
        contour = np.array([
            [[0, 0]],
            [[100, 0]],
            [[50, 100]]
        ], dtype=np.int32)
        
        centroid = ColorDetector.calculate_centroid(contour)
        
        assert centroid is not None
        cx, cy = centroid
        # Centroid of triangle is at (sum_x/3, sum_y/3)
        assert abs(cx - 50) <= 1  # Allow small rounding error
        assert abs(cy - 33) <= 1
    
    def test_centroid_rectangle_contour(self):
        """Test centroid calculation for a rectangle."""
        # Rectangle from (10, 20) to (110, 80)
        contour = np.array([
            [[10, 20]],
            [[110, 20]],
            [[110, 80]],
            [[10, 80]]
        ], dtype=np.int32)
        
        centroid = ColorDetector.calculate_centroid(contour)
        
        assert centroid is not None
        cx, cy = centroid
        assert cx == 60  # (10 + 110) / 2
        assert cy == 50  # (20 + 80) / 2
    
    def test_centroid_invalid_contour(self):
        """Test centroid calculation with zero-area contour returns None."""
        # Line (zero area)
        contour = np.array([
            [[0, 0]],
            [[100, 0]]
        ], dtype=np.int32)
        
        centroid = ColorDetector.calculate_centroid(contour)
        
        assert centroid is None


class TestPixelToWorldConversion:
    """Tests for pixel to world coordinate conversion."""
    
    def test_center_pixel(self):
        """Test that center pixel maps to world origin."""
        pixel = (320, 240)  # Center of 640x480 image
        
        pose = ColorDetector.pixel_to_world(pixel)
        
        assert isinstance(pose, WorldPose)
        assert pose.x == pytest.approx(0.0)
        assert pose.y == pytest.approx(0.0)
        assert pose.z == pytest.approx(0.1)
    
    def test_top_left_pixel(self):
        """Test top-left corner conversion."""
        pixel = (0, 0)
        
        pose = ColorDetector.pixel_to_world(pixel, image_width=640, image_height=480)
        
        assert pose.x == pytest.approx(-0.32)  # -320 * 0.001
        assert pose.y == pytest.approx(-0.24)  # -240 * 0.001
    
    def test_bottom_right_pixel(self):
        """Test bottom-right corner conversion."""
        pixel = (640, 480)
        
        pose = ColorDetector.pixel_to_world(pixel, image_width=640, image_height=480)
        
        assert pose.x == pytest.approx(0.32)  # 320 * 0.001
        assert pose.y == pytest.approx(0.24)  # 240 * 0.001
    
    def test_custom_scale(self):
        """Test with custom scale factor."""
        pixel = (320, 240)  # Center
        
        # With center pixel, result should still be 0 regardless of scale
        pose = ColorDetector.pixel_to_world(pixel, scale=0.01)
        
        assert pose.x == pytest.approx(0.0)
        assert pose.y == pytest.approx(0.0)
        
        # Test non-center pixel with custom scale
        pixel = (420, 340)  # 100 pixels right and 100 down from center
        pose = ColorDetector.pixel_to_world(pixel, scale=0.01)
        
        assert pose.x == pytest.approx(1.0)  # 100 * 0.01
        assert pose.y == pytest.approx(1.0)
    
    def test_custom_z_offset(self):
        """Test with custom z offset."""
        pixel = (320, 240)
        
        pose = ColorDetector.pixel_to_world(pixel, z_offset=0.5)
        
        assert pose.z == pytest.approx(0.5)
    
    def test_quaternion_defaults(self):
        """Test that quaternion defaults to identity."""
        pixel = (320, 240)
        
        pose = ColorDetector.pixel_to_world(pixel)
        
        assert pose.qw == pytest.approx(1.0)
        assert pose.qx == pytest.approx(0.0)
        assert pose.qy == pytest.approx(0.0)
        assert pose.qz == pytest.approx(0.0)


class TestMaskCreation:
    """Tests for color mask creation."""
    
    def test_mask_shape(self, detector):
        """Test that created mask has correct shape."""
        # Create a test HSV image
        hsv_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        mask = detector.create_color_mask(hsv_image, 'red')
        
        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8
    
    def test_mask_unknown_color(self, detector):
        """Test mask creation for unknown color returns empty mask."""
        hsv_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        mask = detector.create_color_mask(hsv_image, 'unknown_color')
        
        assert mask.shape == (480, 640)
        assert np.all(mask == 0)
    
    def test_mask_detects_red(self, detector):
        """Test that red regions are detected in mask."""
        # Create HSV image with a red square
        hsv_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Red in HSV: H=0-10 or 170-180, S=150+, V=150+
        hsv_image[200:280, 280:360] = [5, 200, 200]  # Red square
        
        mask = detector.create_color_mask(hsv_image, 'red')
        
        # Check that some pixels are detected (after morphological ops)
        assert np.sum(mask) > 0


class TestColorConfigLoading:
    """Tests for loading color configurations from dict."""
    
    def test_load_valid_config(self):
        """Test loading a valid configuration dictionary."""
        config_dict = {
            'colors': {
                'test_color': {
                    'hsv_ranges': [
                        {'lower': [0, 100, 100], 'upper': [10, 255, 255]}
                    ],
                    'min_area': 500,
                    'display_color': [255, 0, 0],
                    'message': 'Test color detected!'
                }
            }
        }
        
        configs = load_color_configs_from_dict(config_dict)
        
        assert 'test_color' in configs
        assert configs['test_color'].min_area == 500
        assert configs['test_color'].message == 'Test color detected!'
        assert len(configs['test_color'].hsv_ranges) == 1
    
    def test_load_empty_config(self):
        """Test loading empty configuration."""
        config_dict = {}
        
        configs = load_color_configs_from_dict(config_dict)
        
        assert configs == {}
    
    def test_load_multiple_ranges(self):
        """Test loading color with multiple HSV ranges."""
        config_dict = {
            'colors': {
                'red': {
                    'hsv_ranges': [
                        {'lower': [0, 150, 150], 'upper': [10, 255, 255]},
                        {'lower': [170, 150, 150], 'upper': [180, 255, 255]}
                    ],
                    'min_area': 1000,
                    'display_color': [0, 0, 255],
                    'message': 'Red!'
                }
            }
        }
        
        configs = load_color_configs_from_dict(config_dict)
        
        assert len(configs['red'].hsv_ranges) == 2


class TestDetectedObjectDataclass:
    """Tests for DetectedObject dataclass."""
    
    def test_create_detected_object(self):
        """Test creating a DetectedObject instance."""
        contour = np.array([[[0, 0]], [[100, 0]], [[100, 100]], [[0, 100]]])
        
        obj = DetectedObject(
            color_name='red',
            centroid=(50, 50),
            area=10000.0,
            contour=contour
        )
        
        assert obj.color_name == 'red'
        assert obj.centroid == (50, 50)
        assert obj.area == 10000.0
        assert np.array_equal(obj.contour, contour)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])