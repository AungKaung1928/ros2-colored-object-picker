# Colored Object Picker

A ROS2 package for real-time detection and tracking of colored objects using computer vision. Perfect for robotic pick-and-place applications and color-based object manipulation.

## Features

- **6-Color Detection**: Yellow, Pink, Blue, Red, Green, Purple
- **Real-time Processing**: Live camera feed with OpenCV
- **Precise Tracking**: Advanced HSV color space filtering with validation
- **Smart Filtering**: Morphological operations to reduce noise
- **Pose Publishing**: `PoseStamped` in the camera frame; image-plane offset at an assumed scale (see note below)
- **Visual Feedback**: Live display with color labels and detection markers
- **Configurable**: YAML-based color configuration
- **Testable**: Unit tests for core detection logic

## Package Structure

```
colored_object_picker/
├── colored_object_picker/
│   ├── __init__.py
│   ├── color_detector.py      # Detection logic (pure Python)
│   └── detector_node.py       # ROS2 node wrapper
├── config/
│   └── colors.yaml            # Color configuration
├── test/
│   └── test_color_detector.py # Unit tests
├── launch/
│   └── detector.launch.py
├── package.xml
├── setup.py
├── setup.cfg
└── README.md
```

## How It Works

### Color Detection Pipeline
1. **Camera Capture**: Captures frames from USB camera (640x480)
2. **HSV Conversion**: Converts BGR to HSV color space for better color detection
3. **Color Filtering**: Uses configurable HSV ranges for each target color
4. **Noise Reduction**: Applies morphological operations and median blur
5. **Contour Detection**: Finds object boundaries and calculates centroids
6. **Validation**: Multi-level validation (center pixel + region analysis)
7. **Pose Publishing**: Converts pixel coordinates to 3D world coordinates

### Advanced Features
- **Multi-Range Detection**: Handles colors like red that span HSV boundaries
- **Area Filtering**: Minimum area thresholds prevent false positives
- **Region Validation**: Checks surrounding pixels for color consistency
- **Enhanced Precision**: Extra validation for similar colors (pink/purple/blue)

## ROS2 Topics

### Publishers
- `/camera/image_raw` (sensor_msgs/Image): raw camera feed (camera-device source only)
- `/detected_object_pose` (geometry_msgs/PoseStamped): one message per detection, `frame_id` = `camera_frame`

### Subscribers
- `<image_topic>` (sensor_msgs/Image, best-effort): used instead of the camera device when the
  `image_topic` parameter is non-empty. Lets the node run on a Gazebo camera, a bag, or a test
  publisher.

### About the pose
`pixel_to_world()` returns the pixel offset from the image centre scaled by 1 mm/px with a fixed
z of 0.1 m. That is a planar target in the camera frame, **not** a metric 3D position: a metric
pose needs the camera intrinsics plus a depth source (RGB-D, stereo, or a known object size).

### Parameters
| Parameter | Type | Default | Description |
| `image_topic` | string | `''` | Non-empty: subscribe to this topic instead of opening `camera_id` |
| `camera_frame` | string | `camera_link` | `frame_id` stamped on images and poses |
|-----------|------|---------|-------------|
| `camera_id` | int | 0 | Camera device ID |
| `frame_width` | int | 640 | Camera frame width |
| `frame_height` | int | 480 | Camera frame height |
| `update_rate` | float | 10.0 | Detection rate (Hz) |
| `show_visualization` | bool | true | Show OpenCV window |
| `show_masks` | bool | false | Show color mask windows |
| `config_file` | string | colors.yaml | Path to config file |

## Color Ranges

| Color | HSV Range | Min Area | Display Color |
|-------|-----------|----------|---------------|
| 🟡 Yellow | [18,120,180] - [32,255,255] | 800px | Yellow |
| 🩷 Pink | [140,50,180] - [170,150,255] | 800px | Pink |
| 🔵 Blue | [100,150,150] - [120,255,255] | 1200px | Blue |
| 🔴 Red | [0,150,150] - [10,255,255] + [170,150,150] - [180,255,255] | 1200px | Red |
| 🟢 Green | [50,150,100] - [70,255,255] | 1200px | Green |
| 🟣 Purple | [125,120,120] - [140,255,255] | 1000px | Purple |

## Installation

### Prerequisites
```bash
# ROS2 dependencies
sudo apt install ros-humble-cv-bridge
sudo apt install ros-humble-sensor-msgs
sudo apt install ros-humble-geometry-msgs

# Python dependencies
pip install opencv-python numpy pyyaml pytest
```

### Build Package
```bash
cd ~/colored_object_ws/src
git clone <your-repo-url>
cd ~/colored_object_ws
colcon build --packages-select colored_object_picker
source install/setup.bash
```

## Usage

### Launch with Launch File
```bash
ros2 launch colored_object_picker detector.launch.py
```

### Launch with Custom Parameters
```bash
ros2 launch colored_object_picker detector.launch.py camera_id:=1 update_rate:=30.0 show_masks:=true
```

### Monitor Detection
```bash
# View detected poses
ros2 topic echo /detected_object_pose

# View camera feed
ros2 topic echo /camera/image_raw
```

## Configuration

### Adjust Color Ranges
Edit `config/colors.yaml`:
```yaml
colors:
  your_color:
    hsv_ranges:
      - lower: [h_min, s_min, v_min]
        upper: [h_max, s_max, v_max]
    min_area: 1000
    display_color: [b, g, r]
    message: "Color detected!"
```

### Modify Detection Parameters
- **Camera Resolution**: Set via launch parameters or `colors.yaml`
- **Minimum Areas**: Adjust in `colors.yaml`
- **Update Rate**: Set via launch parameter `update_rate`

## Testing

### Unit tests (no camera)
```bash
python3 -m pytest test/   # 17 tests on masks, centroids, validation, config loading
```

### End-to-end without hardware
Publish a synthetic image with a yellow and a blue square, run the detector on that topic, and
watch the poses. Verified 2026-09-05 on Humble: both colours detected every frame, poses at the
input rate.
```bash
# terminal 1: synthetic camera (see scripts in the README history or write a 15-line rclpy
# publisher that draws two cv2.rectangle() blocks and publishes bgr8 on /synthetic/image_raw)
# terminal 2
ros2 run colored_object_picker detector_node --ros-args -p image_topic:=/synthetic/image_raw -p show_visualization:=false
# terminal 3
ros2 topic echo /detected_object_pose
```
Expected for a blue square centred at pixel (490, 360) in a 640x480 image: `position: {x: 0.17, y: 0.12, z: 0.1}`.

### With a webcam
```bash
ros2 launch colored_object_picker detector.launch.py
```
Hold a coloured object in front of the camera; the window labels it and `/detected_object_pose`
publishes. HSV ranges are in `config/colors.yaml`; tune them under your lighting.

## Troubleshooting

### Common Issues
- **Camera Not Opening**: Check camera permissions and USB connection
- **False Positives**: Adjust minimum area thresholds or HSV ranges in `colors.yaml`
- **Poor Detection**: Ensure good lighting and camera focus
- **No Objects Found**: Verify color ranges match your target objects

### Debug Tools
- Individual color mask windows (`show_masks:=true`)
- Console logs provide detection status and error messages
- Live visualization window shows detection results

## Dependencies

- **ROS2**: Humble or later
- **OpenCV**: 4.x
- **NumPy**: Latest stable
- **PyYAML**: Configuration loading
- **cv_bridge**: ROS2 OpenCV bridge
- **rclpy**: ROS2 Python client library
- **pytest**: Unit testing
