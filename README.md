# Colored Object Picker 

A ROS2 package for real-time detection and tracking of colored objects using computer vision. Perfect for robotic pick-and-place applications and color-based object manipulation.

## Features 

-  **6-Color Detection**: Yellow, Pink, Blue, Red, Green, Purple
-  **Real-time Processing**: Live camera feed with OpenCV
-  **Precise Tracking**: Advanced HSV color space filtering with validation
-  **Smart Filtering**: Morphological operations to reduce noise
-  **3D Pose Publishing**: Converts pixel coordinates to world coordinates
-  **Visual Feedback**: Live display with color labels and detection markers

## Package Structure 

```
colored_object_picker/
├── launch/
│   └── pick_and_place.launch.py
├── colored_object_picker/      
│   └── object_picker.py         # Main detection node
│   └── __init__.py            
├── package.xml
├── setup.py
├── setup.cfg
└── README.md
```

## How It Works 

### Color Detection Pipeline
1. **Camera Capture** : Captures frames from USB camera (640x480)
2. **HSV Conversion** : Converts BGR to HSV color space for better color detection
3. **Color Filtering** : Uses predefined HSV ranges for each target color
4. **Noise Reduction** : Applies morphological operations and median blur
5. **Contour Detection** : Finds object boundaries and calculates centroids
6. **Validation** : Multi-level validation (center pixel + region analysis)
7. **Pose Publishing** : Converts pixel coordinates to 3D world coordinates

### Advanced Features
- **Multi-Range Detection**: Handles colors like red that span HSV boundaries
- **Area Filtering**: Minimum area thresholds prevent false positives
- **Region Validation**: Checks surrounding pixels for color consistency
- **Enhanced Precision**: Extra validation for similar colors (pink/purple/blue)

## ROS2 Topics 

### Publishers
- `/camera/image_raw` (sensor_msgs/Image): Raw camera feed
- `/detected_object_pose` (geometry_msgs/Pose): 3D pose of detected objects

### Node Information
- **Node Name**: `precise_color_object_picker`
- **Update Rate**: 10 Hz (100ms timer)

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
pip install opencv-python numpy
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
ros2 launch colored_object_picker pick_and_place.launch.py
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
Edit the `color_ranges` dictionary in `object_picker.py`:
```python
self.color_ranges = {
    'your_color': [(np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))]
}
```

### Modify Detection Parameters
- **Camera Resolution**: Change `CAP_PROP_FRAME_WIDTH` and `CAP_PROP_FRAME_HEIGHT`
- **Minimum Areas**: Adjust `min_areas` dictionary values
- **Update Rate**: Modify timer period in `create_timer()`

## Troubleshooting 

### Common Issues
- **Camera Not Opening**: Check camera permissions and USB connection
- **False Positives**: Adjust minimum area thresholds or HSV ranges
- **Poor Detection**: Ensure good lighting and camera focus
- **No Objects Found**: Verify color ranges match your target objects

### Debug Tools
- Individual color mask windows show detection quality
- Console logs provide detection status and error messages
- Live visualization window shows detection results

## Dependencies 

- **ROS2**: Humble or later
- **OpenCV**: 4.x
- **NumPy**: Latest stable
- **cv_bridge**: ROS2 OpenCV bridge
- **rclpy**: ROS2 Python client library
