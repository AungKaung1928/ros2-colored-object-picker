#!/usr/bin/env python3
"""
Launch file for the color detector node.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for color detector."""
    
    # Get package share directory
    pkg_share = get_package_share_directory('colored_object_picker')
    default_config = os.path.join(pkg_share, 'config', 'colors.yaml')
    
    # Declare launch arguments
    camera_id_arg = DeclareLaunchArgument(
        'camera_id',
        default_value='0',
        description='Camera device ID'
    )
    
    frame_width_arg = DeclareLaunchArgument(
        'frame_width',
        default_value='640',
        description='Camera frame width'
    )
    
    frame_height_arg = DeclareLaunchArgument(
        'frame_height',
        default_value='480',
        description='Camera frame height'
    )
    
    update_rate_arg = DeclareLaunchArgument(
        'update_rate',
        default_value='10.0',
        description='Detection update rate in Hz'
    )
    
    show_visualization_arg = DeclareLaunchArgument(
        'show_visualization',
        default_value='true',
        description='Show OpenCV visualization window'
    )
    
    show_masks_arg = DeclareLaunchArgument(
        'show_masks',
        default_value='false',
        description='Show individual color mask windows'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Path to color configuration YAML file'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    # Color detector node
    detector_node = Node(
        package='colored_object_picker',
        executable='detector_node',
        name='color_detector',
        output='screen',
        parameters=[{
            'camera_id': LaunchConfiguration('camera_id'),
            'frame_width': LaunchConfiguration('frame_width'),
            'frame_height': LaunchConfiguration('frame_height'),
            'update_rate': LaunchConfiguration('update_rate'),
            'show_visualization': LaunchConfiguration('show_visualization'),
            'show_masks': LaunchConfiguration('show_masks'),
            'config_file': LaunchConfiguration('config_file'),
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )
    
    return LaunchDescription([
        camera_id_arg,
        frame_width_arg,
        frame_height_arg,
        update_rate_arg,
        show_visualization_arg,
        show_masks_arg,
        config_file_arg,
        use_sim_time_arg,
        detector_node
    ])