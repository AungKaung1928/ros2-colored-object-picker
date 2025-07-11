from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Launch object picker node
        Node(
            package='colored_object_picker',
            executable='object_picker',
            name='object_picker',
            output='screen',
            parameters=[
                {'use_sim_time': False}
            ]
        ),
        
        # Launch RViz (optional - comment out if not needed)
        # Node(
        #     package='rviz2',
        #     executable='rviz2',
        #     name='rviz2',
        #     output='screen'
        # )
    ])