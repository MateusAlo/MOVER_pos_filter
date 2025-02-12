import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    package_dir = get_package_share_directory('ps_localization')

    
    param_file_path = os.path.join(package_dir, 'config', 'ps_node_params.yaml')

    return LaunchDescription([
        Node(
            package='ps_localization',
            executable='ps_node',
            name='particle_filter_localization_node',
            output='screen',
            parameters=[param_file_path]
        )
    ])