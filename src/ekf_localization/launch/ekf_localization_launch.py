from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # GNSS Converter Node (navsat_transform_node)
        Node(
            package='robot_localization',
            executable='navsat_transform_node',
            name='navsat_transform_node',
            output='screen',
            parameters=['share/ekf_localization/config/navsat_transform.yaml/']
        ),
        # Custom EKF Node
        Node(
            package='ekf_localization',
            executable='ekf_node',
            name='ekf_localization_node',
            output='screen',
            parameters=['share/ekf_localization/config/ekf_params.yaml/']
        )
    ])
