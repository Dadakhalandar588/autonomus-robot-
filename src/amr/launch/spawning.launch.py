import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    package_path = get_package_share_directory('amr')

    urdf_file = os.path.join(package_path, 'urdf', 'vaccum_robot.urdf')
    world_file = os.path.join(package_path, 'sdf', 'room.sdf')

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # Gazebo
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', world_file, '-r'],
        output='screen',
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
    )

    # Spawn robot (2 s after Gazebo starts)
    spawn_robot_node = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-name', 'vaccum_robot',
            '-topic', '/robot_description',
            '-x', '-2.0',
            '-y', '-1',
            '-z', '0.11',
        ],
    )

    spawn_after_gazebo_starts = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=gazebo,
            on_start=[
                TimerAction(period=2.0, actions=[spawn_robot_node]),
            ],
        )
    )

    # ROS <-> Gazebo bridge
    # rgbd_camera with <topic>camera</topic> publishes /camera/image (color)
    # and /camera/depth_image, so bridge the real name and remap to image_raw.
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
        ],
        remappings=[
            ('/camera/image', '/camera/image_raw'),
        ],
        output='screen',
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_after_gazebo_starts,
        bridge,
    ])