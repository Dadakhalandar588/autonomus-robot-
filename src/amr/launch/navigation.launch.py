import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import TimerAction


def generate_launch_description():

    package_path = get_package_share_directory('amr')

    ekf_config_path = os.path.join(package_path, "config", "ekf.yaml")
    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[ekf_config_path, {"use_sim_time": True}],
    )

    amcl_config_path = os.path.join(package_path, "config", "amcl.yaml")
    amcl_node = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[amcl_config_path, {"use_sim_time": True}],
    )

    map_config = os.path.join(package_path, "map", "map_1788686899.yaml")
    map_node = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[{"use_sim_time": True, "yaml_filename": map_config}]
    )

    lifecycle_manager_localization = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "autostart": True,
            "node_names": ["map_server", "amcl"]
        }]
    )

    global_costmap_config = os.path.join(package_path, "config", "global_costmap.yaml")
    global_costmap_node = Node(
        package="nav2_costmap_2d",
        executable="nav2_costmap_2d",
        name="global_costmap",
        namespace="global_costmap",
        output="screen",
        parameters=[global_costmap_config, {"use_sim_time": True}],
    )

    local_costmap_config = os.path.join(package_path, "config", "local_costmap.yaml")
    local_costmap_node = Node(
        package="nav2_costmap_2d",
        executable="nav2_costmap_2d",
        name="local_costmap",
        namespace="local_costmap",
        output="screen",
        parameters=[local_costmap_config, {"use_sim_time": True}],
    )

    lifecycle_manager_costmaps = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_costmaps",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "autostart": True,
            "bond_timeout": 0.0,
            "node_names": ["global_costmap/global_costmap", "local_costmap/local_costmap"]
        }]
    )

    delayed_costmap_bringup = TimerAction(
        period=3.0,
        actions=[
            global_costmap_node,
            local_costmap_node,
            lifecycle_manager_costmaps,
        ]
    )

    return LaunchDescription([
        ekf_node,
        map_node,
        amcl_node,
        lifecycle_manager_localization,
        delayed_costmap_bringup,
    ])