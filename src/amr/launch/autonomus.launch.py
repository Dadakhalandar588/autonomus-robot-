import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    package_path = get_package_share_directory("amr")

    # Planner configuration
    planner_config = os.path.join(
        package_path,
        "config",
        "planner.yaml"
    )

    global_costmap_config = os.path.join(
        package_path,
        "config",
        "global_costmap.yaml"
    )

    planner = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=[
            planner_config,
            global_costmap_config
        ]
    )

    # Controller configuration
    controller_config = os.path.join(
        package_path,
        "controller",
        "regulated_pure_puseit_.yaml"
    )

    local_costmap_config = os.path.join(
        package_path,
        "config",
        "local_costmap.yaml"
    )

    controller = Node(
        package="nav2_controller",
        executable="controller_server",
        name="controller_server",
        output="screen",
        parameters=[
            controller_config,
            local_costmap_config
        ]
    )

    # BT Navigator configuration
    bt_xml = os.path.join(
        package_path,
        "bt",
        "bt.xml"
    )

    bt_config = os.path.join(
        package_path,
        "config",
        "bt.yaml"
    )

    bt_navigator = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        name="bt_navigator",
        output="screen",
        parameters=[
            bt_config,
            {
                "default_nav_to_pose_bt_xml": bt_xml
            }
        ]
    )

    # Behavior Server
    behavior_config = os.path.join(
        package_path,
        "config",
        "behavior.yaml"
    )

    behavior_server = Node(
        package="nav2_behaviors",
        executable="behavior_server",
        name="behavior_server",
        output="screen",
        parameters=[
            behavior_config
        ]
    )

    # Lifecycle Manager
    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_navigation",
        output="screen",
        parameters=[
            {
                "autostart": True,
                "node_names": [
                    "planner_server",
                    "controller_server",
                    "behavior_server",
                    "bt_navigator"
                ]
            }
        ]
    )

    return LaunchDescription([
        planner,
        controller,
        behavior_server,
        bt_navigator,
        lifecycle_manager
    ])