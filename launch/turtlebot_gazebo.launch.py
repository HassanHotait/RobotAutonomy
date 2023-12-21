"""
Author: David Valencia
Date: 25 / 08 /2021

Describer:  Simple launch to SIMULATE the doosan robot in GAZEBO in my own package
            Based on the original git package from doosan-robot2
            This scripts just spawns the robot arm in GAZEBO
            the robot description (urdf and xacro) are in: src/my_doosan_pkg/description/xacro

            Robot model m1013 color white.
            Robot model a0912 color blue.
"""

import os
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import Command
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    #xacro_file = get_package_share_directory('turtlebot') + '/urdf/turtlebot3_$(arg model).urdf.xacro'

    # Start Gazebo with my empty world
    world = get_package_share_directory('my_turtlebot') + '/worlds/lab_world.world'
    #world = get_package_share_directory('my_turtlebot') + '/worlds/test.world'
    #gazebo_node = ExecuteProcess(cmd=['gazebo', '--verbose', world, '-s', 'libgazebo_ros_factory.so'], output='screen')

    #return LaunchDescription([gazebo_node])
    
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')

    #world = os.path.join(
    #    get_package_share_directory('turtlebot3_gazebo'),
    #    'worlds',
    #    'empty_world.world'
    #)

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose
        }.items()
    )

    ld = LaunchDescription()

    # Add the commands to the launch description
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)

    return ld
