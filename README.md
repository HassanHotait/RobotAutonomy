# Getting Startred

## Environment

We are using the ROS2 Humble environment from [docker-ros2-desktop-vnc](https://github.com/Tiryoh/docker-ros2-desktop-vnc)

## Installation
To get started, follow these steps:

1. Pull the docker image:
    ```bash
    docker pull tiryoh/ros2-desktop-vnc:humble
    ```

2. Run the Docker container:
    ```bash
    docker run -p 6080:80 --security-opt seccomp=unconfined --shm-size=512m tiryoh/ros2-desktop-vnc:humble
    ```

## Setup Workspace

1. Clone Workspace:
    ```bash
    git clone https://github.com/HassanHotait/RobotAutonomy
    ```
2. Setup .bashrc file 
    ```bash
    source /opt/ros/humble/setup.bash
    source /home/ubuntu/Desktop/RobotAutonomy/install/setup.bash
    export ROS_DOMAIN_ID=11
    export TURTLEBOT3_MODEL=burger
    export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(ros2 pkg prefix my_turtlebot)/share/my_turtlebot/models
    export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(ros2 pkg prefix turtlebot3_gazebo)/share/turtlebot3_gazebo/models
    ```
2. Build Workspace (Make sure .bashrc is setup):
    ```bash
    cd RobotAutonomy && colcon build --symlink-install
    ```
3. If you face any problems with step 2 install turtlebot3 packages and retry step 2:
    ```bash
    sudo apt install ros-humble-turtlebot3*
    ```
## Launch Simulation
```bash
ros2 launch my_turtlebot turtlebot_simulation.launch
```
## Run AutonomyStack

1. Mapping:
    ```bash
    ros2 run my_turtlebot MapPublisher.py
    ```
    Drive around to discover the map using:
    ```bash
    ros2 run turtlebot3_teleop teleop_keyboard 
    ```
    Visualize Map Discovered Map in RVIZ: Requires the following changes to rviz_launch.rviz

    ```yaml
    Name: Map
      Topic:
        Depth: 1
        Durability Policy: System Default
        Filter size: 10
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /map2
    ```
2. Particle Filter Localization
    ```bash
    ros2 run my_turtlebot ParticleFilter.py
    ```









