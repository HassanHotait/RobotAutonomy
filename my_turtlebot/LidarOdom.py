#!/usr/bin/env python

import cmd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
import tf2_ros
import tf2_geometry_msgs
import numpy as np

class LocalizationNode(Node):
    def __init__(self):
        super().__init__('LidarOdom')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.scan_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pose_pub = self.create_publisher(Odometry, '/odom', 10)

        self.x0 = 0.0
        self.y0 = 0.0
        self.theta0 = 0.0
        self.t0 = None

        self.linear_velocity_mps = 0.0
        self.angular_velocity_radps = 0.0

    def cmd_vel_callback(self, cmd_vel_msg):
        # Get Velocity Command
        self.t0 = cmd_vel_msg.header.stamp
        self.linear_velocity_mps = cmd_vel_msg.linear.x
        self.angular_velocity_radps = cmd_vel_msg.angular.z

    def scan_callback(self, scan_msg):
        # Perform localization using constant velocity model
        # Replace this with your localization algorithm
        # For example, you could use particle filter localization or Extended Kalman Filter

        # For simplicity, let's assume we just use the robot's linear velocity and angular velocity
        # to update its position and orientation

        # Estimate new pose based on constant velocity model
        # Example code:
        delta_t = (self.get_clock().now() - self.t0).nanoseconds / 1e9
        x = self.x0 + (self.linear_velocity_mps * np.cos(self.theta0) * delta_t)
        y = self.y0 + (self.linear_velocity_mps * np.sin(self.theta0) * delta_t)
        theta = self.theta0 + self.angular_velocity_radps * delta_t

        # Create PoseStamped message
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'lidar_odom'
        odom.child_frame_id = '???'
        odom.pose.pose.position.x = ...
        odom.pose.pose.position.y = ...
        odom.pose.pose.position.z = ...
        odom.pose.pose.orientation = ...
        odom.twist.twist.linear.x = ...
        odom.twist.twist.angular.z = ...
        odom.pose.covariance = [0.1, 0, 0, 0, 0, 0,
                                0, 0.1, 0, 0, 0, 0,
                                0, 0, 0.1, 0, 0, 0,
                                0, 0, 0, 0.1, 0, 0,
                                0, 0, 0, 0, 0.1, 0,
                                0, 0, 0, 0, 0, 0.1]
        odom.twist.covariance = [0.1, 0, 0, 0, 0, 0,
                                 0, 0.1, 0, 0, 0, 0,
                                 0, 0, 0.1, 0, 0, 0,
                                 0, 0, 0, 0.1, 0, 0,
                                 0, 0, 0, 0, 0.1, 0,
                                 0, 0, 0, 0, 0, 0.1]
        
        # Publish the estimated pose
        # Set estimated position and orientation
        # estimated_pose.pose.position.x = ...
        # estimated_pose.pose.position.y = ...
        # estimated_pose.pose.position.z = ...
        # estimated_pose.pose.orientation = ...

        # Publish the estimated pose
        self.pose_pub.publish(estimated_pose)

def main(args=None):
    rclpy.init(args=args)
    localization_node = LocalizationNode()
    rclpy.spin(localization_node)
    localization_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
