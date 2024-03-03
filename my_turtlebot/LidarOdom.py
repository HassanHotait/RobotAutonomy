#!/usr/bin/env python

import cmd
from os import wait
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from scipy.spatial.distance import cdist

class LocalizationNode(Node):
    def __init__(self):
        super().__init__('LidarOdom')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pose_pub = self.create_publisher(Odometry, '/odom', 10)

        transform_scanner = self.wait_for_transform('odom', 'base_scan')


        print(f'Transform Scanner: \n {transform_scanner}')


        self.x0 = transform_scanner.transform.translation.x
        self.y0 = transform_scanner.transform.translation.y
        self.z0 = transform_scanner.transform.translation.z
        self.theta0 = transform_scanner.transform.rotation.z
        self.t0 = self.get_clock().now().nanoseconds / 1e9

        self.scan_msg_prev = None

        self.linear_velocity_mps = 0.0
        self.angular_velocity_radps = 0.0

        self.cmd_vel_msg = Twist()

    def wait_for_transform(self, target_frame, source_frame):
        while True:
            try:
                trans = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
                return trans
            except tf2_ros.LookupException as e:
                self.get_logger().info(f"Transform lookup failed: {str(e)}")
                rclpy.spin_once(self)

    def process_scan(self, scan_msg):
        pc_x = []
        pc_y = []

        for i,range in enumerate(scan_msg.ranges):
            if range < scan_msg.range_min or range > scan_msg.range_max:
                continue
            pc_x.append(range * np.cos(scan_msg.angle_min + (i*scan_msg.angle_increment)))
            pc_y.append(range * np.sin(scan_msg.angle_min + (i*scan_msg.angle_increment)))

        pc_x = np.hstack(pc_x)
        pc_y = np.hstack(pc_y)

        pc = np.vstack((pc_x, pc_y))
        mu = np.mean(pc, axis=1).reshape(-1,1)

        return pc, mu
    
    
    def find_nearest_neighbors(self, pc1, pc2):
        distances = cdist(pc1.T, pc2.T)  # Compute pairwise distances
        nearest_indices = np.argmin(distances, axis=1)  # Find indices of nearest neighbors
        sorted_pc1 = pc1[:, np.argsort(nearest_indices)]  # Sort pc1 based on nearest_indices
        sorted_pc2 = pc2[:, np.argsort(nearest_indices)]  # Sort pc2 based on nearest_indices
        return sorted_pc1, sorted_pc2
    
    def icp(self, pc_2, pc_1, mu_2, mu_1,max_iterations=100, tolerance=1e-5):
        pc_1_norm = pc_1 - mu_1
        pc_2_norm = pc_2 - mu_2

        for i in range(max_iterations):
            # Find the closest points
            if i != 0:
                pc1_crspd, pc2_crspd = self.find_nearest_neighbors(pc1_crspd, pc2_crspd)
            else:
                pc1_crspd, pc2_crspd = self.find_nearest_neighbors(pc_1_norm, pc_2_norm)
            # combined_pc = np.concatenate((pc_1, pc_2), axis=1)

            # Get Transformation
            cov = np.cov(pc1_crspd @ pc2_crspd.T)
            U, S, Vt= np.linalg.svd(cov)
            R = U @ Vt

            t = mu_1 - R @ mu_2
            T = np.vstack((np.hstack((R, t)), np.array([0, 0, 1])))
            # Transform pc_2
            pc1_crspd = T @ np.vstack((pc1_crspd, np.ones((1, pc_2.shape[1]))))

            if pc1_crspd == pc2_crspd:
                print(f'Converged at iteration {i}')
                return T
        
        
        return T

    def cmd_vel_callback(self, cmd_vel_msg):
        # Get Velocity Command
        self.cmd_vel_msg = cmd_vel_msg
        # self.t0 = cmd_vel_msg.header.stamp
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
        delta_t = (self.get_clock().now().nanoseconds / 1e9 - self.t0)
        delta_x = (self.linear_velocity_mps * np.cos(self.theta0) * delta_t)
        delta_y = (self.linear_velocity_mps * np.sin(self.theta0) * delta_t)
        delta_theta = self.angular_velocity_radps * delta_t


        pc_2, mu_2  = self.process_scan(scan_msg)
        # print(f'Point Cloud Mean {pc.shape}: \n {mu}')

        if self.scan_msg_prev is not None:
            pc_1, mu_1 = self.process_scan(self.scan_msg_prev)
            # Perform ICP
            T = self.icp(pc_2, pc_1, mu_2, mu_1)
            print(f'Transformation: \n {T}')


            
            
            
            # Replace this with your ICP algorithm
            # For example, you could use the ICP algorithm from the previous lab
            # or use the ICP algorithm from the PCL library
            # Example code:
            # delta_x, delta_y, delta_theta = self.icp(self.pc_kminus1, pc)

        # x = self.x0 + delta_x
        # y = self.y0 + delta_y
        # theta = self.theta0 + delta_theta

        # # Create PoseStamped message
        # odom = Odometry()

        # odom.header.stamp = self.get_clock().now().to_msg()
        # odom.header.frame_id = 'lidar_odom'

        # odom.child_frame_id = '???'

        # odom.pose.pose.position.x = x
        # odom.pose.pose.position.y = y
        # odom.pose.pose.position.z = self.z0
        # odom.pose.pose.orientation.z = theta

        # odom.twist.twist = self.cmd_vel_msg
        # # Publish the estimated pose
        # self.pose_pub.publish(odom)


        self.t0 = self.t0+ delta_t
        # self.x0 = x
        # self.y0 = y
        # self.theta0 = theta
        self.scan_msg_prev = scan_msg

def main(args=None):
    rclpy.init(args=args)
    localization_node = LocalizationNode()
    rclpy.spin(localization_node)
    localization_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
