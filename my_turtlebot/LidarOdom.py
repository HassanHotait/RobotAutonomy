

import cmd
from os import wait

from yaml import scan
from osrf_pycommon.terminal_color.windows import _print_ansi_color_win32
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from scipy.spatial.distance import cdist
import math
from simpleicp import PointCloud, SimpleICP

class LocalizationNode(Node):
    def __init__(self):
        super().__init__('LidarOdom')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pose_pub = self.create_publisher(Odometry, '/odom', 1)

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
            if range > scan_msg.range_max:
                pc_x.append(scan_msg.range_max * np.cos(scan_msg.angle_min + (i*scan_msg.angle_increment)))
                pc_y.append(scan_msg.range_max * np.sin(scan_msg.angle_min + (i*scan_msg.angle_increment)))
            else:
                pc_x.append(range * np.cos(scan_msg.angle_min + (i*scan_msg.angle_increment)))
                pc_y.append(range * np.sin(scan_msg.angle_min + (i*scan_msg.angle_increment)))

        pc_x = np.hstack(pc_x)
        pc_y = np.hstack(pc_y)

        pc = np.vstack((pc_x, pc_y))

        return pc
    
    
    def find_nearest_neighbors(self, pc1, pc2):
        distances = cdist(pc1.T, pc2.T)  # Compute pairwise distances
        nearest_indices = np.argmin(distances, axis=1)  # Find indices of nearest neighbors
        sorted_pc1 = pc1[:, np.argsort(nearest_indices)]  # Sort pc1 based on nearest_indices
        sorted_pc2 = pc2[:, np.argsort(nearest_indices)]  # Sort pc2 based on nearest_indices
        return sorted_pc1, sorted_pc2
    
    def RMSE(self, pc1, pc2):

        e = np.mean(np.sqrt( (pc1[0] - pc2[0])** 2 + (pc1[1] - pc2[1])** 2))

        return e
    
    def icp(self, pc1,pc2,T,max_iterations=1000, tolerance=0.05,epsilon=1e-8):

        def Pi(pts_inhomogenous):
            """
            Converts homogeneous points to inhomogeneous coordinates.
            
            Args:
                pts_inhomogenous (numpy.ndarray): Homogeneous points represented as 3xN or 4xN.
                
            Returns:
                numpy.ndarray: The inhomogeneous representation of the input point.
            """

            assert pts_inhomogenous.shape[0] == 3 or pts_inhomogenous.shape[0] == 4, "Input points must be 3xN or 4xN"
            tmp = pts_inhomogenous[:-1] / pts_inhomogenous[-1]
            return tmp
        
        def PiInv(pts_homogenous):
            """
            Converts inhomogeneous points to homogeneous coordinates.

            Parameters:
            p (numpy.ndarray): Inhomogeneous point represents as 2xN or 3xN.

            Returns:
            numpy.ndarray: The homogeneous representation of the input point.
            """

            assert pts_homogenous.shape[0] == 2 or pts_homogenous.shape[0] == 3, "Input points must be 2xN or 3xN"
            tmp = np.vstack((pts_homogenous,np.ones(pts_homogenous.shape)[0]))

            return tmp
        
        pc1 = Pi(T@PiInv(pc1))
        for i in range(max_iterations):
            pc1 = (pc1- np.mean(pc1, axis=1, keepdims=True)) / np.std(pc1, axis=1, keepdims=True)
            pc2 = (pc2 - np.mean(pc2, axis=1, keepdims=True)) / np.std(pc2, axis=1, keepdims=True)
            # # Get Transformation
            assert pc1.shape == pc2.shape, 'Point clouds must have the same shape'
            # Get Covariance Matrix of Point Clouds 1 and 2
            cov = np.cov(pc1 @ pc2.T)
            # Add a small positive constant to the diagonal for regularization
            cov += np.eye(cov.shape[0]) * epsilon
            # Get SVD of Covariance Matrix
            U, S, Vt= np.linalg.svd(cov)
            # Get Rotation Matrix
            R = (U @ Vt).reshape(2,2) 
            # Get Translation
            t = np.mean(pc1, axis=1, keepdims=True) - (R @ np.mean(pc2, axis=1, keepdims=True))
            # Transform pc_1
            T = np.vstack((np.hstack((R, t)), np.array([[0, 0, 1]])))
            pc1 = Pi(T @ np.vstack((pc1, np.ones((1, pc1.shape[1])))))


        if self.RMSE(pc1, pc2) < tolerance:
                return R, t
        else:
            print(f'RMSE: {self.RMSE(pc1, pc2)}')
            raise ValueError('ICP did not converge')


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
        T = np.array([
            [np.cos(delta_theta), -np.sin(delta_theta), delta_x],
            [np.sin(delta_theta), np.cos(delta_theta), delta_y],
            [0, 0, 1]
        ])

        if self.scan_msg_prev is not None:
            pc_1= self.process_scan(self.scan_msg_prev)
            pc_2= self.process_scan(scan_msg)
            # Perform ICP
            R, t = self.icp(pc_1,pc_2,T)
            x = self.x0 + t[0]
            y = self.y0 + t[1]
            z = self.z0 
            theta = self.theta0 + R[1,1]
            print(f'Rotation Matrix: \n {R}')
            print(f'Translation: \n {t}')

        else:
            x = self.x0 
            y = self.y0 
            z = self.z0
            theta = self.theta0 

        odom = Odometry()

        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'lidar_odom'

        odom.child_frame_id = '???'

        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.position.z = float(z)
        odom.pose.pose.orientation.z = float(theta)
        odom.twist.twist = self.cmd_vel_msg
        # Publish the estimated pose
        self.pose_pub.publish(odom)

        self.t0 = self.t0+ delta_t
        self.x0 = x
        self.y0 = y
        self.z0 = z
        self.theta0 = theta
        self.scan_msg_prev = scan_msg

def main(args=None):
    rclpy.init(args=args)
    localization_node = LocalizationNode()
    rclpy.spin(localization_node)
    localization_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
