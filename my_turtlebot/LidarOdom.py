
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import tf2_ros
import numpy as np
from scipy.spatial.distance import cdist

class LocalizationNode(Node):
    """
    A ROS2 node for localization using lidar odometry.

    This node subscribes to the `/cmd_vel` topic for velocity commands,
    the `/scan` topic for lidar scan data, and publishes the estimated
    odometry on the `/odom` topic.

    Attributes:
        tf_buffer (tf2_ros.Buffer): Buffer for storing transforms.
        tf_listener (tf2_ros.TransformListener): Listener for transforms.
        cmd_sub (rclpy.subscription.Subscription): Subscription to the `/cmd_vel` topic.
        scan_sub (rclpy.subscription.Subscription): Subscription to the `/scan` topic.
        pose_pub (rclpy.publisher.Publisher): Publisher for the estimated odometry.
        x0 (float): Initial x position.
        y0 (float): Initial y position.
        z0 (float): Initial z position.
        theta0 (float): Initial orientation.
        t0 (float): Initial time.
        scan_msg_prev (sensor_msgs.msg.LaserScan): Previous lidar scan message.
        lidar_odom (??): Placeholder for lidar odometry.
        linear_velocity_mps (float): Linear velocity in meters per second.
        angular_velocity_radps (float): Angular velocity in radians per second.
        cmd_vel_msg (geometry_msgs.msg.Twist): Velocity command message.
    """

    def __init__(self):
        super().__init__('LidarOdom')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pose_pub = self.create_publisher(Odometry, '/odom', 1)

        transform_scanner = self.wait_for_transform('odom', 'base_scan')

        # Initialize variables
        self.x0 = transform_scanner.transform.translation.x
        self.y0 = transform_scanner.transform.translation.y
        self.z0 = transform_scanner.transform.translation.z
        self.theta0 = transform_scanner.transform.rotation.z
        self.t0 = self.get_clock().now().nanoseconds / 1e9
        self.scan_msg_prev = None
        self.lidar_odom = None
        self.linear_velocity_mps = 0.0
        self.angular_velocity_radps = 0.0
        self.cmd_vel_msg = Twist()

    def wait_for_transform(self, target_frame, source_frame):
        """
        Wait for a transform between two frames.

        Args:
            target_frame (str): The target frame.
            source_frame (str): The source frame.

        Returns:
            geometry_msgs.msg.TransformStamped: The transform between the frames.
        """
        while True:
            try:
                trans = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
                return trans
            except tf2_ros.LookupException as e:
                self.get_logger().info(f"Transform lookup failed: {str(e)}")
                rclpy.spin_once(self)

    def process_scan(self, scan_msg):
        """
        Process a lidar scan message and convert it to a point cloud.

        Args:
            scan_msg (sensor_msgs.msg.LaserScan): The lidar scan message.

        Returns:
            numpy.ndarray: The point cloud as a 2D array of shape (2, N).
        """
        pc_x = []
        pc_y = []

        # Range to Point Cloud
        for i, range in enumerate(scan_msg.ranges):
            if range > scan_msg.range_max:
                pc_x.append(scan_msg.range_max * np.cos(scan_msg.angle_min + (i * scan_msg.angle_increment)))
                pc_y.append(scan_msg.range_max * np.sin(scan_msg.angle_min + (i * scan_msg.angle_increment)))
            else:
                pc_x.append(range * np.cos(scan_msg.angle_min + (i * scan_msg.angle_increment)))
                pc_y.append(range * np.sin(scan_msg.angle_min + (i * scan_msg.angle_increment)))

        # Return Point Cloud (2, N)
        pc_x = np.hstack(pc_x)
        pc_y = np.hstack(pc_y)
        pc = np.vstack((pc_x, pc_y))

        return pc

    def RMSE(self, pc1, pc2):
        """
        Calculate the root mean square error (RMSE) between two point clouds.

        Args:
            pc1 (numpy.ndarray): The first point cloud as a 2D array of shape (2, N).
            pc2 (numpy.ndarray): The second point cloud as a 2D array of shape (2, N).

        Returns:
            float: The RMSE between the two point clouds.
        """
        e = np.mean(np.sqrt((pc1[0] - pc2[0]) ** 2 + (pc1[1] - pc2[1]) ** 2))
        return e

    def icp(self, pc1, pc2, T, max_iterations=100, tolerance=0.05, epsilon=1e-8):
        """
        Perform the iterative closest point (ICP) algorithm to align two point clouds.

        Args:
            pc1 (numpy.ndarray): The first point cloud as a 2D array of shape (2, N).
            pc2 (numpy.ndarray): The second point cloud as a 2D array of shape (2, N).
            T (numpy.ndarray): The initial transformation matrix.
            max_iterations (int): The maximum number of iterations for ICP.
            tolerance (float): The tolerance for convergence.
            epsilon (float): A small positive constant for regularization.

        Returns:
            numpy.ndarray: The rotation matrix.
            numpy.ndarray: The translation vector.
        
        Raises:
            ValueError: If ICP does not converge.
        """
        # Helper functions for homogeneous coordinates
        def Pi(pts_inhomogenous):
            # Converts homogeneous points to inhomogeneous coordinates.
            assert pts_inhomogenous.shape[0] == 3 or pts_inhomogenous.shape[0] == 4, "Input points must be 3xN or 4xN"
            tmp = pts_inhomogenous[:-1] / pts_inhomogenous[-1]
            return tmp

        def PiInv(pts_homogenous):
            # Converts inhomogeneous points to homogeneous coordinates.
            assert pts_homogenous.shape[0] == 2 or pts_homogenous.shape[0] == 3, "Input points must be 2xN or 3xN"
            tmp = np.vstack((pts_homogenous, np.ones(pts_homogenous.shape)[0]))
            return tmp

        # pc1 = Pi(T @ PiInv(pc1))
        for i in range(max_iterations):
            pc1 = (pc1 - np.mean(pc1, axis=1, keepdims=True)) / np.std(pc1, axis=1, keepdims=True)
            pc2 = (pc2 - np.mean(pc2, axis=1, keepdims=True)) / np.std(pc2, axis=1, keepdims=True)
            assert pc1.shape == pc2.shape, 'Point clouds must have the same shape'
            cov = np.cov(pc1 @ pc2.T)
            cov += np.eye(cov.shape[0]) * epsilon
            U, S, Vt = np.linalg.svd(cov)
            R = (U @ Vt).reshape(2, 2)
            t = np.mean(pc1, axis=1, keepdims=True) - (R @ np.mean(pc2, axis=1, keepdims=True))
            T = np.vstack((np.hstack((R, t)), np.array([[0, 0, 1]])))
            pc1 = Pi(T @ PiInv(pc1))

            if self.RMSE(pc1, pc2) < tolerance:
                return R, t
 
        print(f'RMSE: {self.RMSE(pc1, pc2)}')
        raise ValueError('ICP did not converge')

    def cmd_vel_callback(self, cmd_vel_msg):
        """
        Callback function for the `/cmd_vel` topic.

        Args:
            cmd_vel_msg (geometry_msgs.msg.Twist): The velocity command message.
        """
        self.cmd_vel_msg = cmd_vel_msg
        self.linear_velocity_mps = cmd_vel_msg.linear.x
        self.angular_velocity_radps = cmd_vel_msg.angular.z

    def scan_callback(self, scan_msg):
        """
        Callback function for the `/scan` topic.

        Args:
            scan_msg (sensor_msgs.msg.LaserScan): The lidar scan message.
        """
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
            pc_1 = self.process_scan(self.scan_msg_prev)
            pc_2 = self.process_scan(scan_msg)
            R, t = self.icp(pc_1, pc_2, T)
            # T = np.vstack((np.hstack((R, t)), np.array([[0, 0, 1]])))
            print(f'Translation Vector Matrix {t.shape}: \n {t}')
            x = self.x0 + delta_x + t[0][0]
            y = self.y0 + delta_y + t[1][0]
            z = self.z0
            theta = self.theta0 + delta_theta + np.arctan2(R[1, 0], R[0, 0])

            # Debugging
            pred_pose = np.array([x, y, z, theta]).reshape(-1, 1)
            gt = self.wait_for_transform("odom", "base_scan")
            gt_pose = np.array([gt.transform.translation.x, gt.transform.translation.y, gt.transform.translation.z, gt.transform.rotation.z]).reshape(-1, 1)
            print(f'Predicted Pose: \n {pred_pose}')
            print(f'Ground Truth Pose: \n {gt_pose}')
        else:
            x = self.x0
            y = self.y0
            z = self.z0
            theta = self.theta0

        # Create Odometry message
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'lidar_odom'
        odom.child_frame_id = '???'  # Replace with the correct child frame id
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.position.z = float(z)
        odom.pose.pose.orientation.z = float(theta)
        odom.twist.twist = self.cmd_vel_msg
        # Publish the estimated pose
        self.pose_pub.publish(odom)

        # Update the previous scan message
        self.t0 = self.t0 + delta_t
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
