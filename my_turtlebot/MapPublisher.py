

import cmd
from os import wait
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry,OccupancyGrid
from geometry_msgs.msg import TransformStamped
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from scipy.spatial.distance import cdist

class MappingNode(Node):
    def __init__(self):
        super().__init__('MapPublisher')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 2)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map2', 2)

        transform_scanner = self.wait_for_transform('odom', 'base_scan')
        transform_map = self.wait_for_transform('odom')


        print(f'Transform Scanner: \n {transform_scanner}')


        self.x0 = transform_scanner.transform.translation.x
        self.y0 = transform_scanner.transform.translation.y
        self.z0 = transform_scanner.transform.translation.z
        self.theta0 = transform_scanner.transform.rotation.z
        self.t0 = self.get_clock().now().nanoseconds / 1e9

        self.scan_msg_prev = None

        self.linear_velocity_mps = 0.0
        self.angular_velocity_radps = 0.0

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Create a transform
        self.static_transform = TransformStamped()
        self.static_transform.header.stamp = self.get_clock().now().to_msg()
        self.static_transform.header.frame_id = 'odom'
        self.static_transform.child_frame_id = 'map'
        self.static_transform.transform.translation.x = 0.0
        self.static_transform.transform.translation.y = 0.0
        self.static_transform.transform.translation.z = 0.0
        self.static_transform.transform.rotation.x = 0.0
        self.static_transform.transform.rotation.y = 0.0
        self.static_transform.transform.rotation.z = 0.0
        self.static_transform.transform.rotation.w = 1.0

    def wait_for_transform(self, target_frame, source_frame = None):
        while True:
            try:
                if source_frame is not None:
                    trans = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
                else:
                    trans = self.tf_buffer.lookup_transform(target_frame,target_frame, rclpy.time.Time())
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


        # Create a 2D occupancy grid
        occupancy_grid = np.zeros((100, 100), dtype=np.int8)
        # Fill random cells with value 100
        random_indices = np.random.randint(0, 100, size=(50, 2))
        occupancy_grid[random_indices[:, 0], random_indices[:, 1]] = int(100)

        # Create an OccupancyGrid message and fill in the information
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = 'map'
        map_msg.info.resolution = 0.05
        map_msg.info.width = 400
        map_msg.info.height = 400
        map_msg.info.origin.position.x =0 - (map_msg.info.width * map_msg.info.resolution/2)  # adjust according to your map size
        map_msg.info.origin.position.y = 0 - (map_msg.info.height * map_msg.info.resolution/2)
        map_msg.info.origin.position.z = 0.0
        # map_msg.info.origin.orientation.x = 0.0
        # map_msg.info.origin.orientation.y = 0.0
        # map_msg.info.origin.orientation.z = 0.0
        # map_msg.info.origin.orientation.w = 1.0
        map_msg.data = list([0]*map_msg.info.width*map_msg.info.height)

        # Publish the map
        self.map_pub.publish(map_msg)

        # Publish the static transform
        self.tf_broadcaster.sendTransform(self.static_transform)

def main(args=None):
    rclpy.init(args=args)
    localization_node = MappingNode()
    rclpy.spin(localization_node)
    localization_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
