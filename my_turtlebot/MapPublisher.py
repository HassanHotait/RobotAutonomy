

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
from scipy.spatial.transform import Rotation

class MappingNode(Node):
    def __init__(self):
        super().__init__('MapPublisher')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map2', 10)

        # transform_scanner = self.wait_for_transform('odom', 'base_scan')
        # transform_map = self.wait_for_transform('odom')

        # self.x0 = transform_scanner.transform.translation.x
        # self.y0 = transform_scanner.transform.translation.y
        # self.z0 = transform_scanner.transform.translation.z
        # self.theta0 = transform_scanner.transform.rotation.z
        self.t0 = self.get_clock().now().nanoseconds / 1e9
        self.scan_msg_prev = None
        self.linear_velocity_mps = 0.0
        self.angular_velocity_radps = 0.0
        self.resolution = 0.05
        self.width_m = 30
        self.height_m = 10


        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.map_msg0 = OccupancyGrid()
        self.map_msg0.header.stamp = self.get_clock().now().to_msg()
        self.map_msg0.header.frame_id = 'map'
        self.map_msg0.info.resolution = self.resolution
        self.map_msg0.info.width = int(self.width_m / self.resolution)
        self.map_msg0.info.height = int(self.height_m / self.resolution)
        self.map_msg0.info.origin.position.x = -self.width_m / 2
        self.map_msg0.info.origin.position.y = -self.height_m / 2
        self.map_msg0.info.origin.position.z = 0.0
        self.map_msg0.info.origin.orientation.x = 0.0
        self.map_msg0.info.origin.orientation.y = 0.0
        self.map_msg0.info.origin.orientation.z = 0.0
        self.map_msg0.info.origin.orientation.w = 1.0
        self.map_msg0.data = [0] * self.map_msg0.info.width * self.map_msg0.info.height
        

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
                pass
            else:
                pc_x.append(range * np.cos(scan_msg.angle_min + (i * scan_msg.angle_increment)))
                pc_y.append(range * np.sin(scan_msg.angle_min + (i * scan_msg.angle_increment)))

        # Return Point Cloud (2, N)
        pc_x = np.hstack(pc_x)
        pc_y = np.hstack(pc_y)
        pc = np.vstack((pc_x, pc_y))

        return pc
    
    def point_to_grid(self, pc, map_msg,transform):
        """
        Converts a point cloud to a grid representation based on the given map message.

        Args:
            pc (numpy.ndarray): The point cloud to convert.
            map_msg (MapMessage): The map message containing information about the grid.

        Returns:
            list: The grid data as a flattened list.

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
        
        data = np.array(map_msg.data).reshape(map_msg.info.height, map_msg.info.width)
        R = Rotation.from_euler('z', transform.transform.rotation.z).as_matrix()

        # pc_r = R @ PiInv(pc) + np.array([transform.transform.translation.x, transform.transform.translation.y, 0]).reshape(3,1)

        for pt in pc.T:


            x = int((map_msg.info.width / 2) + (pt[0] / map_msg.info.resolution) + (transform.transform.translation.x / map_msg.info.resolution))
            y = int((map_msg.info.height / 2) + (pt[1] / map_msg.info.resolution) + (transform.transform.translation.y / map_msg.info.resolution))

            # pt_ro= R @ np.array([x, y,1]).reshape(3,1)
  
            if x >= 0 and x < map_msg.info.width and y >= 0 and y < map_msg.info.height:
                data[y, x] = 100

        data = data.flatten().tolist()

        return data
    
    def scan_callback(self, scan_msg):

        # Delta Time
        # delta_t = (self.get_clock().now().nanoseconds / 1e9 - self.t0)
        # delta_x = (self.linear_velocity_mps * np.cos(self.theta0) * delta_t)
        # delta_y = (self.linear_velocity_mps * np.sin(self.theta0) * delta_t)
        # delta_theta = self.angular_velocity_radps * delta_t

        pc = self.process_scan(scan_msg)
        # print(f'Point Cloud Mean {pc.shape}: \n {mu}')

        # Random Map Exercise
        # # Create a 2D occupancy grid
        # occupancy_grid = np.zeros((100, 100), dtype=np.int8)
        # # Fill random cells with value 100
        # random_indices = np.random.randint(0, 100, size=(50, 2))
        # occupancy_grid[random_indices[:, 0], random_indices[:, 1]] = int(100)
        transform = self.wait_for_transform('odom', 'base_scan')

        # Create an OccupancyGrid message and fill in the information
        map_msg = self.map_msg0
        map_msg.header.stamp = self.get_clock().now().to_msg()
        # map_msg.info.origin.orientation.x = transform.transform.rotation.x
        # map_msg.info.origin.orientation.y = transform.transform.rotation.y
        # map_msg.info.origin.orientation.z = transform.transform.rotation.z
        # map_msg.info.origin.orientation.w = transform.transform.rotation.w

        map_msg.data = self.point_to_grid(pc,map_msg,transform)#list([0]*map_msg.info.width*map_msg.info.height)
        # Publish the map
        self.map_pub.publish(map_msg)
        # Publish the static transform
        self.tf_broadcaster.sendTransform(self.static_transform)

        self.map_msg0 = map_msg

def main(args=None):
    rclpy.init(args=args)
    localization_node = MappingNode()
    rclpy.spin(localization_node)
    localization_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
