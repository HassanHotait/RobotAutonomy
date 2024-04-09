
from turtle import position
import rclpy
from rclpy.node import Node
from nav2_msgs.msg import ParticleCloud
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import Pose
import tf2_ros
import numpy as np

class ParticleFilter(Node):
    """
    A ROS2 node for localization using particle filter.
    """

    def __init__(self):
        super().__init__('ParticleFilter')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # Create a QoS profile with reliability set to the same as the publisher
        

        # Subscribe to the '/particle_cloud' topic with the specified QoS profile
        qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.cmd_sub = self.create_subscription(ParticleCloud, '/particle_cloud', self.check_particle_cloud, qos)
        # self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        # self.pose_pub = self.create_publisher(Odometry, '/odom', 1)

        # transform_scanner = self.wait_for_transform('odom', 'base_scan')

        # # Initialize variables
        # self.x0 = transform_scanner.transform.translation.x
        # self.y0 = transform_scanner.transform.translation.y
        # self.z0 = transform_scanner.transform.translation.z
        # self.theta0 = transform_scanner.transform.rotation.z
        # self.t0 = self.get_clock().now().nanoseconds / 1e9
        # self.scan_msg_prev = None
        # self.lidar_odom = None
        # self.linear_velocity_mps = 0.0
        # self.angular_velocity_radps = 0.0
        # self.cmd_vel_msg = Twist()

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
    
    def unpack_pose(self,pose_msg):
        # Extract position and orientation from the pose message
        position = np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
        orientation = np.array([pose_msg.orientation.x, pose_msg.orientation.y,
                                pose_msg.orientation.z, pose_msg.orientation.w])
        return position, orientation
    def pack_pose(self, position, orientation):
        pose_msg = Pose()
        # Set position components
        pose_msg.position.x = position[0]
        pose_msg.position.y = position[1]
        pose_msg.position.z = position[2]
        # Set orientation components (quaternion)
        pose_msg.orientation.x = orientation[0]
        pose_msg.orientation.y = orientation[1]
        pose_msg.orientation.z = orientation[2]
        pose_msg.orientation.w = orientation[3]
        return pose_msg

    def check_particle_cloud(self, particle_cloud_msg):
        """
        Callback function for the `/cmd_vel` topic.

        Args:
            cmd_vel_msg (geometry_msgs.msg.Twist): The velocity command message.
        """
        # self.cmd_vel_msg = cmd_vel_msg
        # self.linear_velocity_mps = cmd_vel_msg.linear.x
        # self.angular_velocity_radps = cmd_vel_msg.angular.z
        
        position = 0
        orientation = 0
        for i in range(len(particle_cloud_msg.particles)):
            # self.get_logger().info(f"Particle {i}: pose x  {i}: {particle_cloud_msg.particles[i].pose.position.x}")
            # self.get_logger().info(f"Particle {i}: pose y  {i}: {particle_cloud_msg.particles[i].pose.position.y}")
            # self.get_logger().info(f"Particle {i}: pose z  {i}: {particle_cloud_msg.particles[i].pose.position.z}")
            # self.get_logger().info(f"Particle {i}: orientation x  {i}: {particle_cloud_msg.particles[i].pose.orientation.x}")
            # self.get_logger().info(f"Particle {i}: orientation y  {i}: {particle_cloud_msg.particles[i].pose.orientation.y}")
            # self.get_logger().info(f"Particle {i}: orientation z  {i}: {particle_cloud_msg.particles[i].pose.orientation.z}")
            # self.get_logger().info(f"Particle {i}: orientation w  {i}: {particle_cloud_msg.particles[i].pose.orientation.w}")
            # self.get_logger().info(f"Particle {i}: weight  {i}: {particle_cloud_msg.particles[i].weight}")
            weight = particle_cloud_msg.particles[i].weight
            position, orientation = self.unpack_pose(particle_cloud_msg.particles[i].pose)
            position += position * weight
            orientation += orientation * weight
        
        position = position / len(particle_cloud_msg.particles)
        orientation = orientation / len(particle_cloud_msg.particles)
        pose = self.pack_pose(position, orientation)

        print('------------------------------------')
        print(f'Position x : {position[0]}')
        print(f'Position y : {position[1]}')
        print(f'Position z : {position[2]}')
        print(f'Orientation z : {orientation[2]}')
        print(f'Orientation w : {orientation[3]}')


        # print(f'Pose: {pose}')





def main(args=None):
    rclpy.init(args=args)
    localization_node = ParticleFilter()
    rclpy.spin(localization_node)
    localization_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
