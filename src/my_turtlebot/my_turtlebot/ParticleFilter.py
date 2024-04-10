
from email.mime import image
from turtle import position

from matplotlib import cm
import rclpy
from rclpy.node import Node
from nav2_msgs.msg import ParticleCloud
from nav_msgs.msg import Odometry,OccupancyGrid
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan
import tf2_ros
from scipy.spatial.transform import Rotation
import numpy as np
import matplotlib.pyplot as plt
import imageio

class ParticleFilter(Node):
    """
    A ROS2 node for localization using particle filter.
    """

    def __init__(self):
        super().__init__('ParticleFilter')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.n_particles = 10
        self.particles = np.zeros((self.n_particles, 4)) 
        self.map = None#np.array(imageio.imread('/home/ubuntu/Desktop/RobotAutonomy/src/my_turtlebot/maps/map.pgm'))

        # Create a QoS profile with reliability set to the same as the publisher
        

        # Subscribe to the '/particle_cloud' topic with the specified QoS profile
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.process_scan, 10)
        map_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, history = QoSHistoryPolicy.KEEP_LAST)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.cmd_sub = self.create_subscription(Pose, '/cmd_vel', self.motion_model_update, 10)
        self.particle_pub = self.create_publisher(ParticleCloud, '/particle_cloud', 10)
        self.pose_pub = self.create_publisher(Odometry, '/amcl_pose', 1)

        transform_scanner = self.wait_for_transform('odom', 'base_scan')

        # # Initialize variables
        self.x0 = transform_scanner.transform.translation.x
        self.y0 = transform_scanner.transform.translation.y
        self.z0 = transform_scanner.transform.translation.z
        self.theta0 = transform_scanner.transform.rotation.z
        self.t0 = self.get_clock().now().nanoseconds / 1e9

        # self.scan_msg_prev = None
        # self.lidar_odom = None
        # self.linear_velocity_mps = 0.0
        # self.angular_velocity_radps = 0.0
        # self.cmd_vel_msg = Twist()

        self.init_particles()
    def map_callback(self, map_msg):
        self.map = map_msg

        print('Map received')

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

        # Viz In Map 
        if self.map is not None:
            img = 255 - np.array(self.map.data).reshape(self.map.info.height,self.map.info.width)
            pt_grid = self.point_to_map(np.array([self.x0, self.y0, self.theta0]))

            plt.arrow(pt_grid[0], pt_grid[1], 0.5 * np.cos(pt_grid[3]), 0.5 * np.sin(pt_grid[3]), head_width=5, head_length=5, fc='green', ec='green')  # Plot orientation arrow
            plt.imshow(img,cmap ='gray')

            


            for point in self.particles:
                particle_grid = self.point_to_map(point)
                plt.arrow(particle_grid[0], particle_grid[1], 0.5 * np.cos(particle_grid[3]), 0.5 * np.sin(particle_grid[3]), head_width=3, head_length=3, fc='red', ec='red')  # Plot orientation arrow

                self.particle_features(particle_grid)



        return pc
    

    def init_particles(self):
        """
        Initialize the particles.
        """
        for i in range(self.n_particles):
            x = self.x0 + np.random.uniform(-1, 1)
            y = self.y0 + np.random.uniform(-1, 1)
            theta = self.theta0 + np.random.uniform(-np.pi/4, np.pi/4)
            weight = 1.0 / self.n_particles
            self.particles[i] = np.array([x, y, theta, weight]).reshape(1, 4)

    def DrawLine(self,l, shape):
    #Checks where the line intersects the four sides of the image
    # and finds the two intersections that are within the frame
        def in_frame(l_im):
            q = np.cross(l.flatten(), l_im)
            q = q[:2]/q[2]
            if all(q>=0) and all(q+1<=shape[1::-1]):
                return q
        lines = [[1, 0, 0], [0, 1, 0], [1, 0, 1-shape[1]], [0, 1, 1-shape[0]]]
        P = [in_frame(l_im) for l_im in lines if in_frame(l_im) is not None]
        if (len(P)==0):
            print("Line is completely outside image")
        plt.plot(*np.array(P).T)

    def point_to_map(self, point):
        """
        Convert a point in the map frame to the grid coordinates.

        Args:
            point (numpy.ndarray): The point as a 1D array of shape (2,).

        Returns:
            numpy.ndarray: The grid coordinates as a 1D array of shape (2,).
        """
        x = int((self.map.info.width / 2) + (point[0] / self.map.info.resolution))
        y = int((self.map.info.height / 2) + (point[1] / self.map.info.resolution))
        return np.array([x, y,1, point[2]])

    def particle_features(self, particle_pose):
        """
        Extract features from the particle pose.

        Args:
            particle_pose (numpy.ndarray): The particle pose as a 1D array of shape (3,).

        Returns:
            numpy.ndarray: The extracted features as a 1D array of shape (2,).
        """

        features = np.zeros((360, 1))
        print(f'Unique Values in Map: {np.unique(self.map.data)}')
        #particle_pose = np.array([100, self.map.info.height - 10, particle_pose[3]])
        for i,alpha in enumerate(np.linspace(0, 2*np.pi/4,360)):#, np.pi, 3*np.pi/2]:

            alpha += 2*np.pi%particle_pose[3]
            if alpha in [0, 2 *np.pi]:
                l = np.array([0, 1, -particle_pose[1]])
                #self.DrawLine(l, (self.map.info.height, self.map.info.width))
                for x in range(int(particle_pose[0]), self.map.info.width):
                    y = -l[2] - l[0] * x
                    if y >= 0 and y < self.map.info.height:
                        if self.map.data[int(x + y * self.map.info.width)] == 100:
                            #print(f'Intersection at: {x, y}')
                            features[i] = np.array([alpha, np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)])
                            plt.scatter(x, y, c='blue')
                            break
                for x in range(int(particle_pose[0]),0,-1):
                    y = -l[2] - l[0] * x
                    if y >= 0 and y < self.map.info.height:
                        if self.map.data[int(x + y * self.map.info.width)] == 100:
                            #print(f'Intersection at: {x, y}')
                            features[i] = np.array([alpha, np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)])
                            plt.scatter(x, y, c='blue')
                            break
            elif alpha in [np.pi/2, 3*np.pi/2]:
                l = np.array([1, 0, -particle_pose[0]])
                #self.DrawLine(l, (self.map.info.height, self.map.info.width))
                # print(f'Particle Pose {type(particle_pose)}: {particle_pose}')
                # print(f'Map Height {type(self.map.info.height)}: {self.map.info.height}')
                for y in range(int(particle_pose[1]), self.map.info.height):
                    x = -l[2] - l[1] * y
                    print(f'X: {x}, Y: {y}')
                    if x >= 0 and x < self.map.info.width:
                        if self.map.data[int(x + y * self.map.info.width)] == 100:
                            #print(f'Intersection at: {x, y}')
                            features[i] = np.array([alpha, np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)])
                            plt.scatter(x, y, c='blue')
                            break
                for y in range(int(particle_pose[1]),0,-1):
                    x = -l[2] - l[1] * y
                    if x >= 0 and x < self.map.info.width:
                        if self.map.data[int(x + y * self.map.info.width)] == 100:
                            #print(f'Intersection at: {x, y}')
                            features[i] = np.array([alpha, np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)])
                            plt.scatter(x, y, c='blue')
                            break
            else:
                a = np.tan(alpha)
                b = particle_pose[1] - (a * particle_pose[0])
                l = np.array([a, -1, b])
                self.DrawLine(l, (self.map.info.height, 200))#self.map.info.width))

                for x in range(round(particle_pose[0]), self.map.info.width):
                    y = round(a*x + b)
                    if y >= 0 and y < self.map.info.height:
                        if self.map.data[round(x + (y * self.map.info.width))] == 100:
                            #print(f'Intersection at: {x, y}')
                            features[i] = np.array([alpha, np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)])
                            plt.scatter(x, y, c='blue')
                            break   
                for x in range(round(particle_pose[0]),0,-1):
                    y = round(a*x + b)
                    if y >= 0 and y < self.map.info.height:
                        if self.map.data[round(x + (y * self.map.info.width))] == 100:
                            #print(f'Intersection at: {x, y}')
                            features[i] = np.array([alpha, np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)])
                            plt.scatter(x, y, c='blue')
                            break

                #print(f'No Intersection Found')
            # a = np.tan(alpha)
            # b = particle_pose[1] - (a * particle_pose[0])
            # self.DrawLine(np.array([1, -1, 0]), (self.map.info.height, self.map.info.width))

        plt.show()
        return features
    def motion_model_update(self,cmd_vel_msg):
        delta_t = (self.get_clock().now().nanoseconds / 1e9 - self.t0)
        delta_x = (cmd_vel_msg.linear.x * np.cos(self.theta0) * delta_t)
        delta_y = (cmd_vel_msg.linear.x * np.sin(self.theta0) * delta_t)
        delta_theta = cmd_vel_msg.angular.z * delta_t

        self.particles[:,0] +=  delta_x
        self.particles[:,1] +=  delta_y
        self.particles[:,2] +=  delta_theta

        self.t0 = self.get_clock().now().nanoseconds / 1e9
        self.x0 += delta_x
        self.y0 += delta_y
        self.theta0 += delta_theta

        #self.measurement_model_update()

    def measurement_model_update(self,pc):
        if self.map is not None:
            img = np.array(self.map.data).reshape(self.map.info.height,self.map.info.width)
            plt.imshow(img)
            plt.show()

            for i in range(self.n_particles):
                pass
            
        
        





def main(args=None):
    rclpy.init(args=args)
    localization_node = ParticleFilter()
    rclpy.spin(localization_node)
    localization_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
