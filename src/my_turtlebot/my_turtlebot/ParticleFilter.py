
from email.mime import image
from pyexpat import features
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

        self.n_particles = 1
        self.n_features = 4
        self.max_range = None
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
    def map_callback(self, map_msg):
        self.map = map_msg
        print('Map received')

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
        print(f'PC Shape: {pc.shape}')

        # Viz In Map 
        if self.map is not None:
            img = 255 - np.array(self.map.data).reshape(self.map.info.height,self.map.info.width)
            robot_gridpose = self.point_to_map(np.array([self.x0, self.y0]).reshape(2,1))

            print(f'Robot Pose Grid {robot_gridpose.shape}: {robot_gridpose}')
            plt.arrow(robot_gridpose[0][0], robot_gridpose[1][0], 0.5 * np.cos(self.theta0), 0.5 * np.sin(self.theta0), head_width=5, head_length=5, fc='green', ec='green')  # Plot orientation arrow
            plt.imshow(img,cmap ='gray')

            for i,point in enumerate(self.particles):
                particle_grid = self.point_to_map(point)
                plt.arrow(particle_grid[0][0], particle_grid[1][0], 0.5 * np.cos(point[3]), 0.5 * np.sin(point[3]), head_width=3, head_length=3, fc='red', ec='red')  # Plot orientation arrow

            features = self.particle_features(scan_msg)
            print(f'PC Shape: {pc.shape}')
            pc_grid = self.point_to_map(pc).T

            feature_indices = [round(np.rad2deg(ang)) for ang in np.linspace(0, np.pi, self.n_features)][:-1]
            feature_pc = pc_grid[feature_indices]
            print(f'Feature Indices: {feature_indices}')

            print(f'pc features {feature_pc.shape}: \n{feature_pc}')

            print(f'Features {features.shape}: \n{features}')

            
            plt.show()

        return pc
    
    def point_to_map(self, pc):
        """
        Convert a point in the map frame to the grid coordinates.

        Args:
            point (numpy.ndarray): The point as a 1D array of shape (2,).

        Returns:
            numpy.ndarray: The grid coordinates as a 1D array of shape (2,).
        """
        pc_grid = np.array([self.map.info.width/2,self.map.info.height/2]).reshape(2,1) + pc/self.map.info.resolution
        return pc_grid
    

    def get_line_intersection(self, particle_pose):
        features = []
        for i,alpha in enumerate(np.linspace(0, np.pi,self.n_features)[:-1]):#, np.pi, 3*np.pi/2]:
            # if alpha == np.pi:
            #     break
            #alpha -= (2*np.pi)%particle_pose[2]
            print(f'Alpha {i}: {np.rad2deg(alpha)}')
            if alpha in [0, 2 *np.pi]:
                l = np.array([0, 1, -particle_pose[1]])
                self.DrawLine(l, (self.map.info.height, self.map.info.width))
                for x in range(int(particle_pose[0]), self.map.info.width):
                    y = -l[2] - l[0] * x
                    if y >= 0 and y < self.map.info.height:
                        if self.map.data[int(x + y * self.map.info.width)] == 100:
                            #print(f'Intersection at: {x, y}')
                            #features.append(([alpha, np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)]))
                            features.append(([x, y]))
                            plt.scatter(x, y, c='blue')
                            break
                    # if x == self.map.info.width - 1:
                    #     features.append([max_range*np.cos(alpha), max_range*np.sin(alpha)])
                for x in range(int(particle_pose[0]),0,-1):
                    y = -l[2] - l[0] * x
                    if y >= 0 and y < self.map.info.height:
                        if self.map.data[int(x + y * self.map.info.width)] == 100:
                            #print(f'Intersection at: {x, y}')
                            #features.append([alpha, np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)])
                            features.append(([x, y]))
                            plt.scatter(x, y, c='blue')
                            break
                    # if x == 0 + 1:
                    #     features.append([max_range*np.cos(alpha), max_range*np.sin(alpha)])
                    
                # features.append([self.max_range, self.max_range])
            elif alpha in [np.pi/2, 3*np.pi/2]:
                l = np.array([1, 0, -particle_pose[0]])
                self.DrawLine(l, (self.map.info.height, self.map.info.width))
                # print(f'Particle Pose {type(particle_pose)}: {particle_pose}')
                # print(f'Map Height {type(self.map.info.height)}: {self.map.info.height}')
                for y in range(int(particle_pose[1]), self.map.info.height):
                    x = -l[2] - l[1] * y
                    print(f'X: {x}, Y: {y}')
                    if x >= 0 and x < self.map.info.width:
                        if self.map.data[int(x + round(y * self.map.info.width))] == 100:
                            #print(f'Intersection at: {x, y}')
                            #features.append([alpha, np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)])
                            features.append(([x, y]))
                            plt.scatter(x, y, c='blue')
                            break
                            
                    # if y == self.map.info.height - 1:
                    #     features.append([max_range*np.cos(alpha), max_range*np.sin(alpha)])
                for y in range(int(particle_pose[1]),0,-1):
                    x = -l[2] - l[1] * y
                    if x >= 0 and x < self.map.info.width:
                        if self.map.data[int(x + round(y * self.map.info.width))] == 100:
                            #print(f'Intersection at: {x, y}')
                            #features.append([alpha, np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)])
                            features.append(([x, y]))
                            plt.scatter(x, y, c='blue')
                            break
                # features.append([self.max_range, self.max_range])
                    # if y == 0 + 1:
                    #     features.append([max_range*np.cos(alpha), max_range*np.sin(alpha)])
            else:
                a = np.tan(alpha)
                b = particle_pose[1] - (a * particle_pose[0])
                l = np.array([a, -1, b])
                self.DrawLine(l, (self.map.info.height, self.map.info.width))

                for x in range(round(particle_pose[0]), self.map.info.width):
                    y = round(a*x + b)
                    if y >= 0 and y < self.map.info.height:
                        if self.map.data[int(x + round(y * self.map.info.width))] == 100:
                            #print(f'Intersection at: {x, y}')
                            #features.append([alpha, np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)])
                            features.append(([x, y]))
                            plt.scatter(x, y, c='blue')
                            break   
                    # if x == self.map.info.width - 1:
                    #     features.append([max_range*np.cos(alpha), max_range*np.sin(alpha)])
                for x in range(round(particle_pose[0]),0,-1):
                    y = round(a*x + b)
                    if y >= 0 and y < self.map.info.height:
                        if self.map.data[int(x + round(y * self.map.info.width))] == 100:
                            #print(f'Intersection at: {x, y}')
                            #features.append([alpha, np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)])
                            features.append(([x, y]))
                            plt.scatter(x, y, c='blue')
                            break

                #plt.show()
        features = np.vstack(features)
        return features
            

    def particle_features(self,scan_msg):
        """
        Extract features from the particle pose.

        Args:
            particle_pose (numpy.ndarray): The particle pose as a 1D array of shape (3,).

        Returns:
            numpy.ndarray: The extracted features as a 1D array of shape (2,).
        """

        features = []
        for particle in self.particles:
            particle_pose = self.point_to_map(particle)
            particle_orientation = particle[2]
            particle_pose = np.array([particle_pose[0][0], particle_pose[1][0], particle_orientation])
            
            print(f'Unique Values in Map: {np.unique(self.map.data)}')
            #particle_pose = np.array([100, self.map.info.height - 10, particle_pose[3]])
            #alpha_range = 
            print(f'Alpha Range: {np.rad2deg(np.linspace(0, np.pi,self.n_features))}')
            features.append(self.get_line_intersection(particle_pose))
            
        features = np.vstack(features)
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



def main(args=None):
    rclpy.init(args=args)
    localization_node = ParticleFilter()
    rclpy.spin(localization_node)
    localization_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
