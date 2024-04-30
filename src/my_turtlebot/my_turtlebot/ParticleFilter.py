
from email.mime import base, image
from operator import gt
from pyexpat import features
from turtle import position

from matplotlib import cm
import rclpy
from rclpy.node import Node
from nav2_msgs.msg import Particle,ParticleCloud
from nav_msgs.msg import Odometry,OccupancyGrid, Path
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Pose, TransformStamped, Twist, PoseStamped
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
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.n_particles = 50
        self.n_features = 2
        self.max_range = None
        self.particles = np.zeros((self.n_particles, 4)) 
        self.map = None#np.array(imageio.imread('/home/ubuntu/Desktop/RobotAutonomy/src/my_turtlebot/maps/map.pgm'))

        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'
        self.gt_path_msg = Path()
        self.gt_path_msg.header.frame_id = 'odom'
        self.motion_model_path_msg = Path()
        self.motion_model_path_msg.header.frame_id = 'odom'

        self.path_list = []
        self.absolute_pose = None
        self.motion_model_pose = None
        self.gt_pose = None
        self.gt_path_list = []
        self.motion_model_path_list = []

        transform_scanner = self.wait_for_transform('odom', 'base_scan')
        self.base2scan = self.wait_for_transform('base_footprint', 'base_scan')

        self.gt_array = np.empty((2,1))
        self.mm_array = np.empty((2,1))
        self.pf_array = np.empty((2,1))

        # # Initialize variables
        self.x0 = transform_scanner.transform.translation.x
        self.y0 = transform_scanner.transform.translation.y
        self.z0 = transform_scanner.transform.translation.z
        self.theta0 = np.arctan2(Rotation.from_quat([transform_scanner.transform.rotation.x,transform_scanner.transform.rotation.y,transform_scanner.transform.rotation.z,transform_scanner.transform.rotation.w]).as_matrix()[1, 0], Rotation.from_quat([transform_scanner.transform.rotation.x,transform_scanner.transform.rotation.y,transform_scanner.transform.rotation.z,transform_scanner.transform.rotation.w]).as_matrix()[0, 0])

        # # Initialize Motion Model Variables
        self.x_mm = transform_scanner.transform.translation.x
        self.y_mm = transform_scanner.transform.translation.y
        self.z_mm = transform_scanner.transform.translation.z
        #self.theta_mm = np.arctan2(Rotation.from_quat([transform_scanner.transform.rotation.x,transform_scanner.transform.rotation.y,transform_scanner.transform.rotation.z,transform_scanner.transform.rotation.w]).as_matrix()[1, 0], Rotation.from_quat([transform_scanner.transform.rotation.x,transform_scanner.transform.rotation.y,transform_scanner.transform.rotation.z,transform_scanner.transform.rotation.w]).as_matrix()[0, 0])


        print(f'Init Pose: {self.x0, self.y0}')

        R = Rotation.from_quat([transform_scanner.transform.rotation.x,transform_scanner.transform.rotation.y,transform_scanner.transform.rotation.z,transform_scanner.transform.rotation.w]).as_matrix()
        self.rx = transform_scanner.transform.rotation.x
        self.ry = transform_scanner.transform.rotation.y
        self.theta0 = np.arctan2(R[1, 0], R[0, 0])
        self.rw = transform_scanner.transform.rotation.w
        self.t0 = self.get_clock().now().nanoseconds / 1e9

        print(f'Initialized Particles')
        self.init_particles()

        # Subscribe to the '/particle_cloud' topic with the specified QoS profile
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.process_scan, 50)
        map_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, history = QoSHistoryPolicy.KEEP_LAST)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.motion_model_update, 50)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.particle_pub = self.create_publisher(ParticleCloud, '/particle_cloud', 10)
        self.pose_pub = self.create_publisher(Odometry, '/amcl_pose', 10)
        self.path_pub = self.create_publisher(Path, '/path', 10)
        self.gt_path_pub = self.create_publisher(Path, '/gt_path', 10)
        self.motion_model_path_pub = self.create_publisher(Path, '/motion_model_path', 10)

    def init_particles(self):
        """
        Initialize the particles.
        """
        for i in range(self.n_particles):
            x = self.x0 + np.random.uniform(-0.5, 0.5)
            y = self.y0 + np.random.uniform(-0.5, 0.5)
            theta = self.theta0 + np.random.uniform(-np.pi/4, np.pi/4)
            weight = 1.0 / self.n_particles
            self.particles[i] = np.array([x, y, theta, weight]).reshape(1, 4)
    def map_callback(self, map_msg):
        self.map = map_msg
        #print('Map received')

    def process_scan(self, scan_msg):
        """
        Process a lidar scan message and convert it to a point cloud.

        Args:
            scan_msg (sensor_msgs.msg.LaserScan): The lidar scan message.

        Returns:
            numpy.ndarray: The point cloud as a 2D array of shape (2, N).
        """

        ## Process the scan message and convert it to a point cloud
        pc_x = []
        pc_y = []

        # Range to Point Cloud
        for i, r in enumerate(scan_msg.ranges):
            if r > scan_msg.range_max:
                pc_x.append(scan_msg.range_max * np.cos(scan_msg.angle_min + (i * scan_msg.angle_increment)))
                pc_y.append(scan_msg.range_max * np.sin(scan_msg.angle_min + (i * scan_msg.angle_increment)))
            else:
                pc_x.append(r * np.cos(scan_msg.angle_min + (i * scan_msg.angle_increment)))
                pc_y.append(r * np.sin(scan_msg.angle_min + (i * scan_msg.angle_increment)))

        # Return Point Cloud (2, N)
        pc_x = np.hstack(pc_x)
        pc_y = np.hstack(pc_y)
        pc = np.vstack((pc_x, pc_y))

        # # Viz In Map 
        # if self.map is not None:
        #     img = 255 - np.array(self.map.data).reshape(self.map.info.height,self.map.info.width)
        #     #np.save('map.npy', img)
        #     robot_gridpose = self.point_to_map(np.array([self.x0, self.y0]).reshape(2,1))

        #     #print(f'Robot Pose Grid {robot_gridpose.shape}: {robot_gridpose}')
        #     plt.arrow(robot_gridpose[0][0], robot_gridpose[1][0], 0.5 * np.cos(self.theta0), 0.5 * np.sin(self.theta0), head_width=5, head_length=5, fc='green', ec='green')  # Plot orientation arrow
        #     plt.imshow(img,cmap ='gray')


        #     for i,point in enumerate(self.particles):
        #         particle_grid = self.point_to_map(point[:2].reshape(2,1))
        #         # print(f'Particle Grid {i} {particle_grid.shape}: {particle_grid}')
        #         plt.arrow(particle_grid[0][0], particle_grid[1][0], 0.5 * np.cos(point[2]), 0.5 * np.sin(point[2]), head_width=3, head_length=3, fc='red', ec='red')  # Plot orientation arrow
                #print(f'Candidate Particle Plotted')

        if self.map is not None:
            # Get Features with Raycasting
            features = self.particle_features(scan_msg)
            #Absolute Pose to Grid Pose [m] -> [cells]
            pc_grid = self.point_to_map(pc).T

            # Update Weights
            weights = []
            for feature in features:
                feature_error = []
                for key, value in feature.items():

                    gt_range = scan_msg.ranges[key] if scan_msg.ranges[key] < scan_msg.range_max else scan_msg.range_max
                    value = value if value < scan_msg.range_max else scan_msg.range_max
                    error = np.abs(value - gt_range)
                    feature_error.append(error)
                feature_error = np.mean(feature_error)
                weights.append(feature_error)
            probabilities = self.normalize_errors(weights)
            self.particles[:,3] = probabilities

            # Publish Particle Cloud For RVIZ Visualization
            msg_particle_cloud = ParticleCloud()
            msg_particle_cloud.header.stamp = self.get_clock().now().to_msg()
            msg_particle_cloud.header.frame_id = 'odom'
            msg_particle_cloud.particles = []
            for particle in self.particles:
                p = Particle()
                p.pose.position.x = particle[0]
                p.pose.position.y = particle[1]
                p.pose.position.z = self.z0
                particle_rotations = Rotation.from_euler('z', particle[2]).as_quat()
                p.pose.orientation.x = particle_rotations[0]
                p.pose.orientation.y = particle_rotations[1]
                p.pose.orientation.z = particle_rotations[2]
                p.pose.orientation.w = particle_rotations[3]
                p.weight = particle[3]
                msg_particle_cloud.particles.append(p)
            self.particle_pub.publish(msg_particle_cloud)


            # weighted_likelihood_particle = np.average(self.particles[:,:2], axis=0, weights=self.particles[:,3])
            # weighted_pose = self.point_to_map(weighted_likelihood_particle.reshape(2,1))
            # weighted_orientation = np.average(self.particles[:,2], axis=0,weights=self.particles[:,3])
            # print(f'Weighted Likelihood Particle: {weighted_likelihood_particle}')
            # print(f'Weighted Pose Grid {weighted_pose.shape}: {weighted_pose}')
            # print(f'Weighted Orientation {weighted_orientation.shape}: {weighted_orientation}')
            self.absolute_pose =  [0, 0, 0]
            for i in range(len(msg_particle_cloud.particles)):
                weight = msg_particle_cloud.particles[i].weight
                self.absolute_pose[0] += msg_particle_cloud.particles[i].pose.position.x * weight
                self.absolute_pose[1] += msg_particle_cloud.particles[i].pose.position.y * weight
                self.absolute_pose[2] += Rotation.from_quat([msg_particle_cloud.particles[i].pose.orientation.x,msg_particle_cloud.particles[i].pose.orientation.y,msg_particle_cloud.particles[i].pose.orientation.z,msg_particle_cloud.particles[i].pose.orientation.w]).as_euler('xyz')[2] * weight

            self.x0 = self.absolute_pose[0]
            self.y0 = self.absolute_pose[1]
            self.theta0 = self.absolute_pose[2]

            # Publish Path For RVIZ Visualization
            path_pose = PoseStamped()
            path_pose.pose.position.x = self.absolute_pose[0]
            path_pose.pose.position.y = self.absolute_pose[1]
            path_pose.pose.position.z = self.z0
            self.path_list.append(path_pose)
            self.path_msg.poses = self.path_list
            self.path_msg.header.stamp = self.get_clock().now().to_msg()
            self.path_pub.publish(self.path_msg)

            #self.pf_array = np.vstack((self.pf_array, np.array([self.absolute_pose[0], self.absolute_pose[1]])))
            self.pf_array = np.hstack((self.pf_array, np.array([self.x0, self.y0]).reshape(2,1)))
            np.save('pf.npy', self.pf_array)

            # Publish Transform Frame
            
            self.static_transform = TransformStamped()
            self.static_transform.header.stamp = self.get_clock().now().to_msg()
            self.static_transform.header.frame_id = 'odom'
            self.static_transform.child_frame_id = 'amcl_basescan'
            self.static_transform.transform.translation.x = float(self.absolute_pose[0])
            self.static_transform.transform.translation.y = float(self.absolute_pose[1])
            self.static_transform.transform.translation.z = float(self.z0)
            R = Rotation.from_euler('z', self.absolute_pose[2]).as_quat()
            self.static_transform.transform.rotation.x = float(R[0])
            self.static_transform.transform.rotation.y = float(R[1])
            self.static_transform.transform.rotation.z = float(R[2])
            self.static_transform.transform.rotation.w = float(R[3])

            self.tf_broadcaster.sendTransform(self.static_transform)


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
        features = {}
        alpha_features = []
        for i,alpha in enumerate(np.linspace(0, np.pi,self.n_features + 1)[:-1]):#, np.pi, 3*np.pi/2]:
            # if alpha == np.pi:
            #     break
            alpha_features = []
            alpha += particle_pose[2]
            #print(f'Alpha {i}: {np.rad2deg(alpha)}')
            if alpha in [0, 2 *np.pi]:
                l = np.array([0, 1, -particle_pose[1]])
                self.DrawLine(l, (self.map.info.height, self.map.info.width))
                for x in range(int(particle_pose[0]), self.map.info.width):
                    y = -l[2] - l[0] * x
                    if y >= 0 and y < self.map.info.height:
                        if self.map.data[int(x + y * self.map.info.width)] == 100:
                            #print(f'Intersection at: {x, y}')
                            #features.append(([alpha, np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)]))
                            # alpha_features.append(([x, y]))
                            # plt.scatter(x, y, c='blue')
                            if np.dot([x - particle_pose[0], y - particle_pose[1]], [np.cos(alpha), np.sin(alpha)]) > 0:
                                features[round(np.rad2deg(alpha))] = np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)
                            else:
                                features[round(np.rad2deg(alpha + np.pi))] = np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)
                            break
                    # if x == self.map.info.width - 1:
                    #     features.append([max_range*np.cos(alpha), max_range*np.sin(alpha)])
                for x in range(int(particle_pose[0]),0,-1):
                    y = -l[2] - l[0] * x
                    if y >= 0 and y < self.map.info.height:
                        if self.map.data[int(x + y * self.map.info.width)] == 100:
                            #print(f'Intersection at: {x, y}')
                            #features.append([alpha, np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)])
                            # alpha_features.append(([x, y]))
                            # plt.scatter(x, y, c='blue')
                            if np.dot([x - particle_pose[0], y - particle_pose[1]], [np.cos(alpha), np.sin(alpha)]) > 0:
                                features[round(np.rad2deg(alpha))] = np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)
                            else:
                                features[round(np.rad2deg(alpha + np.pi))] = np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)
                            break
                    # if x == 0 + 1:
                    #     features.append([max_range*np.cos(alpha), max_range*np.sin(alpha)])
                    
                # features.append([self.max_range, self.max_range])
            elif alpha in [np.pi/2, 3*np.pi/2]:
                l = np.array([1, 0, -particle_pose[0]])
                #self.DrawLine(l, (self.map.info.height, self.map.info.width))
                # print(f'Particle Pose {type(particle_pose)}: {particle_pose}')
                # print(f'Map Height {type(self.map.info.height)}: {self.map.info.height}')
                for y in range(int(particle_pose[1]), self.map.info.height):
                    x = -l[2] - l[1] * y
                    print(f'X: {x}, Y: {y}')
                    if x >= 0 and x < self.map.info.width:
                        if self.map.data[int(x + round(y * self.map.info.width))] == 100:
                            #print(f'Intersection at: {x, y}')
                            #features.append([alpha, np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)])
                            # alpha_features.append(([x, y]))
                            # plt.scatter(x, y, c='blue')
                            if np.dot([x - particle_pose[0], y - particle_pose[1]], [np.cos(alpha), np.sin(alpha)]) > 0:
                                features[round(np.rad2deg(alpha))] = np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)
                            else:
                                features[round(np.rad2deg(alpha + np.pi))] = np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)
                            break
                            
                    # if y == self.map.info.height - 1:
                    #     features.append([max_range*np.cos(alpha), max_range*np.sin(alpha)])
                for y in range(int(particle_pose[1]),0,-1):
                    x = -l[2] - l[1] * y
                    if x >= 0 and x < self.map.info.width:
                        if self.map.data[int(x + round(y * self.map.info.width))] == 100:
                            #print(f'Intersection at: {x, y}')
                            #features.append([alpha, np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)])
                            #alpha_features.append(([x, y]))
                            #plt.scatter(x, y, c='blue')
                            if np.dot([x - particle_pose[0], y - particle_pose[1]], [np.cos(alpha), np.sin(alpha)]) > 0:
                                features[round(np.rad2deg(alpha))] = np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)
                            else:
                                features[round(np.rad2deg(alpha + np.pi))] = np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)
                            break
                # features.append([self.max_range, self.max_range])
                    # if y == 0 + 1:
                    #     features.append([max_range*np.cos(alpha), max_range*np.sin(alpha)])
            else:
                a = np.tan(alpha)
                b = particle_pose[1] - (a * particle_pose[0])
                l = np.array([a, -1, b])
                #self.DrawLine(l, (self.map.info.height, self.map.info.width))

                for x in range(round(particle_pose[0]), self.map.info.width):
                    y = round(a*x + b)
                    if y >= 0 and y < self.map.info.height:
                        if self.map.data[int(x + round(y * self.map.info.width))] == 100:
                            #print(f'Intersection at: {x, y}')
                            #features.append([alpha, np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)])
                            alpha_features.append(([x, y]))
                            if np.dot([x - particle_pose[0], y - particle_pose[1]], [np.cos(alpha), np.sin(alpha)]) > 0:
                                features[round(np.rad2deg(alpha))] = np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)
                            else:
                                features[round(np.rad2deg(alpha + np.pi))] = np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)
                            #plt.scatter(x, y, c='blue')
                            break   
                    # if x == self.map.info.width - 1
                    #     features.append([max_range*np.cos(alpha), max_range*np.sin(alpha)])
                for x in range(round(particle_pose[0]),0,-1):
                    y = round(a*x + b)
                    if y >= 0 and y < self.map.info.height:
                        if self.map.data[int(x + round(y * self.map.info.width))] == 100:
                            #print(f'Intersection at: {x, y}')
                            #features.append([alpha, np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)])
                            alpha_features.append(([x, y]))
                            if np.dot([x - particle_pose[0], y - particle_pose[1]], [np.cos(alpha), np.sin(alpha)]) > 0:
                                features[round(np.rad2deg(alpha))] = np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)
                            else:
                                features[round(np.rad2deg(alpha + np.pi))] = np.sqrt((x - particle_pose[0])**2 + (y - particle_pose[1])**2)
                            plt.scatter(x, y, c='blue')
                            break

                #plt.show()
        #features = np.vstack(features)
            #features[round(np.rad2deg(alpha))] = alpha_features
        features = {int(key - np.rad2deg(particle_pose[2])): value for key, value in features.items()}
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
            #print(f'Particle {particle.shape}: \n{particle}')
            particle_pose = self.point_to_map(particle[:2].reshape(2,1))
            #print(f'Particle Pose {particle_pose.shape}: \n{particle_pose}')
            particle_orientation = particle[2]
            particle_pose = np.array([particle_pose[0][0], particle_pose[1][0], particle_orientation])
            #plt.scatter(particle_pose[0], particle_pose[1], c='orange')
            features.append(self.get_line_intersection(particle_pose))
            
        return features

    def normalize_errors(self, errors):
        """
        Normalize the errors.

        Args:
            errors (numpy.ndarray): The errors as a 1D array of shape (N,).

        Returns:
            numpy.ndarray: The normalized errors as a 1D array of shape (N,).

            
        """
        max_error = max(errors)
        normalized_errors = [error / max_error for error in errors]
        
        # Step 2: Transform to probability distribution using the inverse of the errors
        probabilities = [1 - error for error in normalized_errors]
        
        # Step 3: Normalize the probabilities to ensure the sum equals 1
        sum_probabilities = sum(probabilities)
        normalized_probabilities = [prob / sum_probabilities for prob in probabilities]
        
        return normalized_probabilities
    


    def motion_model_update(self,cmd_vel_msg):
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

        delta_t = (self.get_clock().now().nanoseconds / 1e9 - self.t0)
        delta_x = (cmd_vel_msg.linear.x * np.cos(self.theta0) * delta_t)
        delta_y = (cmd_vel_msg.linear.x * np.sin(self.theta0) * delta_t)
        delta_theta = cmd_vel_msg.angular.z * delta_t
        

        print(f'Delta X: {delta_x}, Delta Y: {delta_y}, Delta Theta Degrees: {np.rad2deg(delta_theta)}')
        
        if self.absolute_pose is not None:

            self.particles[:,0] +=  delta_x
            self.particles[:,1] +=  delta_y
            self.particles[:,2] +=  delta_theta

            R = Rotation.from_euler('z', delta_theta).as_matrix()
            transformed_points = self.particles[:,:2] - np.array([self.x_mm, self.y_mm]).reshape(1,2)
            self.particles[:,:2] = Pi(R @ PiInv(transformed_points.T)).T + np.array([self.x_mm, self.y_mm]).reshape(1,2)
            # transformed_points = self.particles[:,:2] - np.array(self.absolute_pose[:2]).reshape(1,2)
            # self.particles[:,:2] = Pi(R @ PiInv(transformed_points.T)).T + np.array(self.absolute_pose[:2]).reshape(1,2)

            # Motion Model
            self.x_mm += delta_x
            self.y_mm += delta_y


            # Publish Path For RVIZ Visualization
            motion_model_path_pose = PoseStamped()
            motion_model_path_pose.pose.position.x = self.x_mm
            motion_model_path_pose.pose.position.y = self.y_mm
            motion_model_path_pose.pose.position.z = self.z0
            self.motion_model_path_list.append(motion_model_path_pose)
            self.motion_model_path_msg.poses = self.motion_model_path_list
            self.motion_model_path_msg.header.stamp = self.get_clock().now().to_msg()
            self.motion_model_path_pub.publish(self.motion_model_path_msg)

            self.mm_array = np.hstack((self.mm_array, np.array([self.x_mm, self.y_mm]).reshape(2,1)))
            np.save('mm.npy', self.mm_array)

            print(f'Publishing Motion Model Path')
            self.t0 = self.get_clock().now().nanoseconds / 1e9

        

        # if self.absolute_pose is not None:


            

        #     # transformed_points = self.particles[:,:2] - np.array([self.x_mm, self.y_mm]).reshape(1,2)
        #     # self.particles[:,:2] = Pi(R @ PiInv(transformed_points.T)).T + np.array([self.x_mm, self.y_mm]).reshape(1,2)
        #     # transformed_points = self.particles[:,:2] - np.array(self.absolute_pose[:2]).reshape(1,2)
        #     # self.particles[:,:2] = Pi(R @ PiInv(transformed_points.T)).T + np.array(self.absolute_pose[:2]).reshape(1,2)


        #     # rotated_pts = R @ np.array([self.x_mm, self.y_mm, 1]).reshape(3,1)

        #     # print(f'Rotated Points {rotated_pts.shape}: {rotated_pts}')
        #     # self.x_mm = rotated_pts[0].item()
        #     # self.y_mm = rotated_pts[1].item()
        #     # self. += delta_theta



        

    def odom_callback(self, odom_msg):
        if odom_msg.header.frame_id == "odom" and odom_msg.child_frame_id == "base_footprint":
            # Publish Path For RVIZ Visualization

            print(f'Odom Msg: {odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y}')
            gt_path_pose = PoseStamped()
            gt_path_pose.pose.position.x = odom_msg.pose.pose.position.x + self.base2scan.transform.translation.x
            gt_path_pose.pose.position.y = odom_msg.pose.pose.position.y + self.base2scan.transform.translation.y
            gt_path_pose.pose.position.z = odom_msg.pose.pose.position.z + self.base2scan.transform.translation.z
            self.gt_path_list.append(gt_path_pose)
            self.gt_path_msg.poses = self.gt_path_list
            self.gt_path_msg.header.stamp = self.get_clock().now().to_msg()
            self.gt_path_pub.publish(self.gt_path_msg)

            self.gt_array = np.hstack((self.gt_array, np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y]).reshape(2,1)))
            np.save('gt.npy', self.gt_array)





        
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
            #plt.plot(*np.array(P).T)



def main(args=None):
    rclpy.init(args=args)
    localization_node = ParticleFilter()
    rclpy.spin(localization_node)
    localization_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
