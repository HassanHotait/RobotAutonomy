
from email.mime import image
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

        self.path_list = []

        transform_scanner = self.wait_for_transform('odom', 'base_scan')

        # # Initialize variables
        self.x0 = transform_scanner.transform.translation.x
        self.y0 = transform_scanner.transform.translation.y
        self.z0 = transform_scanner.transform.translation.z

        print(f'Init Pose: {self.x0, self.y0}')

        R = Rotation.from_quat([transform_scanner.transform.rotation.x,transform_scanner.transform.rotation.y,transform_scanner.transform.rotation.z,transform_scanner.transform.rotation.w]).as_matrix()
        self.rx = transform_scanner.transform.rotation.x
        self.ry = transform_scanner.transform.rotation.y
        self.theta0 = np.arctan2(R[1, 0], R[0, 0])
        self.rw = transform_scanner.transform.rotation.w
        self.t0 = self.get_clock().now().nanoseconds / 1e9

        self.init_particles()

        # Subscribe to the '/particle_cloud' topic with the specified QoS profile
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.process_scan, 10)
        map_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, history = QoSHistoryPolicy.KEEP_LAST)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.motion_model_update, 10)
        self.particle_pub = self.create_publisher(ParticleCloud, '/particle_cloud', 10)
        self.pose_pub = self.create_publisher(Odometry, '/amcl_pose', 10)
        self.path_pub = self.create_publisher(Path, '/path', 10)

    def init_particles(self):
        """
        Initialize the particles.
        """
        for i in range(self.n_particles):
            x = self.x0 + np.random.uniform(-0.5, 0.5)
            y = self.y0 + np.random.uniform(-0.5, 0.5)
            theta = self.theta0 + np.random.uniform(-np.pi/6, np.pi/6)
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
        # print(f'PC Shape: {pc.shape}')
        # print(f'Scan Angle Min: {np.rad2deg(scan_msg.angle_min)}')
        # print(f'Scan Angle Max: {np.rad2deg(scan_msg.angle_max)}')
        # print(f'Scan Angle Increment: {np.rad2deg(scan_msg.angle_increment)}')

        # Viz In Map 
        if self.map is not None:
            img = 255 - np.array(self.map.data).reshape(self.map.info.height,self.map.info.width)
            #np.save('map.npy', img)
            robot_gridpose = self.point_to_map(np.array([self.x0, self.y0]).reshape(2,1))

            #print(f'Robot Pose Grid {robot_gridpose.shape}: {robot_gridpose}')
            plt.arrow(robot_gridpose[0][0], robot_gridpose[1][0], 0.5 * np.cos(self.theta0), 0.5 * np.sin(self.theta0), head_width=5, head_length=5, fc='green', ec='green')  # Plot orientation arrow
            plt.imshow(img,cmap ='gray')


            for i,point in enumerate(self.particles):
                particle_grid = self.point_to_map(point[:2].reshape(2,1))
                # print(f'Particle Grid {i} {particle_grid.shape}: {particle_grid}')
                plt.arrow(particle_grid[0][0], particle_grid[1][0], 0.5 * np.cos(point[2]), 0.5 * np.sin(point[2]), head_width=3, head_length=3, fc='red', ec='red')  # Plot orientation arrow
                #print(f'Candidate Particle Plotted')

            features = self.particle_features(scan_msg)
            
            # for i, feature in enumerate(features):
            #     print(f'Particle {i}: {feature}')

            


            # print(f'PC Shape: {pc.shape}')
            pc_grid = self.point_to_map(pc).T


            weights = []
            for feature in features:
                feature_error = []
                for key, value in feature.items():

                    error = np.linalg.norm(np.array(value).reshape(2,1) - np.array([pc_grid[key][0], pc_grid[key][1]]))
                    feature_error.append(error)
                feature_error = np.mean(feature_error)
                weights.append(feature_error)


            probabilities = self.normalize_errors(weights)
            self.particles[:,3] = probabilities

            msg_particle_cloud = ParticleCloud()
            msg_particle_cloud.header.stamp = self.get_clock().now().to_msg()
            msg_particle_cloud.header.frame_id = 'odom'
            msg_particle_cloud.particles = []
            for particle in self.particles:
                p = Particle()
                p.pose.position.x = particle[0]
                p.pose.position.y = particle[1]
                p.pose.position.z = self.z0
                p.pose.orientation.z = particle[2]
                p.weight = particle[3]
                msg_particle_cloud.particles.append(p)
            self.particle_pub.publish(msg_particle_cloud)
            highest_prob_index = np.argmin(probabilities)

            #print(f'Highest Prob Particle Index: {highest_prob_index}')
            # max_likelihood_particle = self.particles[highest_prob_index][:2]
            weighted_likelihood_particle = np.average(self.particles[:,:2], axis=0, weights=self.particles[:,3])
            # #print(f'Max Likelihood Particle: {max_likelihood_particle}')
            print(f'Weighted Likelihood Particle: {weighted_likelihood_particle}')

            path_pose = PoseStamped()
            path_pose.pose.position.x = weighted_likelihood_particle[0]
            path_pose.pose.position.y = weighted_likelihood_particle[1]
            path_pose.pose.position.x = self.z0
            self.path_list.append(path_pose)
            self.path_msg.poses = self.path_list

            self.path_msg.header.stamp = self.get_clock().now().to_msg()
            self.path_pub.publish(self.path_msg)


            # # print(f'Weights: \n{weights}')
            # # print(f'Probabilities: \n{probabilities}')
            # estimated_pose = self.point_to_map(max_likelihood_particle.reshape(2,1))
            # estimated_orientation = self.particles[highest_prob_index][2]
            # plt.arrow(estimated_pose[0][0], estimated_pose[1][0], 0.5 * np.cos(estimated_orientation), 0.5 * np.sin(estimated_orientation), head_width=3, head_length=3, fc='black', ec='black')  # Plot orientation arrow

            # print(f'Highest Probability Pose: {estimated_pose} - Probability: {probabilities[highest_prob_index]} - Weight: {weights[highest_prob_index]}')
            # print(f'len weights: {len(weights)}')
            # print(f'self.point_to_map(self.particles[:,:2].T): {self.point_to_map(self.particles[:,:2].T).shape}')


            weighted_pose = self.point_to_map(weighted_likelihood_particle.reshape(2,1))
            print(f'Weighted Pose Grid {weighted_pose.shape}: {weighted_pose}')
            weighted_orientation = np.average(self.particles[:,2], axis=0,weights=self.particles[:,3])
            print(f'Weighted Orientation {weighted_orientation.shape}: {weighted_orientation}')

            # print(f'Weighted Pose [m]: {np.average(self.particles[:,:2], axis=0, weights=self.particles[:,3])}')
            # print(f'Weighted Pose Grid: {weighted_pose}')
            #plt.arrow(weighted_pose[0], weighted_pose[1], 0.5 * np.cos(weighted_orientation), 0.5 * np.sin(weighted_orientation), head_width=3, head_length=3, fc='yellow', ec='yellow')  # Plot orientation arrow

            # feature_indices = [round(np.rad2deg(ang)) for ang in np.linspace(0, np.pi, self.n_features)][:-1]
            # feature_pc = pc_grid[feature_indices]
            # print(f'Feature Indices: {feature_indices}')

            # print(f'pc features {feature_pc.shape}: \n{feature_pc}')

            # print(f'Features {features.shape}: \n{features}')
            # Create Odometry message
            # odom = Odometry()
            # odom.header.stamp = self.get_clock().now().to_msg()
            # odom.header.frame_id = 'amcl_baselink'
            # odom.child_frame_id = '???'  # Replace with the correct child frame id
            # odom.pose.pose.position.x = float(weighted_likelihood_particle[0])
            # odom.pose.pose.position.y = float(weighted_likelihood_particle[1])
            # odom.pose.pose.position.z = float(self.z0)
            # odom.pose.pose.orientation.z = float(weighted_orientation)
            # odom.twist.twist = self.cmd_vel_msg
            # # Publish the estimated pose
            # self.pose_pub.publish(odom)
                    # Create a transform
            self.static_transform = TransformStamped()
            self.static_transform.header.stamp = self.get_clock().now().to_msg()
            self.static_transform.header.frame_id = 'odom'
            self.static_transform.child_frame_id = 'amcl_basescan'
            self.static_transform.transform.translation.x = float(weighted_likelihood_particle[0])
            self.static_transform.transform.translation.y = float(weighted_likelihood_particle[1])
            self.static_transform.transform.translation.z = float(self.z0)
            self.static_transform.transform.rotation.x = float(self.rx)
            self.static_transform.transform.rotation.y = float(self.ry)
            self.static_transform.transform.rotation.z = float(weighted_orientation)
            self.static_transform.transform.rotation.w = float(self.rw)

            self.tf_broadcaster.sendTransform(self.static_transform)

        #plt.show()

        return pc
    
    def point_to_map(self, pc):
        """
        Convert a point in the map frame to the grid coordinates.

        Args:
            point (numpy.ndarray): The point as a 1D array of shape (2,).

        Returns:
            numpy.ndarray: The grid coordinates as a 1D array of shape (2,).
        """



        #print(f'PC In Conversion to grid {pc.shape}: {pc}')
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
                            alpha_features.append(([x, y]))
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
                            alpha_features.append(([x, y]))
                            plt.scatter(x, y, c='blue')
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
                            alpha_features.append(([x, y]))
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
                            alpha_features.append(([x, y]))
                            #plt.scatter(x, y, c='blue')
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
                                features[round(np.rad2deg(alpha))] = [x, y]
                            else:
                                features[round(np.rad2deg(alpha + np.pi))] = [x, y]
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
                                features[round(np.rad2deg(alpha))] = [x, y]
                            else:
                                features[round(np.rad2deg(alpha + np.pi))] = [x, y]
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

        delta_t = (self.get_clock().now().nanoseconds / 1e9 - self.t0)
        delta_x = (cmd_vel_msg.linear.x * np.cos(self.theta0) * delta_t)
        delta_y = (cmd_vel_msg.linear.x * np.sin(self.theta0) * delta_t)
        delta_theta = cmd_vel_msg.angular.z * delta_t

        self.particles[:,0] +=  delta_x
        self.particles[:,1] +=  delta_y
        self.particles[:,2] +=  delta_theta

        self.t0 = self.get_clock().now().nanoseconds / 1e9

        weighted_pose = np.average(self.particles[:,:2], axis=0, weights=self.particles[:,3])
        self.x0 = weighted_pose[0]
        self.y0 = weighted_pose[1]
        self.theta0 = np.average(self.particles[:,2], weights=self.particles[:,3])
        print(f'--------------------------------- CMD VEL ------------------------------')
        print(f'Delta X: {delta_x}, Delta Y: {delta_y}, Delta Theta: {delta_theta}')

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
            #plt.plot(*np.array(P).T)



def main(args=None):
    rclpy.init(args=args)
    localization_node = ParticleFilter()
    rclpy.spin(localization_node)
    localization_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
