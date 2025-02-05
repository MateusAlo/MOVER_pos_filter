import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import numpy as np

class ParticleFilterNode(Node):
    def __init__(self):
        super().__init__('particle_filter_localization_node') 

        # Subscribe to GNSS and IMU topics
        self.gnss_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/gps/filtered', self.gnss_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)

        # Publisher for Particle Filter localization result
        self.pf_pub = self.create_publisher(Odometry, '/pf/odom', 10)

        # Particle Filter state variables
        self.num_particles = 100
        self.particles = np.zeros((self.num_particles, 3))  # [x, y, theta]
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.last_time = self.get_clock().now()

        # Particle Filter parameters
        self.process_noise = np.array([0.1, 0.1, 0.01])
        self.measurement_noise_gnss = np.array([0.5, 0.5])
        self.measurement_noise_imu = 0.1

    def gnss_callback(self, msg):
        z = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        R = self.measurement_noise_gnss
    
        # Perform measurement update
        self.pf_update(z, R)

    def imu_callback(self, msg):
        dt = (self.get_clock().now() - self.last_time).nanoseconds / 1e9
        self.last_time = self.get_clock().now()

        # Use IMU angular velocity for prediction
        omega = msg.angular_velocity.z
        self.pf_predict(dt, omega)

    def pf_predict(self, dt, omega):
        # Add process noise
        noise = np.random.normal(0, self.process_noise, (self.num_particles, 3))
        self.particles[:, 2] += omega * dt + noise[:, 2]
        self.particles[:, 0] += np.cos(self.particles[:, 2]) * dt + noise[:, 0]
        self.particles[:, 1] += np.sin(self.particles[:, 2]) * dt + noise[:, 1]

    def pf_update(self, z, R):
        # Compute weights based on GNSS measurement
        for i in range(self.num_particles):
            dx = self.particles[i, 0] - z[0]
            dy = self.particles[i, 1] - z[1]
            self.weights[i] = np.exp(-0.5 * (dx**2 / R[0] + dy**2 / R[1]))

        # Normalize weights
        self.weights += 1.e-300  # avoid division by zero
        self.weights /= np.sum(self.weights)

        # Resample particles
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

        # Publish the estimated state
        self.publish_state()

    def publish_state(self):
        # Estimate state as the mean of the particles
        mean_state = np.average(self.particles, axis=0, weights=self.weights)
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "map"
        odom_msg.pose.pose.position.x = mean_state[0]
        odom_msg.pose.pose.position.y = mean_state[1]
        odom_msg.pose.pose.orientation.z = np.sin(mean_state[2] / 2)
        odom_msg.pose.pose.orientation.w = np.cos(mean_state[2] / 2)
        self.pf_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilterNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()