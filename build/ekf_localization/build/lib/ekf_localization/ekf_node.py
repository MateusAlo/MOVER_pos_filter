import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import numpy as np

def main(args=None):
    class EKFNode(Node):
        def __init__(self):
            super().__init__('ekf_localization_node')

            # Subscribe to GNSS and IMU topics
            self.gnss_sub = self.create_subscription(
                PoseWithCovarianceStamped, '/gps/filtered', self.gnss_callback, 10)
            self.imu_sub = self.create_subscription(
                Imu, '/imu/data', self.imu_callback, 10)

            # Publisher for EKF localization result
            self.ekf_pub = self.create_publisher(Odometry, '/ekf/odom', 10)

            # EKF state variables
            self.state = np.zeros(6)  # [x, y, theta, vx, vy, omega]
            self.covariance = np.eye(6) * 0.1
            self.last_time = self.get_clock().now()

            # EKF parameters
            self.process_noise = np.eye(6) * 0.01
            self.measurement_noise_gnss = np.eye(2) * 0.5  # GPS x, y noise
            self.measurement_noise_imu = np.eye(3) * 0.1  # IMU roll, pitch, yaw noise

        def gnss_callback(self, msg):
            z = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
            R = self.measurement_noise_gnss

            # Perform measurement update
            self.ekf_update(z, R)

        def imu_callback(self, msg):
            dt = (self.get_clock().now() - self.last_time).nanoseconds / 1e9
            self.last_time = self.get_clock().now()

            # Use IMU angular velocity for prediction
            omega = msg.angular_velocity.z
            self.ekf_predict(dt, omega)

        def ekf_predict(self, dt, omega):
            # State transition model
            F = np.eye(6)
            F[0, 3] = dt  # x += vx * dt
            F[1, 4] = dt  # y += vy * dt
            F[2, 5] = dt  # theta += omega * dt

            # Control input model
            u = np.array([0, 0, 0, 0, 0, omega])

            # Predict state
            self.state = F @ self.state + u * dt

            # Predict covariance
            self.covariance = F @ self.covariance @ F.T + self.process_noise

        def ekf_update(self, z, R):
            # Measurement model
            H = np.zeros((2, 6))
            H[0, 0] = 1  # Measurement directly observes x
            H[1, 1] = 1  # Measurement directly observes y

            # Measurement prediction
            z_pred = H @ self.state

            # Innovation
            y = z - z_pred

            # Innovation covariance
            S = H @ self.covariance @ H.T + R

            # Kalman gain
            K = self.covariance @ H.T @ np.linalg.inv(S)

            # Update state
            self.state += K @ y

            # Update covariance
            self.covariance = (np.eye(6) - K @ H) @ self.covariance

            # Publish the updated state
            self.publish_state()

        def publish_state(self):
            odom_msg = Odometry()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = "map"
            odom_msg.pose.pose.position.x = self.state[0]
            odom_msg.pose.pose.position.y = self.state[1]
            odom_msg.pose.pose.orientation.z = np.sin(self.state[2] / 2)
            odom_msg.pose.pose.orientation.w = np.cos(self.state[2] / 2)
            self.ekf_pub.publish(odom_msg)


    rclpy.init(args=args)
    node = EKFNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
