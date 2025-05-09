#!/usr/bin/env python3
"""
This node simulates fake data for testing a particle filter.
It publishes three topics:
  - /gps/filtered (PoseWithCovarianceStamped): simulated GNSS data (true state + noise)
  - /imu/data (Imu): simulated IMU data (orientation, angular velocity, acceleration)
  - /filter/velocity (TwistStamped): simulated velocity data (linear and angular velocities)
  
To record the bag file, run this node and then in another terminal:
  ros2 bag record -o fake_data_bag /gps/filtered /imu/data /filter/velocity
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
from sensor_msgs.msg import Imu
import numpy as np
import math


class FakeDataPublisher(Node):
    def __init__(self):
        super().__init__('fake_data_publisher')

        # Publishers for each topic
        self.gnss_pub = self.create_publisher(PoseWithCovarianceStamped, '/gps/filtered', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.velocity_pub = self.create_publisher(TwistStamped, '/filter/velocity', 10)

        # Timer to update at 10 Hz (adjust as needed)
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Initialize the "true" state (this is not published as a separate topic but used to generate sensor data)
        self.true_x = 0.0       # true x position (meters)
        self.true_y = 0.0       # true y position (meters)
        self.true_yaw = 0.0     # true orientation (radians)

        # Define constant motion parameters
        self.linear_velocity = 1.0    # meters per second
        self.angular_velocity = 0.1   # radians per second

        self.prev_time = self.get_clock().now()

    def timer_callback(self):

        self.get_logger().info('Publishing fake data...')

        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds * 1e-9
        if dt == 0:
            dt = 0.1

        # --- Update the ground truth state ---
        self.true_x += self.linear_velocity * math.cos(self.true_yaw) * dt
        self.true_y += self.linear_velocity * math.sin(self.true_yaw) * dt
        self.true_yaw += self.angular_velocity * dt

        # --- Publish simulated GNSS data on /gps/filtered ---
        gnss_msg = PoseWithCovarianceStamped()
        gnss_msg.header.stamp = current_time.to_msg()
        gnss_msg.header.frame_id = "map"
        # Add noise to the true position (simulate GNSS error)
        noise_std_pos = 0.5  # standard deviation in meters
        noisy_x = self.true_x + np.random.normal(0, noise_std_pos)
        noisy_y = self.true_y + np.random.normal(0, noise_std_pos)
        gnss_msg.pose.pose.position.x = noisy_x
        gnss_msg.pose.pose.position.y = noisy_y
        gnss_msg.pose.pose.position.z = 0.0

        # Add noise to the orientation (simulate heading error)
        noise_std_yaw = 0.05  # radians
        noisy_yaw = self.true_yaw + np.random.normal(0, noise_std_yaw)
        # Convert yaw to quaternion (assuming roll=pitch=0)
        qz = math.sin(noisy_yaw / 2.0)
        qw = math.cos(noisy_yaw / 2.0)
        gnss_msg.pose.pose.orientation.z = qz
        gnss_msg.pose.pose.orientation.w = qw

        # Set an example covariance (modify as needed)
        cov = noise_std_pos ** 2
        gnss_msg.pose.covariance = [
            float(cov), 0.0,       0.0,      0.0,      0.0,      0.0,
            0.0,        float(cov), 0.0,      0.0,      0.0,      0.0,
            0.0,        0.0,       99999.0,   0.0,      0.0,      0.0,
            0.0,        0.0,        0.0,     99999.0,   0.0,      0.0,
            0.0,        0.0,        0.0,      0.0,     99999.0,   0.0,
            0.0,        0.0,        0.0,      0.0,      0.0,     99999.0,
]


        self.gnss_pub.publish(gnss_msg)

        # --- Publish simulated IMU data on /imu/data ---
        imu_msg = Imu()
        imu_msg.header.stamp = current_time.to_msg()
        imu_msg.header.frame_id = "imu_link"

        # For simplicity, we use the true yaw (without integrating a full orientation) and assume zero roll and pitch.
        imu_msg.orientation.z = math.sin(self.true_yaw / 2.0)
        imu_msg.orientation.w = math.cos(self.true_yaw / 2.0)
        # (Optional: you can add noise to the orientation too if desired)

        # Simulate angular velocity with small noise
        imu_msg.angular_velocity.z = self.angular_velocity + np.random.normal(0, 0.01)
        imu_msg.angular_velocity.x = 0.0 + np.random.normal(0, 0.01)
        imu_msg.angular_velocity.y = 0.0 + np.random.normal(0, 0.01)

        # Simulate linear acceleration; since our speed is constant the ideal acceleration is zero,
        # but you might include a bit of noise and gravity (if in the sensor frame).
        imu_msg.linear_acceleration.x = 0.0 + np.random.normal(0, 0.1)
        imu_msg.linear_acceleration.y = 0.0 + np.random.normal(0, 0.1)
        imu_msg.linear_acceleration.z = 9.81 + np.random.normal(0, 0.1)  # gravity

        self.imu_pub.publish(imu_msg)

        # --- Publish simulated velocity data on /filter/velocity ---
        twist_msg = TwistStamped()
        twist_msg.header.stamp = current_time.to_msg()
        twist_msg.header.frame_id = "base_link"
        # Add a little noise to the linear and angular velocities
        twist_msg.twist.linear.x = self.linear_velocity + np.random.normal(0, 0.1)
        twist_msg.twist.linear.y = 0.0
        twist_msg.twist.linear.z = 0.0
        twist_msg.twist.angular.z = self.angular_velocity + np.random.normal(0, 0.01)
        twist_msg.twist.angular.x = 0.0
        twist_msg.twist.angular.y = 0.0

        self.velocity_pub.publish(twist_msg)

        # Update previous time for the next iteration
        self.prev_time = current_time


def main(args=None):
    rclpy.init(args=args)
    node = FakeDataPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt detected, shutting down...')
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception as e:
            # This exception occurs if shutdown was already called
            node.get_logger().info('Shutdown already called: {}'.format(e))


if __name__ == '__main__':
    main()
