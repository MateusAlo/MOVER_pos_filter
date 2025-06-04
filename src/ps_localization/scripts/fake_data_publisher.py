#!/usr/bin/env python3
"""
This node simulates fake data for testing a particle filter.
It publishes three topics:
  - /gnss (NavSatFix): simulated GNSS data (lat, lon, alt + covariance)
  - /imu/data (Imu): simulated IMU data (orientation, angular velocity, acceleration)
  - /filter/velocity (Vector3Stamped): simulated velocity data (x,y,z components)

To record the bag file, run this node and then in another terminal:
  ros2 bag record -o fake_data_bag /gnss /imu/data /filter/velocity
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from geometry_msgs.msg import Vector3Stamped
import numpy as np
import math

class FakeDataPublisher(Node):
    def __init__(self):
        super().__init__('fake_data_publisher')

        # --- Publishers ---
        self.gnss_pub = self.create_publisher(NavSatFix, '/gnss', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.vel_pub = self.create_publisher(Vector3Stamped, '/filter/velocity', 10)

        # Timer to update at 10 Hz
        self.timer = self.create_timer(0.1, self.timer_callback)

        # --- True state in a local ENU (meters) ---
        self.true_x = 0.0       # East (m)
        self.true_y = 0.0       # North (m)
        self.true_yaw = 0.0     # Heading (rad)

        # Constant motion parameters
        self.linear_velocity = 1.0    # m/s forward in body frame
        self.angular_velocity = 0.1   # rad/s yaw rate

        self.prev_time = self.get_clock().now()

        # --- Fixed reference latitude/longitude for GNSS (degrees) ---
        # We assume a “truth” origin at these coordinates. 
        # Then any local (true_x, true_y) are offsets from this lat0,lon0.
        self.lat0 = 48.0   # degrees
        self.lon0 = 11.0   # degrees
        self.alt0 = 0.0    # meters

        # Precompute cos(lat0) and Earth‐radius
        self.R_earth = 6378137.0  # mean Earth radius (m)
        self.cos_lat0 = math.cos(math.radians(self.lat0))

    def timer_callback(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds * 1e-9
        if dt <= 0.0:
            dt = 0.1

        # --- Update the “true” state in local ENU (meters) ---
        self.true_x += self.linear_velocity * math.cos(self.true_yaw) * dt
        self.true_y += self.linear_velocity * math.sin(self.true_yaw) * dt
        self.true_yaw += self.angular_velocity * dt

        # --- Publish simulated GNSS as NavSatFix on /gnss ---
        gnss_msg = NavSatFix()
        gnss_msg.header.stamp = current_time.to_msg()
        gnss_msg.header.frame_id = "map"

        # Convert local ENU (true_x, true_y) → lat/lon with small‐angle approx
        #   Δlat  ≈ true_y / R_earth
        #   Δlon  ≈ true_x / (R_earth * cos(lat0))
        lat_noise = (self.true_y / self.R_earth) + np.random.normal(0, 0.5 / self.R_earth)
        lon_noise = (self.true_x / (self.R_earth * self.cos_lat0)) + np.random.normal(0, 0.5 / (self.R_earth * self.cos_lat0))
        gnss_msg.latitude  = self.lat0 + math.degrees(lat_noise)  # degrees
        gnss_msg.longitude = self.lon0 + math.degrees(lon_noise)  # degrees
        gnss_msg.altitude  = self.alt0 + np.random.normal(0, 1.0)  # meters

        # Fill status: pretend we always have a valid fix
        gnss_msg.status.status = NavSatStatus.STATUS_FIX
        gnss_msg.status.service = NavSatStatus.SERVICE_GPS

        # Build a 3×3 covariance for (lat,lon,alt). PF only uses ENU variances,
        # but we must fill the 9‐element array here. We'll assume pos cov ~ (0.5 m)^2.
        pos_var = 0.5 ** 2
        cov3 = [pos_var, 0.0,    0.0,
                0.0,    pos_var, 0.0,
                0.0,    0.0,    pos_var]
        gnss_msg.position_covariance = cov3
        gnss_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED

        self.gnss_pub.publish(gnss_msg)

        # --- Publish simulated IMU on /imu/data (unchanged) ---
        imu_msg = Imu()
        imu_msg.header.stamp = current_time.to_msg()
        imu_msg.header.frame_id = "imu_link"

        # True orientation from yaw only (zero roll/pitch)
        imu_msg.orientation.z = math.sin(self.true_yaw / 2.0)
        imu_msg.orientation.w = math.cos(self.true_yaw / 2.0)

        # Angular velocity around Z + small noise
        imu_msg.angular_velocity.x = 0.0 + np.random.normal(0, 0.01)
        imu_msg.angular_velocity.y = 0.0 + np.random.normal(0, 0.01)
        imu_msg.angular_velocity.z = self.angular_velocity + np.random.normal(0, 0.01)

        # Linear acceleration: assume constant speed → zero ideal accel, plus noise & gravity
        imu_msg.linear_acceleration.x = 0.0 + np.random.normal(0, 0.1)
        imu_msg.linear_acceleration.y = 0.0 + np.random.normal(0, 0.1)
        imu_msg.linear_acceleration.z = 9.81 + np.random.normal(0, 0.1)

        self.imu_pub.publish(imu_msg)

        # --- Publish simulated velocity as Vector3Stamped on /filter/velocity ---
        vel_msg = Vector3Stamped()
        vel_msg.header.stamp = current_time.to_msg()
        vel_msg.header.frame_id = "base_link"
        # Provide (vx, vy, vz) in m/s
        vel_msg.vector.x = self.linear_velocity + np.random.normal(0, 0.1)
        vel_msg.vector.y = 0.0
        vel_msg.vector.z = 0.0
        self.vel_pub.publish(vel_msg)

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
        rclpy.shutdown()

if __name__ == '__main__':
    main()
# This code is part of the ps_localization package, which simulates fake data for testing purposes.