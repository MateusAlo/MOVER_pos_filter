import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from geometry_msgs.msg import Vector3Stamped, PoseWithCovarianceStamped, PoseStamped
import numpy as np
import math

class ParticleFilterNode(Node):
    def __init__(self):
        super().__init__('particle_filter_node')

        # Subscriptions
        self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.create_subscription(Vector3Stamped, '/filter/velocity', self.velocity_callback, 10)
        self.create_subscription(NavSatFix, '/gnss', self.gnss_callback, 10)

        # Publisher for pose with covariance
        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/pf/pose', 10)
        self.raw_pub = self.create_publisher(PoseStamped, '/gnss_local', 10)
        # Parameters
        self.declare_parameter('num_particles', 100)
        self.declare_parameter('imu_covariance', [0.01] * 6)
        self.declare_parameter('default_gnss_variances_enu', [0.25, 0.25, 4.0])
        self.declare_parameter('gnss_lever_arm_body', [0.0, 0.0, 0.0])

        self.num_particles = self.get_parameter('num_particles').value
        self.imu_cov = np.array(self.get_parameter('imu_covariance').value)
        self.gnss_var = np.array(self.get_parameter('default_gnss_variances_enu').value)
        self.lever_arm = np.array(self.get_parameter('gnss_lever_arm_body').value)

        # Particle Filter state
        self.particles = np.zeros((self.num_particles, 6))  # [x, y, z, roll, pitch, yaw]
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.velocity = 0.0
        self.last_time = self.get_clock().now()

        # GNSS reference and precomputed values
        self.lla_ref = None            # (lat0_rad, lon0_rad, alt0_m)
        self.ref_ecef = None           # (x0, y0, z0) in meters
        self.sin_lat0 = None
        self.cos_lat0 = None
        self.sin_lon0 = None
        self.cos_lon0 = None

    def gnss_callback(self, msg):
        # 1) Check GNSS fix status
        if msg.status.status < NavSatStatus.STATUS_FIX:
            return

        # 2) Convert latitude/longitude to radians, altitude in meters
        lat_rad = math.radians(msg.latitude)
        lon_rad = math.radians(msg.longitude)
        alt_m = msg.altitude

        # 3) On first valid fix, store reference and precompute ECEF + sines/cosines
        if self.lla_ref is None:
            self.lla_ref = (lat_rad, lon_rad, alt_m)
            self.ref_ecef = self._lla_to_ecef(lat_rad, lon_rad, alt_m)
            self.sin_lat0 = math.sin(lat_rad)
            self.cos_lat0 = math.cos(lat_rad)
            self.sin_lon0 = math.sin(lon_rad)
            self.cos_lon0 = math.cos(lon_rad)
            return

        # 4) Convert current LLA → ECEF (meters)
        x, y, z = self._lla_to_ecef(lat_rad, lon_rad, alt_m)

        # 5) Compute ENU = R * (ECEF_current - ECEF_ref)
        dx = x - self.ref_ecef[0]
        dy = y - self.ref_ecef[1]
        dz = z - self.ref_ecef[2]

        east  = -self.sin_lon0 * dx + self.cos_lon0 * dy
        north = (
            -self.sin_lat0 * self.cos_lon0 * dx
            - self.sin_lat0 * self.sin_lon0 * dy
            + self.cos_lat0 * dz
        )
        up    = (
            self.cos_lat0 * self.cos_lon0 * dx
            + self.cos_lat0 * self.sin_lon0 * dy
            + self.sin_lat0 * dz
        )
        z_vec = np.array([east, north, up])


        # Publishing the local gnss position as a PoseStamped message
        raw_msg = PoseStamped()
        raw_msg.header.stamp = self.get_clock().now().to_msg()
        raw_msg.header.frame_id = "map"
        raw_msg.pose.position.x = east
        raw_msg.pose.position.y = north
        raw_msg.pose.position.z = up
        # We don’t care about orientation here, so set identity:
        raw_msg.pose.orientation.x = 0.0
        raw_msg.pose.orientation.y = 0.0
        raw_msg.pose.orientation.z = 0.0
        raw_msg.pose.orientation.w = 1.0
        self.raw_pub.publish(raw_msg)

        # 6) Extract GNSS covariance or use default
        if msg.position_covariance_type != NavSatFix.COVARIANCE_TYPE_UNKNOWN:
            R = np.array([
                msg.position_covariance[0],
                msg.position_covariance[4],
                msg.position_covariance[8]
            ])
        else:
            R = self.gnss_var.copy()

        R[R <= 1e-9] = 1e-9
        self.pf_update(z_vec, R)

    def imu_callback(self, msg):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        ang_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        self.pf_predict(dt, ang_vel, self.velocity)

    def velocity_callback(self, msg):
        vx, vy, vz = msg.vector.x, msg.vector.y, msg.vector.z
        self.velocity = math.sqrt(vx**2 + vy**2 + vz**2)

    def pf_predict(self, dt, ang_vel, v):
        noise = np.random.normal(0, self.imu_cov, (self.num_particles, 6))

        # Update particle positions (ENU) and orientations (roll, pitch, yaw)
        self.particles[:, 0] += np.cos(self.particles[:, 4]) * np.cos(self.particles[:, 5]) * v * dt + noise[:, 0]
        self.particles[:, 1] += np.cos(self.particles[:, 4]) * np.sin(self.particles[:, 5]) * v * dt + noise[:, 1]
        self.particles[:, 2] += np.sin(self.particles[:, 4]) * v * dt + noise[:, 2]

        self.particles[:, 3] += ang_vel[0] * dt + noise[:, 3]
        self.particles[:, 4] += ang_vel[1] * dt + noise[:, 4]
        self.particles[:, 5] += ang_vel[2] * dt + noise[:, 5]

        #(maybe put publish state here will increase the freq of the pose publishing, but i dont know if we can integrate
        #that better taking gnss covariance into account)

    def pf_update(self, z, R):
        # 1) Update weights based on GNSS measurement likelihood
        for i in range(self.num_particles):
            dx = self.particles[i, 0] - z[0]
            dy = self.particles[i, 1] - z[1]
            dz = self.particles[i, 2] - z[2]
            self.weights[i] = math.exp(-0.5 * (dx**2 / R[0] + dy**2 / R[1] + dz**2 / R[2]))

        # 2) Normalize
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)

        # 3) Resample
        idx = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[idx]
        self.weights.fill(1.0 / self.num_particles)

        # 4) Publish estimated pose with covariance
        self.publish_state()

    def publish_state(self):
        mean_state = np.average(self.particles, axis=0, weights=self.weights)

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        # Fill in pose:
        msg.pose.pose.position.x = mean_state[0]
        msg.pose.pose.position.y = mean_state[1]
        msg.pose.pose.position.z = mean_state[2]

        q = self.euler_to_quaternion(mean_state[3], mean_state[4], mean_state[5])
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        # Compute 6×6 covariance of the particles around the mean
        # (flattened in row-major order for the message)
        cov = np.zeros((6, 6))
        for i in range(self.num_particles):
            dx = self.particles[i] - mean_state
            cov += self.weights[i] * np.outer(dx, dx)
        # Flatten row-major:
        msg.pose.covariance = list(cov.flatten())

        # Log so you know it’s publishing
        self.get_logger().info(
            f"Publishing PF→pose: x={mean_state[0]:.2f}, y={mean_state[1]:.2f}, z={mean_state[2]:.2f}"
        )

        self.pub.publish(msg)

    def euler_to_quaternion(self, roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy
        return (qx, qy, qz, qw)

    def _lla_to_ecef(self, lat_rad, lon_rad, h):
        # WGS84 ellipsoid constants
        a = 6378137.0                # semi-major axis (m)
        e2 = 6.69437999014e-3        # first eccentricity squared

        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        sin_lon = math.sin(lon_rad)
        cos_lon = math.cos(lon_rad)

        N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
        x = (N + h) * cos_lat * cos_lon
        y = (N + h) * cos_lat * sin_lon
        z = ((1 - e2) * N + h) * sin_lat
        return (x, y, z)

def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilterNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
