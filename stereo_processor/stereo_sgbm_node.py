import os
import yaml
import struct
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np
from rclpy.qos import qos_profile_sensor_data


class StereoSGBMNode(Node):
    def __init__(self):
        super().__init__('stereo_sgbm_node')

        self.declare_parameter('left_topic', '/left/camera/image/compressed')
        self.declare_parameter('right_topic', '/right/camera/image/compressed')
        self.declare_parameter(
            'calib_file', './src/stereo_processor/config/stereo_calibration.yaml')

        left_topic = self.get_parameter('left_topic').value
        right_topic = self.get_parameter('right_topic').value
        calib_file_path = self.get_parameter('calib_file').value

        self.load_calibration_data(calib_file_path)

        self.image_size = (640, 480)
        self.rectification_initialized = False
        self.Q = None  # The Disparity-to-Depth mapping matrix

        # --- NEW: ROS Publishers & Bridge ---
        self.bridge = CvBridge()
        self.depth_img_pub = self.create_publisher(
            Image, '/stereo/depth_image', 10)
        self.point_cloud_pub = self.create_publisher(
            PointCloud2, '/stereo/point_cloud', 10)

        # Setup the SGBM Engine
        window_size = 5
        min_disp = 0
        num_disp = 16 * 5

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=8 * 3 * window_size**2,
            P2=32 * 3 * window_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        self.left_sub = message_filters.Subscriber(
            self, CompressedImage, left_topic, qos_profile=qos_profile_sensor_data)

        self.right_sub = message_filters.Subscriber(
            self, CompressedImage, right_topic, qos_profile=qos_profile_sensor_data)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub], queue_size=10, slop=0.05
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info("Live SGBM Node Initialized. Publishing to RViz2...")

    def load_calibration_data(self, filepath):
        if not os.path.exists(filepath):
            self.get_logger().fatal(
                f"Calibration file not found at: {filepath}")
            raise FileNotFoundError(f"Missing {filepath}")

        with open(filepath, 'r') as file:
            calib = yaml.safe_load(file)

        try:
            self.K1 = np.array(calib['left_camera']
                               ['intrinsics']).reshape(3, 3)
            self.D1 = np.array(calib['left_camera']['distortion'])
            self.K2 = np.array(calib['right_camera']
                               ['intrinsics']).reshape(3, 3)
            self.D2 = np.array(calib['right_camera']['distortion'])
            self.R = np.array(calib['stereo']['rotation']).reshape(3, 3)
            self.T = np.array(calib['stereo']['translation'])
            self.get_logger().info(
                f"Successfully loaded calibration data from {filepath}")
        except KeyError as e:
            self.get_logger().fatal(f"YAML formatting error. Missing key: {e}")
            raise

    def init_rectification_maps(self):
        # We now extract Q to calculate accurate 3D points
        R1, R2, P1, P2, self.Q, _, _ = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2, self.image_size, self.R, self.T
        )
        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(
            self.K1, self.D1, R1, P1, self.image_size, cv2.CV_16SC2)
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(
            self.K2, self.D2, R2, P2, self.image_size, cv2.CV_16SC2)
        self.rectification_initialized = True
        self.get_logger().info("Rectification maps & Q matrix generated.")

    def sync_callback(self, left_msg, right_msg):
        left_np = np.frombuffer(left_msg.data, np.uint8)
        right_np = np.frombuffer(right_msg.data, np.uint8)

        left_frame = cv2.imdecode(left_np, cv2.IMREAD_COLOR)
        right_frame = cv2.imdecode(right_np, cv2.IMREAD_COLOR)

        if left_frame is None or right_frame is None:
            return

        if not self.rectification_initialized:
            self.init_rectification_maps()

        left_rect = cv2.remap(left_frame, self.map1_l,
                              self.map2_l, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_frame, self.map1_r,
                               self.map2_r, cv2.INTER_LINEAR)

        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        disparity = self.stereo.compute(left_gray, right_gray)

        # --- 1. Publish the 2D Depth Image (Colorized for RViz) ---
        disp_normalized = cv2.normalize(
            disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_map_color = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)

        # Use the timestamp from the left camera message
        header = Header()
        header.stamp = left_msg.header.stamp
        header.frame_id = 'stereo_camera_frame'

        depth_msg = self.bridge.cv2_to_imgmsg(depth_map_color, encoding="bgr8")
        depth_msg.header = header
        self.depth_img_pub.publish(depth_msg)

        # --- 2. Publish the 3D Point Cloud ---
        # Convert SGBM 16-bit format to true float disparity
        disp_float = disparity.astype(np.float32) / 16.0

        # Reproject to 3D using the Q matrix
        points_3D = cv2.reprojectImageTo3D(disp_float, self.Q)

        # Filter out bad points (disparity <= 0 or points too far away)
        mask = (disp_float > 0)
        valid_points = points_3D[mask]
        valid_colors = left_rect[mask]

        # Highly optimized NumPy packing for ROS 2 RGB floats
        r = valid_colors[:, 2].astype(np.uint32)
        g = valid_colors[:, 1].astype(np.uint32)
        b = valid_colors[:, 0].astype(np.uint32)
        rgb = (r << 16) | (g << 8) | b
        rgb_float = rgb.view(np.float32)

        # Combine X, Y, Z, and RGB into a single array
        points_and_colors = np.column_stack((valid_points, rgb_float))

        # Create and publish the PointCloud2 message
        fields = [
            PointField(name='x', offset=0,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12,
                       datatype=PointField.FLOAT32, count=1),
        ]

        pc_msg = pc2.create_cloud(header, fields, points_and_colors)
        self.point_cloud_pub.publish(pc_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StereoSGBMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
