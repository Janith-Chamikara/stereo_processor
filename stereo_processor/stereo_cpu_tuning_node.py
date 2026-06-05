import os
import yaml
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from cv_bridge import CvBridge
import message_filters
from rclpy.qos import qos_profile_sensor_data


class StereoCpuTuningNode(Node):
    def __init__(self):
        super().__init__('stereo_cpu_tuning_node')

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
        self.Q = None

        self.bridge = CvBridge()
        self.depth_img_pub = self.create_publisher(
            Image, '/stereo/depth_image', 10)
        self.point_cloud_pub = self.create_publisher(
            PointCloud2, '/stereo/point_cloud', 10)

        # --- CPU SGBM Setup (Better for CPU than standard BM) ---
        self.block_s = 5
        self.num_disp = 16 * 8

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disp,
            blockSize=self.block_s,
            P1=8 * 3 * self.block_s**2,
            P2=32 * 3 * self.block_s**2,
            disp12MaxDiff=1,
            uniquenessRatio=5,
            speckleWindowSize=50,
            speckleRange=2,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        self.left_sub = message_filters.Subscriber(
            self, CompressedImage, left_topic, qos_profile=qos_profile_sensor_data)
        self.right_sub = message_filters.Subscriber(
            self, CompressedImage, right_topic, qos_profile=qos_profile_sensor_data)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info("CPU Tuning Node Active!")
        self.get_logger().info(
            "FOCUS THE OPENCV WINDOW: Use Q/A for Block Size, W/S for Disparities.")

    def load_calibration_data(self, filepath):
        with open(filepath, 'r') as file:
            calib = yaml.safe_load(file)
        self.K1 = np.array(calib['left_camera']['intrinsics']).reshape(3, 3)
        self.D1 = np.array(calib['left_camera']['distortion'])
        self.K2 = np.array(calib['right_camera']['intrinsics']).reshape(3, 3)
        self.D2 = np.array(calib['right_camera']['distortion'])
        self.R = np.array(calib['stereo']['rotation']).reshape(3, 3)
        self.T = np.array(calib['stereo']['translation'])

    def init_rectification_maps(self):
        R1, R2, P1, P2, self.Q, _, _ = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2, self.image_size, self.R, self.T)
        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(
            self.K1, self.D1, R1, P1, self.image_size, cv2.CV_16SC2)
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(
            self.K2, self.D2, R2, P2, self.image_size, cv2.CV_16SC2)
        self.rectification_initialized = True

    def sync_callback(self, left_msg, right_msg):
        left_np = np.frombuffer(left_msg.data, np.uint8)
        right_np = np.frombuffer(right_msg.data, np.uint8)

        left_frame = cv2.imdecode(left_np, cv2.IMREAD_COLOR)
        right_frame = cv2.imdecode(right_np, cv2.IMREAD_COLOR)

        if not self.rectification_initialized:
            self.init_rectification_maps()

        # CPU Remap
        left_rect = cv2.remap(left_frame, self.map1_l,
                              self.map2_l, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_frame, self.map1_r,
                               self.map2_r, cv2.INTER_LINEAR)

        # Grayscale & Compute
        gray_left = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        disparity = self.stereo.compute(gray_left, gray_right)

        # Normalize and Colorize (The Rainbow Effect)
        normalized_disparity = cv2.normalize(
            disparity, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
        depth_map_color = cv2.applyColorMap(
            np.uint8(normalized_disparity * 255), cv2.COLORMAP_JET)

        # Publish 2D Depth
        header = Header()
        header.stamp = left_msg.header.stamp
        header.frame_id = 'stereo_camera_frame'
        depth_msg = self.bridge.cv2_to_imgmsg(depth_map_color, encoding="bgr8")
        depth_msg.header = header
        self.depth_img_pub.publish(depth_msg)

        # Publish True 3D Point Cloud (Rainbow Colored)
        disp_float = disparity.astype(np.float32) / 16.0
        points_3D = cv2.reprojectImageTo3D(disp_float, self.Q)
        mask = (disp_float > 0)

        valid_points = points_3D[mask]
        valid_colors = depth_map_color[mask]  # Rainbow trick applied here!

        r = valid_colors[:, 2].astype(np.uint32)
        g = valid_colors[:, 1].astype(np.uint32)
        b = valid_colors[:, 0].astype(np.uint32)
        rgb = (r << 16) | (g << 8) | b
        rgb_float = rgb.view(np.float32)

        points_and_colors = np.column_stack((valid_points, rgb_float))
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

        # Live Tuning UI
        cv2.imshow("CPU Live Tuning (Depth)", depth_map_color)
        k = cv2.waitKey(1)
        if k == ord('q'):
            self.block_s += 2
            self.stereo.setBlockSize(self.block_s)
            self.stereo.setP1(8 * 3 * self.block_s**2)
            self.stereo.setP2(32 * 3 * self.block_s**2)
            self.get_logger().info(f"Block Size Increased: {self.block_s}")
        elif k == ord('a'):
            self.block_s = max(self.block_s - 2, 5)
            self.stereo.setBlockSize(self.block_s)
            self.stereo.setP1(8 * 3 * self.block_s**2)
            self.stereo.setP2(32 * 3 * self.block_s**2)
            self.get_logger().info(f"Block Size Decreased: {self.block_s}")
        elif k == ord('w'):
            self.num_disp += 16
            self.stereo.setNumDisparities(self.num_disp)
            self.get_logger().info(f"Disparities Increased: {self.num_disp}")
        elif k == ord('s'):
            self.num_disp = max(16, self.num_disp - 16)
            self.stereo.setNumDisparities(self.num_disp)
            self.get_logger().info(f"Disparities Decreased: {self.num_disp}")


def main(args=None):
    rclpy.init(args=args)
    node = StereoCpuTuningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
