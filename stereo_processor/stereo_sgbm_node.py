import os
import yaml
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
from rcl_interfaces.msg import ParameterDescriptor, IntegerRange, SetParametersResult

class StereoSGBMNode(Node):
    def __init__(self):
        super().__init__('stereo_sgbm_node')

        # --- Static Parameters ---
        self.declare_parameter('left_topic', '/left/camera/image/compressed')
        self.declare_parameter('right_topic', '/right/camera/image/compressed')
        self.declare_parameter('calib_file', './src/stereo_processor/config/stereo_calibration.yaml')

        left_topic = self.get_parameter('left_topic').value
        right_topic = self.get_parameter('right_topic').value
        calib_file_path = self.get_parameter('calib_file').value

        self.load_calibration_data(calib_file_path)

        self.image_size = None
        self.rectification_initialized = False
        self.Q = None

        self.bridge = CvBridge()
        self.depth_img_pub = self.create_publisher(Image, '/stereo/depth_image', 10)
        self.point_cloud_pub = self.create_publisher(PointCloud2, '/stereo/point_cloud', 10)

        # --- Dynamic GUI Parameters (Exposed to rqt) ---
        self.declare_parameter('window_size', 7,
            ParameterDescriptor(integer_range=[IntegerRange(from_value=5, to_value=31, step=2)]))
        self.declare_parameter('num_disparities', 64,
            ParameterDescriptor(integer_range=[IntegerRange(from_value=16, to_value=160, step=16)]))
        self.declare_parameter('uniqueness_ratio', 12,
            ParameterDescriptor(integer_range=[IntegerRange(from_value=1, to_value=25, step=1)]))
        self.declare_parameter('speckle_window_size', 100,
            ParameterDescriptor(integer_range=[IntegerRange(from_value=0, to_value=300, step=5)]))
        self.declare_parameter('speckle_range', 2,
            ParameterDescriptor(integer_range=[IntegerRange(from_value=1, to_value=64, step=1)]))
        self.declare_parameter('disp12_max_diff', 2,
            ParameterDescriptor(integer_range=[IntegerRange(from_value=-1, to_value=25, step=1)]))
        self.declare_parameter('display_blur_size', 5,
            ParameterDescriptor(integer_range=[IntegerRange(from_value=1, to_value=15, step=2)]))
        self.declare_parameter('median_blur_size', 5,
            ParameterDescriptor(integer_range=[IntegerRange(from_value=1, to_value=5, step=2)]))
        self.declare_parameter('use_clahe', True)
        self.declare_parameter('use_wls_filter', True)
        self.declare_parameter('wls_lambda', 8000,
            ParameterDescriptor(integer_range=[IntegerRange(from_value=0, to_value=20000, step=500)]))
        self.declare_parameter('wls_sigma_x100', 150,
            ParameterDescriptor(integer_range=[IntegerRange(from_value=50, to_value=300, step=10)]))
        self.declare_parameter('min_texture', 8,
            ParameterDescriptor(integer_range=[IntegerRange(from_value=0, to_value=50, step=1)]))
        self.declare_parameter('temporal_alpha_x100', 25,
            ParameterDescriptor(integer_range=[IntegerRange(from_value=0, to_value=100, step=5)]))
        self.declare_parameter('display_min_disparity', 0,
            ParameterDescriptor(integer_range=[IntegerRange(from_value=0, to_value=160, step=1)]))
        self.declare_parameter('display_max_disparity', 64,
            ParameterDescriptor(integer_range=[IntegerRange(from_value=1, to_value=160, step=1)]))

        self.frame_count = 0
        self.previous_disp_float = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Register callback to catch rqt slider changes
        self.add_on_set_parameters_callback(self.parameters_callback)

        # Initialize SGBM with starting parameters
        self.update_sgbm_matcher()

        self.left_sub = message_filters.Subscriber(self, CompressedImage, left_topic, qos_profile=qos_profile_sensor_data)
        self.right_sub = message_filters.Subscriber(self, CompressedImage, right_topic, qos_profile=qos_profile_sensor_data)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.left_sub, self.right_sub], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info("StereoSGBMNode ready. Open 'rqt' to tune parameters live.")

    def parameters_callback(self, params):
        """Catches live updates from rqt and rebuilds the SGBM matcher."""
        matcher_params = self.current_matcher_params()
        for param in params:
            self.get_logger().info(f"Live Update: {param.name} -> {param.value}")
            if param.name in matcher_params:
                matcher_params[param.name] = param.value
        
        self.update_sgbm_matcher(matcher_params)
        return SetParametersResult(successful=True)

    def current_matcher_params(self):
        return {
            'window_size': self.get_parameter('window_size').value,
            'num_disparities': self.get_parameter('num_disparities').value,
            'disp12_max_diff': self.get_parameter('disp12_max_diff').value,
            'uniqueness_ratio': self.get_parameter('uniqueness_ratio').value,
            'speckle_window_size': self.get_parameter('speckle_window_size').value,
            'speckle_range': self.get_parameter('speckle_range').value,
        }

    def update_sgbm_matcher(self, matcher_params=None):
        """Pulls current parameters and rebuilds the OpenCV object."""
        if matcher_params is None:
            matcher_params = self.current_matcher_params()

        w_size = matcher_params['window_size']
        num_disp = matcher_params['num_disparities']
        
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=w_size,
            P1=8 * 3 * w_size**2,
            P2=32 * 3 * w_size**2,
            disp12MaxDiff=matcher_params['disp12_max_diff'],
            uniquenessRatio=matcher_params['uniqueness_ratio'],
            speckleWindowSize=matcher_params['speckle_window_size'],
            speckleRange=matcher_params['speckle_range'],
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        self.right_matcher = None
        self.wls_filter = None
        has_wls = (
            hasattr(cv2, 'ximgproc') and
            hasattr(cv2.ximgproc, 'createRightMatcher') and
            hasattr(cv2.ximgproc, 'createDisparityWLSFilter')
        )
        if self.get_parameter('use_wls_filter').value and has_wls:
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereo)
            self.wls_filter.setLambda(self.get_parameter('wls_lambda').value)
            self.wls_filter.setSigmaColor(self.get_parameter('wls_sigma_x100').value / 100.0)

    def load_calibration_data(self, filepath):
        if not os.path.exists(filepath):
            self.get_logger().fatal(f"Calibration file not found at: {filepath}")
            raise FileNotFoundError(f"Missing {filepath}")

        with open(filepath, 'r') as file:
            calib = yaml.safe_load(file)
            self.K1 = np.array(calib['left_camera']['intrinsics']).reshape(3, 3)
            self.D1 = np.array(calib['left_camera']['distortion'])
            self.K2 = np.array(calib['right_camera']['intrinsics']).reshape(3, 3)
            self.D2 = np.array(calib['right_camera']['distortion'])
            self.R = np.array(calib['stereo']['rotation']).reshape(3, 3)
            self.T = np.array(calib['stereo']['translation'])
            image_size = calib.get('image_size', {})
            self.calibration_size = (
                int(image_size.get('width', 640)),
                int(image_size.get('height', 480)),
            )

    def scaled_intrinsics(self, image_size):
        calib_width, calib_height = self.calibration_size
        image_width, image_height = image_size
        scale_x = image_width / calib_width
        scale_y = image_height / calib_height

        K1 = self.K1.copy()
        K2 = self.K2.copy()
        K1[0, 0] *= scale_x
        K1[0, 2] *= scale_x
        K1[1, 1] *= scale_y
        K1[1, 2] *= scale_y
        K2[0, 0] *= scale_x
        K2[0, 2] *= scale_x
        K2[1, 1] *= scale_y
        K2[1, 2] *= scale_y
        return K1, K2

    def init_rectification_maps(self, image_size):
        self.image_size = image_size
        K1, K2 = self.scaled_intrinsics(image_size)
        R1, R2, P1, P2, self.Q, _, _ = cv2.stereoRectify(
            K1, self.D1, K2, self.D2, self.image_size, self.R, self.T)
        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(K1, self.D1, R1, P1, self.image_size, cv2.CV_16SC2)
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(K2, self.D2, R2, P2, self.image_size, cv2.CV_16SC2)
        self.rectification_initialized = True
        self.get_logger().info(
            f"Rectification initialized for runtime image size {image_size}; "
            f"calibration size {self.calibration_size}"
        )

    def compute_filtered_disparity(self, left_gray, right_gray, left_rect):
        disparity = self.stereo.compute(left_gray, right_gray)

        if self.wls_filter is not None and self.right_matcher is not None:
            right_disparity = self.right_matcher.compute(right_gray, left_gray)
            disparity = self.wls_filter.filter(disparity, left_rect, None, right_disparity)

        disp_float = disparity.astype(np.float32) / 16.0
        valid_mask = np.isfinite(disp_float) & (disp_float > 0)

        min_texture = self.get_parameter('min_texture').value
        if min_texture > 0:
            grad_x = cv2.Sobel(left_gray, cv2.CV_16S, 1, 0, ksize=3)
            grad_y = cv2.Sobel(left_gray, cv2.CV_16S, 0, 1, ksize=3)
            texture = cv2.addWeighted(
                cv2.convertScaleAbs(grad_x),
                0.5,
                cv2.convertScaleAbs(grad_y),
                0.5,
                0,
            )
            valid_mask &= texture > min_texture

        median_size = self.get_parameter('median_blur_size').value
        if median_size > 1:
            filtered = cv2.medianBlur(disp_float, median_size)
            disp_float[valid_mask] = filtered[valid_mask]

        disp_float[~valid_mask] = 0.0

        alpha = self.get_parameter('temporal_alpha_x100').value / 100.0
        if alpha > 0.0 and self.previous_disp_float is not None and self.previous_disp_float.shape == disp_float.shape:
            previous_valid = self.previous_disp_float > 0
            stable_mask = valid_mask & previous_valid
            disp_float[stable_mask] = (
                alpha * disp_float[stable_mask] +
                (1.0 - alpha) * self.previous_disp_float[stable_mask]
            )

        self.previous_disp_float = disp_float.copy()

        return disp_float

    def colorize_disparity(self, disp_float):
        valid_mask = np.isfinite(disp_float) & (disp_float > 0)
        depth_color = np.zeros((*disp_float.shape, 3), dtype=np.uint8)
        if not np.any(valid_mask):
            return depth_color, valid_mask

        min_disp = float(self.get_parameter('display_min_disparity').value)
        max_disp = float(self.get_parameter('display_max_disparity').value)
        if max_disp <= min_disp:
            max_disp = min_disp + 1.0

        disp_norm = np.zeros_like(disp_float, dtype=np.float32)
        disp_norm[valid_mask] = np.clip(
            (disp_float[valid_mask] - min_disp) / (max_disp - min_disp),
            0.0,
            1.0,
        )

        blur_size = self.get_parameter('display_blur_size').value
        if blur_size > 1:
            disp_norm = cv2.GaussianBlur(disp_norm, (blur_size, blur_size), 0)
            disp_norm[~valid_mask] = 0.0

        depth_color = cv2.applyColorMap(np.uint8(disp_norm * 255), cv2.COLORMAP_JET)
        depth_color[~valid_mask] = 0
        return depth_color, valid_mask

    def sync_callback(self, left_msg, right_msg):
        left_frame = cv2.imdecode(np.frombuffer(left_msg.data, np.uint8), cv2.IMREAD_COLOR)
        right_frame = cv2.imdecode(np.frombuffer(right_msg.data, np.uint8), cv2.IMREAD_COLOR)

        if left_frame is None or right_frame is None: return

        image_size = (left_frame.shape[1], left_frame.shape[0])
        if not self.rectification_initialized or image_size != self.image_size:
            self.init_rectification_maps(image_size)

        left_rect = cv2.remap(left_frame, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_frame, self.map1_r, self.map2_r, cv2.INTER_LINEAR)

        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        if self.get_parameter('use_clahe').value:
            left_gray = self.clahe.apply(left_gray)
            right_gray = self.clahe.apply(right_gray)

        disp_float = self.compute_filtered_disparity(left_gray, right_gray, left_rect)

        # --- Depth Map Publishing ---
        depth_color, valid_mask = self.colorize_disparity(disp_float)

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            valid_ratio = 100.0 * np.count_nonzero(valid_mask) / valid_mask.size
            self.get_logger().info(f"Valid disparity pixels: {valid_ratio:.1f}%")

        header = Header()
        header.stamp = left_msg.header.stamp
        header.frame_id = 'stereo_camera_frame'

        depth_msg = self.bridge.cv2_to_imgmsg(depth_color, encoding="bgr8")
        depth_msg.header = header
        self.depth_img_pub.publish(depth_msg)

        # --- Point Cloud Publishing ---
        points_3D = cv2.reprojectImageTo3D(disp_float, self.Q)
        valid_points = points_3D[valid_mask]
        valid_colors = left_rect[valid_mask]

        r, g, b = valid_colors[:, 2].astype(np.uint32), valid_colors[:, 1].astype(np.uint32), valid_colors[:, 0].astype(np.uint32)
        rgb_float = ((r << 16) | (g << 8) | b).view(np.float32)

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        self.point_cloud_pub.publish(pc2.create_cloud(header, fields, np.column_stack((valid_points, rgb_float))))

def main(args=None):
    rclpy.init(args=args)
    node = StereoSGBMNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
