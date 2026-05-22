import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import message_filters
from rclpy.qos import qos_profile_sensor_data


class StereoCalibrationCapture(Node):
    def __init__(self):
        super().__init__('stereo_sync_capture_node')

        # 1. Setup Topics and Save Directory
        self.declare_parameter('left_topic', '/left/camera/image/compressed')
        self.declare_parameter('right_topic', '/right/camera/image/compressed')
        self.declare_parameter('save_dir', './calibration_dataset')

        left_topic = self.get_parameter('left_topic').value
        right_topic = self.get_parameter('right_topic').value
        base_dir = self.get_parameter('save_dir').value

        # 2. Automatically create the separate 'left' and 'right' folders
        self.left_dir = os.path.join(base_dir, 'left')
        self.right_dir = os.path.join(base_dir, 'right')
        os.makedirs(self.left_dir, exist_ok=True)
        os.makedirs(self.right_dir, exist_ok=True)

        self.img_count = 0

        # 3. Setup Synchronized Subscribers (The 50ms Bouncer)
        self.left_sub = message_filters.Subscriber(
            self, CompressedImage, left_topic, qos_profile=qos_profile_sensor_data)

        self.right_sub = message_filters.Subscriber(
            self, CompressedImage, right_topic, qos_profile=qos_profile_sensor_data)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub], queue_size=10, slop=0.05
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info(f"Folders created at: {base_dir}")
        self.get_logger().info("FOCUS THE OPENCV WINDOW: Press 'c' to capture, 'q' to quit.")

    def sync_callback(self, left_msg, right_msg):
        # 4. Decode the incoming JPEGs
        left_np = np.frombuffer(left_msg.data, np.uint8)
        right_np = np.frombuffer(right_msg.data, np.uint8)

        left_frame = cv2.imdecode(left_np, cv2.IMREAD_COLOR)
        right_frame = cv2.imdecode(right_np, cv2.IMREAD_COLOR)

        if left_frame is not None and right_frame is not None:
            # 5. Display side-by-side
            combined_frame = np.hstack((left_frame, right_frame))
            cv2.imshow("Stereo Sync Capture (Left | Right)", combined_frame)

            key = cv2.waitKey(1) & 0xFF

            # 6. Save images when 'c' is pressed
            if key == ord('c'):
                l_filename = os.path.join(
                    self.left_dir, f"left_{self.img_count:04d}.jpg")
                r_filename = os.path.join(
                    self.right_dir, f"right_{self.img_count:04d}.jpg")

                cv2.imwrite(l_filename, left_frame)
                cv2.imwrite(r_filename, right_frame)

                # Check Chrony accuracy
                l_time = left_msg.header.stamp.sec + \
                    (left_msg.header.stamp.nanosec * 1e-9)
                r_time = right_msg.header.stamp.sec + \
                    (right_msg.header.stamp.nanosec * 1e-9)
                diff_ms = abs(l_time - r_time) * 1000

                self.get_logger().info(
                    f"Saved Pair {self.img_count:04d} | Sync diff: {diff_ms:.2f} ms")
                self.img_count += 1

                # Flash the screen so you know it took the picture
                flash = np.ones(combined_frame.shape, dtype=np.uint8) * 255
                cv2.imshow("Stereo Sync Capture (Left | Right)", flash)
                cv2.waitKey(50)

            elif key == ord('q'):
                self.get_logger().info("Quitting...")
                rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = StereoCalibrationCapture()
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
