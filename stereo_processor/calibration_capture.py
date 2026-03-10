import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class CalibrationCapture(Node):
    def __init__(self):
        super().__init__('calibration_capture_node')

        # 1. Setup Parameters
        self.declare_parameter('topic_name', '/camera/image/compressed')
        self.declare_parameter('save_dir', './calibration_images')

        topic_name = self.get_parameter('topic_name').value
        self.save_dir = self.get_parameter('save_dir').value
        self.img_count = 0

        # 2. Create the save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            self.get_logger().info(f"Created directory: {self.save_dir}")

        # 3. Subscribe to the Pi's compressed stream
        self.subscription = self.create_subscription(
            CompressedImage,
            topic_name,
            self.image_callback,
            10
        )

        self.get_logger().info(f"Listening to {topic_name}")
        self.get_logger().info("-----------------------------------------")
        self.get_logger().info("FOCUS WINDOW & PRESS 'c' TO CAPTURE IMAGE")
        self.get_logger().info("PRESS 'q' TO QUIT")
        self.get_logger().info("-----------------------------------------")

    def image_callback(self, msg):
        # 4. Decode the raw JPEG bytes into an OpenCV matrix
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is not None:
            # 5. Display the live feed
            cv2.imshow("Stereo Calibration Capture", frame)

            # 6. Listen for keystrokes (wait 1ms per frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                # Save the image with an incrementing counter
                filename = os.path.join(
                    self.save_dir, f"checkerboard_{self.img_count:04d}.jpg")
                cv2.imwrite(filename, frame)
                self.get_logger().info(f"Successfully saved: {filename}")
                self.img_count += 1

                # Visual flash to confirm capture
                flash = np.ones(frame.shape, dtype=np.uint8) * 255
                cv2.imshow("Stereo Calibration Capture", flash)
                cv2.waitKey(50)

            elif key == ord('q'):
                self.get_logger().info("Quitting capture node...")
                rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CalibrationCapture()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up the OpenCV window when closing
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
