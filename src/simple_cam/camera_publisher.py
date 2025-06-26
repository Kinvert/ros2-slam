#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera/image', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 FPS
        self.cap = cv2.VideoCapture(0)  # /dev/video0
        self.bridge = CvBridge()
        self.get_logger().info('Camera publisher started')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert OpenCV image to ROS Image message
            img_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            self.publisher_.publish(img_msg)
        else:
            self.get_logger().warn('Failed to capture frame')

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()
    try:
        rclpy.spin(camera_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        camera_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
