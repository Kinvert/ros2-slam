#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2

class SimpleCameraPublisher(Node):
    def __init__(self):
        super().__init__('simple_camera_publisher')
        self.publisher_ = self.create_publisher(CompressedImage, 'camera/image/compressed', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)  # 10 FPS
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        #self.cap.set(cv2.CAP_PROP_MONOCHROME, 1) # doesnt change
        #self.cap.set(cv2.CAP_PROP_FORMAT, cv2.CV_8UC1) # doesnt change
        #self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0) # seems to break it
        self.get_logger().info('Simple camera publisher started')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Encode frame as JPEG
            _, encoded_img = cv2.imencode('.jpg', frame)
            
            # Create CompressedImage message
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = encoded_img.tobytes()
            
            self.publisher_.publish(msg)
            self.get_logger().info('Published frame')
        else:
            self.get_logger().warn('Failed to capture frame')

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = SimpleCameraPublisher()
    try:
        rclpy.spin(camera_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        camera_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
