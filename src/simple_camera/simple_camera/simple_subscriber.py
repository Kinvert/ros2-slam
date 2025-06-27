#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class SimpleCameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.subscription = self.create_subscription(
            CompressedImage,
            'camera/image/compressed',
            self.image_callback,
            10)
        #self.bridge = CvBridge()
        self.get_logger().info('Camera subscriber started')

    def image_callback(self, msg):
        try:
            # Just decode to verify we can process it
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


            # Display the image
            cv2.imshow('Pi Camera Feed', cv_image)
            cv2.waitKey(1)

            self.get_logger().info(f'Received frame: {cv_image.shape[1]}x{cv_image.shape[0]}')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = SimpleCameraSubscriber()
    try:
        rclpy.spin(camera_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        camera_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
