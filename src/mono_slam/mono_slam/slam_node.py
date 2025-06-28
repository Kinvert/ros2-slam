#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import cv2
import numpy as np
from collections import deque

class MonoSLAM(Node):
    def __init__(self):
        super().__init__('mono_slam')
        
        # Subscribers
        self.image_sub = self.create_subscription(
            CompressedImage,
            'camera/image/compressed',
            self.image_callback,
            10)
        
        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, 'slam/pose', 10)
        self.points_pub = self.create_publisher(MarkerArray, 'slam/points', 10)
        self.trajectory_pub = self.create_publisher(Marker, 'slam/trajectory', 10)
        self.debug_image_pub = self.create_publisher(Image, 'slam/debug_image', 10)
        self.bridge = CvBridge()
        
        # SLAM state
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
        
        # Camera pose
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.camera_z = 0.0
        self.camera_R = np.eye(3)  # Current rotation matrix
        
        # Feature detector
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Map points (3D landmarks)
        self.map_points = []
        
        # Relocalization data
        self.keyframes = []  # Store keyframes with their poses and descriptors
        self.keyframe_interval = 10  # Add keyframe every N frames
        self.frame_count = 0
        
        # Trajectory
        self.trajectory = deque(maxlen=1000)
        
        # Camera matrix (you'll need to calibrate your camera for better results)
        # These are rough estimates - calibrate your actual camera!
        self.K = np.array([[509.13, 0, 327.5], # 500 0 320
                          [0, 509.65, 246.27], # 0 500 240
                          [0, 0, 1]], dtype=np.float32) # 0 0 1
        
        self.get_logger().info('SLAM node started')

    def image_callback(self, msg):
        try:
            # Decode image
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect features
            kp, des = self.orb.detectAndCompute(gray, None)
            
            self.frame_count += 1
            
            if self.prev_frame is not None and des is not None and self.prev_des is not None:
                # Match features
                matches = self.matcher.match(self.prev_des, des)
                matches = sorted(matches, key=lambda x: x.distance)
                
                if len(matches) > 10:  # Need sufficient matches
                    # Extract matched points
                    prev_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
                    curr_pts = np.float32([kp[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
                    
                    # Estimate essential matrix
                    E, mask = cv2.findEssentialMat(prev_pts, curr_pts, self.K, method=cv2.RANSAC)
                    
                    if E is not None:
                        # Recover pose
                        _, R, t, mask_pose = cv2.recoverPose(E, prev_pts, curr_pts, self.K)
                        
                        # Update camera pose properly
                        scale = 1.0  # Adjust this based on your scene scale
                        t_world = self.camera_R @ (t * scale)
                        self.camera_x += t_world[0, 0]
                        self.camera_y += t_world[1, 0]
                        self.camera_z += t_world[2, 0]
                        self.camera_R = self.camera_R @ R
                        
                        # Add to trajectory
                        self.trajectory.append((self.camera_x, self.camera_y, self.camera_z))
                        
                        # Triangulate points for map
                        if len(self.map_points) < 100:  # Limit map size
                            self.triangulate_points(prev_pts, curr_pts, R, t)
                        
                        # Check for relocalization opportunity
                        reloc_success = self.check_relocalization(kp, des)
                        if reloc_success:
                            self.get_logger().info("Relocalization successful!")
                        
                        # Add keyframe if needed
                        if self.frame_count % self.keyframe_interval == 0:
                            self.add_keyframe(kp, des)
                        
                        # Publish pose
                        self.publish_pose()
                        
                        # Publish visualization
                        self.publish_trajectory()
                        self.publish_map_points()
                        self.publish_debug_image(frame, kp, matches)
                        
                        self.get_logger().info(f'Position: x={self.camera_x:.2f}, y={self.camera_y:.2f}, z={self.camera_z:.2f}, Features: {len(matches)}')
            
            # Store current frame data
            self.prev_frame = gray.copy()
            self.prev_kp = kp
            self.prev_des = des
            
            # If no previous frame, still publish debug image with keypoints
            if self.prev_frame is None:
                self.publish_debug_image(frame, kp, [])
                # Add first keyframe
                if des is not None:
                    self.add_keyframe(kp, des)
            
        except Exception as e:
            self.get_logger().error(f'SLAM processing error: {str(e)}')

    def triangulate_points(self, prev_pts, curr_pts, R, t):
        """Simple triangulation to create 3D map points"""
        try:
            # Create projection matrices
            P1 = np.dot(self.K, np.hstack((np.eye(3), np.zeros((3, 1)))))
            P2 = np.dot(self.K, np.hstack((R, t)))
            
            # Triangulate a few points
            for i in range(0, min(len(prev_pts), 20), 5):  # Sample every 5th point
                pt1 = prev_pts[i].reshape(2)
                pt2 = curr_pts[i].reshape(2)
                
                # Triangulate
                point_4d = cv2.triangulatePoints(P1, P2, pt1.reshape(2, 1), pt2.reshape(2, 1))
                point_3d = point_4d[:3] / point_4d[3]
                
                # Add to map if reasonable depth
                if 0.1 < point_3d[2] < 10.0:
                    # Transform to world coordinates
                    point_world = self.camera_R @ point_3d.flatten() + np.array([self.camera_x, self.camera_y, self.camera_z])
                    self.map_points.append((
                        float(point_world[0]),
                        float(point_world[1]),
                        float(point_world[2])
                    ))
        except Exception as e:
            self.get_logger().warn(f'Triangulation error: {str(e)}')

    def publish_pose(self):
        """Publish current camera pose"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        
        pose_msg.pose.position.x = self.camera_x
        pose_msg.pose.position.y = self.camera_y
        pose_msg.pose.position.z = self.camera_z
        
        # Convert rotation matrix to quaternion using Shepperd method
        trace = self.camera_R[0,0] + self.camera_R[1,1] + self.camera_R[2,2]
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (self.camera_R[2,1] - self.camera_R[1,2]) / s
            y = (self.camera_R[0,2] - self.camera_R[2,0]) / s
            z = (self.camera_R[1,0] - self.camera_R[0,1]) / s
        else:
            if self.camera_R[0,0] > self.camera_R[1,1] and self.camera_R[0,0] > self.camera_R[2,2]:
                s = np.sqrt(1.0 + self.camera_R[0,0] - self.camera_R[1,1] - self.camera_R[2,2]) * 2
                w = (self.camera_R[2,1] - self.camera_R[1,2]) / s
                x = 0.25 * s
                y = (self.camera_R[0,1] + self.camera_R[1,0]) / s
                z = (self.camera_R[0,2] + self.camera_R[2,0]) / s
            elif self.camera_R[1,1] > self.camera_R[2,2]:
                s = np.sqrt(1.0 + self.camera_R[1,1] - self.camera_R[0,0] - self.camera_R[2,2]) * 2
                w = (self.camera_R[0,2] - self.camera_R[2,0]) / s
                x = (self.camera_R[0,1] + self.camera_R[1,0]) / s
                y = 0.25 * s
                z = (self.camera_R[1,2] + self.camera_R[2,1]) / s
            else:
                s = np.sqrt(1.0 + self.camera_R[2,2] - self.camera_R[0,0] - self.camera_R[1,1]) * 2
                w = (self.camera_R[1,0] - self.camera_R[0,1]) / s
                x = (self.camera_R[0,2] + self.camera_R[2,0]) / s
                y = (self.camera_R[1,2] + self.camera_R[2,1]) / s
                z = 0.25 * s

        pose_msg.pose.orientation.x = float(x)
        pose_msg.pose.orientation.y = float(y)
        pose_msg.pose.orientation.z = float(z)
        pose_msg.pose.orientation.w = float(w)
        
        self.pose_pub.publish(pose_msg)

    def publish_trajectory(self):
        """Publish trajectory as line markers"""
        if len(self.trajectory) < 2:
            return
            
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        marker.scale.x = 0.02  # Line width
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        for x, y, z in self.trajectory:
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = float(z)
            marker.points.append(point)
        
        self.trajectory_pub.publish(marker)

    def publish_map_points(self):
        """Publish 3D map points"""
        if not self.map_points:
            return
            
        marker_array = MarkerArray()
        
        for i, (x, y, z) in enumerate(self.map_points[-50:]):  # Show last 50 points
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = float(z)
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            marker_array.markers.append(marker)
        
        self.points_pub.publish(marker_array)

    def publish_debug_image(self, frame, keypoints, matches):
        """Publish debug image with keypoints and matches visualization"""
        try:
            debug_frame = frame.copy()
            
            # Draw all keypoints in blue
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(debug_frame, (x, y), 3, (255, 0, 0), 1)  # Blue circles
            
            # Draw matched keypoints in green if we have matches
            if matches and self.prev_kp is not None:
                for match in matches[:50]:  # Show first 50 matches
                    curr_kp = keypoints[match.trainIdx]
                    x, y = int(curr_kp.pt[0]), int(curr_kp.pt[1])
                    cv2.circle(debug_frame, (x, y), 5, (0, 255, 0), 2)  # Green circles for matches
            
            # Add text overlay with info
            cv2.putText(debug_frame, f'Keypoints: {len(keypoints)}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(debug_frame, f'Matches: {len(matches) if matches else 0}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(debug_frame, f'Pos: ({self.camera_x:.1f}, {self.camera_y:.1f}, {self.camera_z:.1f})',
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Convert OpenCV image to ROS Image message
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            debug_msg.header.frame_id = 'camera'
            
            self.debug_image_pub.publish(debug_msg)
            
        except Exception as e:
            self.get_logger().warn(f'Debug image publishing error: {str(e)}')

    def add_keyframe(self, keypoints, descriptors):
        """Add current frame as a keyframe for relocalization"""
        if descriptors is not None:
            keyframe = {
                'pose': (self.camera_x, self.camera_y, self.camera_z, self.camera_R.copy()),
                'descriptors': descriptors.copy(),
                'keypoints': keypoints
            }
            self.keyframes.append(keyframe)
            
            # Limit number of keyframes to prevent memory issues
            if len(self.keyframes) > 50:
                self.keyframes.pop(0)
            
            self.get_logger().info(f'Added keyframe {len(self.keyframes)}, total keyframes: {len(self.keyframes)}')

    def check_relocalization(self, current_kp, current_des):
        """Check if current frame matches any previous keyframes"""
        if current_des is None or len(self.keyframes) < 3:
            return False
        
        best_matches = 0
        best_keyframe = None
        
        # Compare against recent keyframes (skip the most recent few to avoid immediate matches)
        for i, keyframe in enumerate(self.keyframes[:-3]):
            matches = self.matcher.match(keyframe['descriptors'], current_des)
            matches = [m for m in matches if m.distance < 50]  # Filter good matches
            
            if len(matches) > best_matches and len(matches) > 20:  # Need significant matches
                best_matches = len(matches)
                best_keyframe = keyframe
        
        if best_keyframe is not None:
            # Found a good match - this is a relocalization opportunity
            self.get_logger().info(f'Relocalization candidate: {best_matches} matches')
            
            # Simple relocalization: slightly adjust pose toward the matched keyframe
            # (In a full SLAM system, you'd do pose optimization here)
            kf_x, kf_y, kf_z, kf_R = best_keyframe['pose']
            
            # Gentle correction toward the keyframe pose
            correction_factor = 0.1  # How much to trust the relocalization
            self.camera_x = (1 - correction_factor) * self.camera_x + correction_factor * kf_x
            self.camera_y = (1 - correction_factor) * self.camera_y + correction_factor * kf_y
            self.camera_z = (1 - correction_factor) * self.camera_z + correction_factor * kf_z
            
            return True
        
        return False

def main(args=None):
    rclpy.init(args=args)
    slam_node = MonoSLAM()
    try:
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        pass
    finally:
        slam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
