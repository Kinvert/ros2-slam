#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
import matplotlib.pyplot as plt
from collections import deque
import threading

class SLAMVisualizer(Node):
    def __init__(self):
        super().__init__('slam_visualizer')
        
        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped,
            'slam/pose',
            self.pose_callback,
            10)
        
        self.points_sub = self.create_subscription(
            MarkerArray,
            'slam/points',
            self.points_callback,
            10)
        
        # Data storage
        self.poses = deque(maxlen=1000)
        self.map_points = []
        
        # Plotting
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('SLAM Trajectory and Map Points')
        self.ax.grid(True)
        self.ax.axis('equal')
        
        # Start plotting in separate thread
        self.plot_thread = threading.Thread(target=self.plot_loop)
        self.plot_thread.daemon = True
        self.plot_thread.start()
        
        self.get_logger().info('SLAM visualizer started')

    def pose_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        self.poses.append((x, y, z))

    def points_callback(self, msg):
        self.map_points = []
        for marker in msg.markers:
            x = marker.pose.position.x
            y = marker.pose.position.y
            z = marker.pose.position.z
            self.map_points.append((x, y, z))

    def plot_loop(self):
        """Continuous plotting loop"""
        plt.ion()
        
        while rclpy.ok():
            try:
                self.ax.clear()
                self.ax.set_xlabel('X (m)')
                self.ax.set_ylabel('Y (m)')
                self.ax.set_title('SLAM Trajectory and Map Points')
                self.ax.grid(True)
                
                # Plot trajectory
                if len(self.poses) > 1:
                    x_traj = [pose[0] for pose in self.poses]
                    y_traj = [pose[1] for pose in self.poses]
                    self.ax.plot(x_traj, y_traj, 'g-', linewidth=2, label='Trajectory')
                    
                    # Mark current position
                    if self.poses:
                        curr_x, curr_y, _ = self.poses[-1]
                        self.ax.plot(curr_x, curr_y, 'go', markersize=8, label='Current Position')
                
                # Plot map points
                if self.map_points:
                    x_points = [point[0] for point in self.map_points]
                    y_points = [point[1] for point in self.map_points]
                    self.ax.scatter(x_points, y_points, c='red', s=10, alpha=0.6, label='Map Points')
                
                self.ax.legend()
                self.ax.axis('equal')
                plt.pause(0.1)
                
            except Exception as e:
                self.get_logger().error(f'Plotting error: {str(e)}')
                break

def main(args=None):
    rclpy.init(args=args)
    visualizer = SLAMVisualizer()
    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
