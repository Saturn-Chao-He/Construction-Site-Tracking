#!/usr/bin/env python3

"""
Standalone 3D Multi-Object Tracking Script with Rotation, ROI, and Kalman Filter Speed Estimation
Applies same rotation and ROI filtering as bounding box generation process
Includes Kalman filter speed estimation and display in km/h for engineering vehicles
runs in conda environment
"""

import numpy as np
import copy
import math
import sys
import os
import time
import pickle
import glob
from pathlib import Path

# ROS imports
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Quaternion
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial.transform import Rotation as R

# Paths
POINTCLOUD_FOLDER = '/media/ie/disk2/Lidar_Data/Construction_Data_Q_Building/One_Lidar/7-23-2025/test/lidar/bin/1'
BBOX_FOLDER = '/home/ie/Documents/Lidar/Cluster/bbox'

# === Rotation and ROI Parameters (same as detection process) ===
PITCH_DEG = 23
ROLL_DEG = 2
YAW_DEG = 2

# Original ROI (before rotation)
x_min, x_max = 15, 70
y_min, y_max = -30, 30
z_min, z_max = -10, 10

# === Speed estimation parameters ===
FRAME_RATE = 10.0  # Hz - frame rate for speed calculation
MIN_SPEED_THRESHOLD = 0.0  # m/s - minimum speed to display

# === Rotation matrix ===
def get_rotation_matrix(pitch_deg=0, roll_deg=0, yaw_deg=0):
    """Create rotation matrix from Euler angles"""
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    yaw = np.radians(yaw_deg)
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

ROTATION_MATRIX = get_rotation_matrix(PITCH_DEG, ROLL_DEG, YAW_DEG)

# === Compute rotated ROI bounds ===
roi_corners = np.array([
    [x_min, y_min, z_min],
    [x_min, y_min, z_max],
    [x_min, y_max, z_min],
    [x_min, y_max, z_max],
    [x_max, y_min, z_min],
    [x_max, y_min, z_max],
    [x_max, y_max, z_min],
    [x_max, y_max, z_max],
])
rotated_corners = (ROTATION_MATRIX @ roi_corners.T).T
x_min_r, y_min_r, z_min_r = np.min(rotated_corners, axis=0)
x_max_r, y_max_r, z_max_r = np.max(rotated_corners, axis=0)

print(f"Original ROI: X[{x_min}, {x_max}], Y[{y_min}, {y_max}], Z[{z_min}, {z_max}]")
print(f"Rotated ROI:  X[{x_min_r:.1f}, {x_max_r:.1f}], Y[{y_min_r:.1f}, {y_max_r:.1f}], Z[{z_min_r:.1f}, {z_max_r:.1f}]")
print(f"Rotation: Pitch={PITCH_DEG}°, Roll={ROLL_DEG}°, Yaw={YAW_DEG}°")

class SimpleKalmanFilter:
    """Simplified Kalman Filter for 3D object tracking with classification and Kalman filter speed estimation"""
    def __init__(self, bbox_data, track_id, dt=0.2):
        self.id = track_id
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 0
        self.age = 0
        self.dt = dt  # Time step for speed calculation
        
        # State: [x, y, z, yaw, l, w, h, vx, vy, vz]
        self.state = np.zeros(10)
        self.state[:7] = bbox_data[:7]  # [x, y, z, yaw, l, w, h]
        self.classification = int(bbox_data[7]) if len(bbox_data) > 7 else 0  # Store classification separately
        
        # State covariance
        self.P = np.eye(10) * 1000
        self.P[7:, 7:] *= 1000  # High uncertainty for velocities initially
        
        # Process noise
        self.Q = np.eye(10) * 0.1
        self.Q[7:, 7:] *= 0.01  # Lower process noise for velocities
        
        # Measurement noise
        self.R = np.eye(7) * 1.0
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(10)
        self.F[0, 7] = self.dt  # x += vx * dt
        self.F[1, 8] = self.dt  # y += vy * dt
        self.F[2, 9] = self.dt  # z += vz * dt
        
        # Measurement matrix (observe position and dimensions)
        self.H = np.zeros((7, 10))
        self.H[:7, :7] = np.eye(7)
        
    def predict(self):
        """Predict next state"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        
    def update(self, measurement):
        """Update with measurement"""
        # Update Kalman filter
        y = measurement[:7] - self.H @ self.state  # Innovation (only use first 7 elements)
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.state = self.state + K @ y
        self.P = (np.eye(10) - K @ self.H) @ self.P
        
        # Update classification with new measurement
        if len(measurement) > 7:
            self.classification = int(measurement[7])
        
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
    def get_bbox(self):
        """Get current bounding box"""
        return self.state[:7]
    
    def get_classification(self):
        """Get object classification"""
        return self.classification
    
    def get_speed_kmh(self):
        """Get speed from Kalman filter velocity estimate in km/h"""
        velocity = self.state[7:10]  # [vx, vy, vz]
        speed_ms = np.linalg.norm(velocity[:2])  # Use only horizontal velocity
        speed_kmh = speed_ms * 3.6  # Convert m/s to km/h
        return speed_kmh if speed_ms >= MIN_SPEED_THRESHOLD else 0.0
    
    def get_velocity_vector(self):
        """Get velocity components from Kalman filter state"""
        return self.state[7:10]  # [vx, vy, vz]

class Simple3DTracker:
    """Simplified 3D Multi-Object Tracker with Kalman filter speed estimation"""
    def __init__(self, max_disappeared=5, min_hits=3, frame_rate=5.0):
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.track_id_count = 1
        self.dt = 1.0 / frame_rate  # Time step for Kalman filter
        
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
        
    def compute_iou_3d(self, bbox1, bbox2):
        """Compute 3D IoU between two bounding boxes"""
        # Simple distance-based association for now
        center1 = bbox1[:3]
        center2 = bbox2[:3]
        distance = np.linalg.norm(center1 - center2)
        return max(0, 5.0 - distance) / 5.0  # Simple distance to similarity
        
    def associate_detections(self, detections):
        """Associate detections with existing trackers"""
        if len(self.trackers) == 0:
            return [], list(range(len(detections)))
            
        # Compute similarity matrix
        similarities = np.zeros((len(self.trackers), len(detections)))
        for t, tracker in enumerate(self.trackers):
            tracker_bbox = tracker.get_bbox()
            for d, detection in enumerate(detections):
                similarities[t, d] = self.compute_iou_3d(tracker_bbox, detection)
                
        # Simple greedy assignment
        matches = []
        unmatched_trackers = list(range(len(self.trackers)))
        unmatched_detections = list(range(len(detections)))
        
        for _ in range(min(len(self.trackers), len(detections))):
            if len(unmatched_trackers) == 0 or len(unmatched_detections) == 0:
                break
                
            # Find best match
            max_sim = 0
            best_match = None
            for t in unmatched_trackers:
                for d in unmatched_detections:
                    if similarities[t, d] > max_sim:
                        max_sim = similarities[t, d]
                        best_match = (t, d)
                        
            if best_match and max_sim > 0.3:  # Threshold
                matches.append(best_match)
                unmatched_trackers.remove(best_match[0])
                unmatched_detections.remove(best_match[1])
            else:
                break
                
        return matches, unmatched_detections
        
    def update(self, detections):
        """Update tracker with new detections"""
        self.frame_count += 1
        
        # Predict all trackers
        for tracker in self.trackers:
            tracker.predict()
            
        # Associate detections with trackers
        matches, unmatched_detections = self.associate_detections(detections)
        
        # Update matched trackers
        for tracker_idx, detection_idx in matches:
            self.trackers[tracker_idx].update(detections[detection_idx])
            
        # Create new trackers for unmatched detections
        for detection_idx in unmatched_detections:
            new_tracker = SimpleKalmanFilter(detections[detection_idx], self.track_id_count, self.dt)
            self.trackers.append(new_tracker)
            self.track_id_count += 1
            
        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_disappeared]
        
        # Return valid tracks with speed information
        valid_tracks = []
        for tracker in self.trackers:
            if tracker.time_since_update < 1 and (tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                bbox = tracker.get_bbox()
                classification = tracker.get_classification()
                speed_kmh = tracker.get_speed_kmh()  # Speed in km/h
                velocity_vector = tracker.get_velocity_vector()
                
                valid_tracks.append({
                    'bbox': bbox,
                    'id': tracker.id,
                    'confidence': min(1.0, tracker.hits / 10.0),
                    'classification': classification,
                    'speed_kmh': speed_kmh,
                    'velocity_vector': velocity_vector,
                    'hits': tracker.hits
                })
                
        return valid_tracks

class StandaloneTrackingNode:
    """Standalone tracking node with rotation, ROI, and Kalman filter speed estimation"""
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('standalone_tracking_node', anonymous=True)
        
        # Publishers
        self.bbox_pub = rospy.Publisher('/tracking_bboxes', BoundingBoxArray, queue_size=10)
        self.marker_pub = rospy.Publisher('/tracking_markers', MarkerArray, queue_size=10)
        self.pointcloud_pub = rospy.Publisher('/pointcloud', PointCloud2, queue_size=10)
        self.roi_marker_pub = rospy.Publisher('/roi_markers', MarkerArray, queue_size=10)
        self.speed_marker_pub = rospy.Publisher('/speed_markers', MarkerArray, queue_size=10)
        
        # Initialize tracker with frame rate
        self.tracker = Simple3DTracker(max_disappeared=5, min_hits=1, frame_rate=FRAME_RATE)
        
        # Load file lists
        self.pointcloud_files = self.load_pointcloud_files()
        self.bbox_files = self.load_bbox_files()
        
        rospy.loginfo(f"Found {len(self.pointcloud_files)} point cloud files")
        rospy.loginfo(f"Found {len(self.bbox_files)} bbox files")
        rospy.loginfo(f"Applied rotation: Pitch={PITCH_DEG}°, Roll={ROLL_DEG}°, Yaw={YAW_DEG}°")
        rospy.loginfo(f"ROI bounds: X[{x_min_r:.1f}, {x_max_r:.1f}], Y[{y_min_r:.1f}, {y_max_r:.1f}], Z[{z_min_r:.1f}, {z_max_r:.1f}]")
        rospy.loginfo(f"Speed estimation: Frame rate={FRAME_RATE}Hz, Min speed={MIN_SPEED_THRESHOLD}m/s")
        
        self.current_frame = 0
        self.frame_rate = FRAME_RATE
        
        # Publish ROI visualization once
        self.publish_roi_visualization()
        
        # Start processing
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.frame_rate), self.process_frame)
        
    def load_pointcloud_files(self):
        """Load sorted list of point cloud files"""
        files = glob.glob(os.path.join(POINTCLOUD_FOLDER, "*.bin"))
        return sorted(files)
        
    def load_bbox_files(self):
        """Load sorted list of bbox files"""
        files = glob.glob(os.path.join(BBOX_FOLDER, "*_detections.txt"))
        return sorted(files)
        
    def apply_roi_filter(self, points):
        """Apply ROI filtering to point cloud (before rotation)"""
        if len(points) == 0:
            return points
            
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        return points[mask]
        
    def apply_rotation(self, points):
        """Apply rotation to point cloud"""
        if len(points) == 0:
            return points
            
        # Apply rotation matrix
        rotated_points = (ROTATION_MATRIX @ points.T).T
        return rotated_points
        
    def apply_rotated_roi_filter(self, points):
        """Apply ROI filtering after rotation"""
        if len(points) == 0:
            return points
            
        mask = (
            (points[:, 0] >= x_min_r) & (points[:, 0] <= x_max_r) &
            (points[:, 1] >= y_min_r) & (points[:, 1] <= y_max_r) &
            (points[:, 2] >= z_min_r) & (points[:, 2] <= z_max_r)
        )
        return points[mask]
        
    def load_pointcloud(self, file_path):
        """Load point cloud from .bin file with rotation and ROI applied"""
        try:
            # Load raw points
            raw_points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            points = raw_points[:, :3]  # x, y, z
            
            # Apply same processing as detection pipeline:
            # 1. Apply original ROI filter
            points = self.apply_roi_filter(points)
            rospy.logdebug(f"After original ROI: {len(points)} points")
            
            # 2. Apply rotation
            points = self.apply_rotation(points)
            rospy.logdebug(f"After rotation: {len(points)} points")
            
            # 3. Apply rotated ROI filter
            points = self.apply_rotated_roi_filter(points)
            rospy.logdebug(f"After rotated ROI: {len(points)} points")
            
            return points
            
        except Exception as e:
            rospy.logwarn(f"Failed to load point cloud {file_path}: {e}")
            return np.empty((0, 3))
            
    def load_detections(self, file_path):
        """Load detection bounding boxes with classification info"""
        try:
            if file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    dets = data['dets']  # [h, w, l, x, y, z, theta]
                    info = data.get('info', np.ones((len(dets), 2)) * 0.8)
                    
                    # Convert to [x, y, z, theta, l, w, h] format for tracking
                    if len(dets) > 0:
                        tracking_format = np.zeros((len(dets), 8))  # Added extra column for class
                        tracking_format[:, 0] = dets[:, 3]  # x
                        tracking_format[:, 1] = dets[:, 4]  # y
                        tracking_format[:, 2] = dets[:, 5]  # z
                        tracking_format[:, 3] = dets[:, 6]  # theta
                        tracking_format[:, 4] = dets[:, 2]  # l
                        tracking_format[:, 5] = dets[:, 1]  # w
                        tracking_format[:, 6] = dets[:, 0]  # h
                        tracking_format[:, 7] = info[:, 0]  # class (0=person, 2=vehicle)
                        return tracking_format
                    else:
                        return np.empty((0, 8))
                        
            elif file_path.endswith('.txt'):
                # Load from text file: h w l x y z theta label score
                data = np.loadtxt(file_path, comments='#')
                if data.size == 0:
                    return np.empty((0, 8))
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                    
                # Convert to [x, y, z, theta, l, w, h, class] format
                tracking_format = np.zeros((len(data), 8))
                tracking_format[:, 0] = data[:, 3]  # x
                tracking_format[:, 1] = data[:, 4]  # y
                tracking_format[:, 2] = data[:, 5]  # z
                tracking_format[:, 3] = data[:, 6]  # theta
                tracking_format[:, 4] = data[:, 2]  # l
                tracking_format[:, 5] = data[:, 1]  # w
                tracking_format[:, 6] = data[:, 0]  # h
                tracking_format[:, 7] = data[:, 7]  # class (0=person, 2=vehicle)
                return tracking_format
                
        except Exception as e:
            rospy.logwarn(f"Failed to load detections {file_path}: {e}")
            return np.empty((0, 8))
            
    def publish_roi_visualization(self):
        """Publish ROI bounds as wireframe box for visualization"""
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "velodyne"
        
        roi_markers = MarkerArray()
        
        # Create wireframe box for rotated ROI
        roi_marker = Marker()
        roi_marker.header = header
        roi_marker.type = Marker.LINE_LIST
        roi_marker.action = Marker.ADD
        roi_marker.id = 999  # Unique ID for ROI marker
        roi_marker.ns = "roi"
        
        # ROI center and dimensions
        roi_center = [(x_min_r + x_max_r) / 2, (y_min_r + y_max_r) / 2, (z_min_r + z_max_r) / 2]
        roi_dims = [x_max_r - x_min_r, y_max_r - y_min_r, z_max_r - z_min_r]
        
        # Set position
        roi_marker.pose.position.x = roi_center[0]
        roi_marker.pose.position.y = roi_center[1]
        roi_marker.pose.position.z = roi_center[2]
        roi_marker.pose.orientation.w = 1.0
        
        # Define ROI box corners
        l, w, h = roi_dims[0], roi_dims[1], roi_dims[2]
        corners = [
            [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2], [l/2, w/2, -h/2], [-l/2, w/2, -h/2],  # bottom
            [-l/2, -w/2, h/2], [l/2, -w/2, h/2], [l/2, w/2, h/2], [-l/2, w/2, h/2],     # top
        ]
        
        # Define edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
        ]
        
        # Create line points
        points = []
        for edge in edges:
            start_corner = corners[edge[0]]
            end_corner = corners[edge[1]]
            
            start_point = Point()
            start_point.x, start_point.y, start_point.z = start_corner
            end_point = Point()
            end_point.x, end_point.y, end_point.z = end_corner
            
            points.extend([start_point, end_point])
        
        roi_marker.points = points
        
        # ROI appearance (yellow wireframe)
        roi_marker.scale.x = 0.2  # Thicker lines
        roi_marker.color.r = 1.0
        roi_marker.color.g = 1.0
        roi_marker.color.b = 0.0  # Yellow
        roi_marker.color.a = 0.5  # Semi-transparent
        roi_marker.lifetime = rospy.Duration(0)  # Persistent
        
        roi_markers.markers.append(roi_marker)
        
        # Add text label for ROI
        roi_label = Marker()
        roi_label.header = header
        roi_label.type = Marker.TEXT_VIEW_FACING
        roi_label.action = Marker.ADD
        roi_label.id = 998
        roi_label.ns = "roi"
        
        roi_label.pose.position.x = roi_center[0]
        roi_label.pose.position.y = roi_center[1]
        roi_label.pose.position.z = roi_center[2] + h/2 + 2.0
        roi_label.pose.orientation.w = 1.0
        
        roi_label.text = f"ROI\n{roi_dims[0]:.1f}m x {roi_dims[1]:.1f}m x {roi_dims[2]:.1f}m\nRotated: P{PITCH_DEG}° R{ROLL_DEG}° Y{YAW_DEG}°"
        roi_label.scale.z = 1.0
        roi_label.color.r = 1.0
        roi_label.color.g = 1.0
        roi_label.color.b = 0.0
        roi_label.color.a = 1.0
        roi_label.lifetime = rospy.Duration(0)
        
        roi_markers.markers.append(roi_label)
        self.roi_marker_pub.publish(roi_markers)
        
    def create_pointcloud_msg(self, points, frame_id="velodyne"):
        """Create PointCloud2 message"""
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        
        return pc2.create_cloud(header, fields, points)
        
    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion"""
        r = R.from_euler('z', yaw, degrees=False)
        return r.as_quat()
        
    def process_frame(self, event):
        """Process next frame"""
        if self.current_frame >= min(len(self.pointcloud_files), len(self.bbox_files)):
            rospy.loginfo("All frames processed")
            self.timer.shutdown()
            return
            
        start_time = time.time()
        
        # Load point cloud (with rotation and ROI applied)
        pc_file = self.pointcloud_files[min(self.current_frame, len(self.pointcloud_files)-1)]
        points = self.load_pointcloud(pc_file)
        
        # Load detections (already in rotated coordinate frame)
        bbox_file = self.bbox_files[min(self.current_frame, len(self.bbox_files)-1)]
        detections = self.load_detections(bbox_file)
        
        rospy.loginfo(f"Frame {self.current_frame}: {len(detections)} detections, {len(points)} points (after rotation+ROI)")
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Log speed information for vehicles
        for track in tracks:
            if track['classification'] == 2:  # Vehicle
                speed_kmh = track['speed_kmh']
                rospy.loginfo(f"Vehicle ID {track['id']}: Speed = {speed_kmh:.1f} km/h")
        
        # Publish point cloud
        if len(points) > 0:
            pc_msg = self.create_pointcloud_msg(points)
            self.pointcloud_pub.publish(pc_msg)
            
        # Publish tracking results
        self.publish_tracks(tracks)
        
        processing_time = time.time() - start_time
        rospy.loginfo(f"Frame {self.current_frame}: {len(tracks)} tracks, processing time: {processing_time:.3f}s")
        
        self.current_frame += 1
        
    def publish_tracks(self, tracks):
        """Publish tracking results with speed information using standard RViz markers"""
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "velodyne"
        
        # Create marker arrays for boxes, labels, and speed indicators
        box_marker_array = MarkerArray()
        label_marker_array = MarkerArray()
        speed_marker_array = MarkerArray()
        
        for i, track in enumerate(tracks):
            bbox_data = track['bbox']  # [x, y, z, theta, l, w, h]
            track_id = track['id']
            confidence = track['confidence']
            classification = track['classification']  # 0=person, 2=vehicle
            speed_kmh = track['speed_kmh']  # km/h from Kalman filter
            velocity_vector = track['velocity_vector']  # [vx, vy, vz]
            
            # Convert class ID to string
            if classification == 0:
                class_name = "person"
                box_color = (0.0, 1.0, 0.0)  # Green for person
            elif classification == 2:
                class_name = "vehicle"
                box_color = (0.0, 0.0, 1.0)  # Blue for vehicle
            else:
                class_name = "unknown"
                box_color = (1.0, 1.0, 0.0)  # Yellow for unknown
            
            if confidence < 0.3:  # Skip low confidence tracks
                continue
                
            # Create wireframe bounding box using LINE_LIST marker
            box_marker = Marker()
            box_marker.header = header
            box_marker.type = Marker.LINE_LIST
            box_marker.action = Marker.ADD
            box_marker.id = int(track_id)
            box_marker.ns = "boxes"
            
            # Position and orientation (already in rotated frame)
            box_marker.pose.position.x = float(bbox_data[0])
            box_marker.pose.position.y = float(bbox_data[1])
            box_marker.pose.position.z = float(bbox_data[2])
            
            q = self.yaw_to_quaternion(bbox_data[3])
            box_marker.pose.orientation.x = float(q[0])
            box_marker.pose.orientation.y = float(q[1])
            box_marker.pose.orientation.z = float(q[2])
            box_marker.pose.orientation.w = float(q[3])
            
            # Create wireframe box lines
            l, w, h = float(bbox_data[4]), float(bbox_data[5]), float(bbox_data[6])
            
            # Define 8 corners of the box (relative to center)
            corners = [
                [-l/2, -w/2, -h/2],  # 0: back-left-bottom
                [ l/2, -w/2, -h/2],  # 1: front-left-bottom
                [ l/2,  w/2, -h/2],  # 2: front-right-bottom
                [-l/2,  w/2, -h/2],  # 3: back-right-bottom
                [-l/2, -w/2,  h/2],  # 4: back-left-top
                [ l/2, -w/2,  h/2],  # 5: front-left-top
                [ l/2,  w/2,  h/2],  # 6: front-right-top
                [-l/2,  w/2,  h/2],  # 7: back-right-top
            ]
            
            # Define lines connecting corners (12 edges of a box)
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
                (4, 5), (5, 6), (6, 7), (7, 4),  # top face
                (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
            ]
            
            # Add line points
            points = []
            for edge in edges:
                start_corner = corners[edge[0]]
                end_corner = corners[edge[1]]
                
                start_point = Point()
                start_point.x, start_point.y, start_point.z = start_corner
                end_point = Point()
                end_point.x, end_point.y, end_point.z = end_corner
                
                points.extend([start_point, end_point])
            
            box_marker.points = points
            
            # Appearance - use classification-based colors
            box_marker.scale.x = 0.15  # Slightly thicker lines
            box_marker.color.r, box_marker.color.g, box_marker.color.b = box_color
            box_marker.color.a = 0.9
            
            box_marker.lifetime = rospy.Duration(0.5)
            box_marker_array.markers.append(box_marker)
            
            # Create text label marker with speed information
            label_marker = Marker()
            label_marker.header = header
            label_marker.type = Marker.TEXT_VIEW_FACING
            label_marker.action = Marker.ADD
            label_marker.id = int(track_id)
            label_marker.ns = "labels"
            
            # Position above bbox
            label_marker.pose.position.x = float(bbox_data[0])
            label_marker.pose.position.y = float(bbox_data[1])
            label_marker.pose.position.z = float(bbox_data[2]) + h/2 + 1.0
            label_marker.pose.orientation.w = 1.0
            
            # Text content with classification and speed
            speed_threshold_kmh = MIN_SPEED_THRESHOLD * 3.6  # Convert threshold to km/h
            
            if classification == 2 and speed_kmh > speed_threshold_kmh:  # Vehicle with significant speed
                label_marker.text = f"ID: {track_id}\n{class_name.capitalize()}\nSpeed: {speed_kmh:.1f} km/h"
            else:
                label_marker.text = f"ID: {track_id}\n{class_name.capitalize()}"
            
            # Appearance
            # label_marker.scale.z = 0.8
            # label_marker.color.r = 1.0
            # label_marker.color.g = 1.0
            # label_marker.color.b = 1.0
            # label_marker.color.a = 1.0
            label_marker.scale.z = 0.8
            label_marker.color.r = 1.0
            label_marker.color.g = 0
            label_marker.color.b = 0
            label_marker.color.a = 1.0
            
            label_marker.lifetime = rospy.Duration(0.5)
            label_marker_array.markers.append(label_marker)
            
            # Create speed vector visualization for moving vehicles
            speed_threshold_kmh = MIN_SPEED_THRESHOLD * 3.6  # Convert threshold to km/h
            if classification == 2 and speed_kmh > speed_threshold_kmh:
                speed_vector_marker = Marker()
                speed_vector_marker.header = header
                speed_vector_marker.type = Marker.ARROW
                speed_vector_marker.action = Marker.ADD
                speed_vector_marker.id = int(track_id)
                speed_vector_marker.ns = "speed_vectors"
                
                # Start position (center of vehicle)
                speed_vector_marker.pose.position.x = float(bbox_data[0])
                speed_vector_marker.pose.position.y = float(bbox_data[1])
                speed_vector_marker.pose.position.z = float(bbox_data[2]) + h/2 + 0.5
                
                # Direction based on velocity vector (use Kalman filter velocity)
                vx, vy = velocity_vector[0], velocity_vector[1]
                velocity_magnitude = np.sqrt(vx*vx + vy*vy)
                
                if velocity_magnitude > 0.1:  # Only show if moving
                    # Calculate arrow direction
                    velocity_angle = np.arctan2(vy, vx)
                    q_vel = self.yaw_to_quaternion(velocity_angle)
                    speed_vector_marker.pose.orientation.x = float(q_vel[0])
                    speed_vector_marker.pose.orientation.y = float(q_vel[1])
                    speed_vector_marker.pose.orientation.z = float(q_vel[2])
                    speed_vector_marker.pose.orientation.w = float(q_vel[3])
                    
                    # Scale arrow based on speed (length proportional to speed in km/h)
                    arrow_length = min(speed_kmh * 0.3, 10.0)  # Scale factor, max 10m
                    speed_vector_marker.scale.x = arrow_length  # Arrow length
                    speed_vector_marker.scale.y = 0.3  # Arrow width
                    speed_vector_marker.scale.z = 0.3  # Arrow height
                    
                    # Color based on speed (green = slow, yellow = medium, red = fast)
                    if speed_kmh < 7.2:  # < 7.2 km/h
                        speed_vector_marker.color.r = 0.0
                        speed_vector_marker.color.g = 1.0
                        speed_vector_marker.color.b = 0.0
                    elif speed_kmh < 18.0:  # < 18 km/h
                        speed_vector_marker.color.r = 1.0
                        speed_vector_marker.color.g = 1.0
                        speed_vector_marker.color.b = 0.0
                    else:  # >= 18 km/h
                        speed_vector_marker.color.r = 1.0
                        speed_vector_marker.color.g = 0.0
                        speed_vector_marker.color.b = 0.0
                    
                    speed_vector_marker.color.a = 0.8
                    speed_vector_marker.lifetime = rospy.Duration(0.5)
                    speed_marker_array.markers.append(speed_vector_marker)
                
                # Create speed text marker
                speed_text_marker = Marker()
                speed_text_marker.header = header
                speed_text_marker.type = Marker.TEXT_VIEW_FACING
                speed_text_marker.action = Marker.ADD
                speed_text_marker.id = int(track_id) + 1000  # Offset to avoid ID conflicts
                speed_text_marker.ns = "speed_text"
                
                # Position next to the vehicle
                speed_text_marker.pose.position.x = float(bbox_data[0]) + l/2 + 1.0
                speed_text_marker.pose.position.y = float(bbox_data[1])
                speed_text_marker.pose.position.z = float(bbox_data[2]) + h/2
                speed_text_marker.pose.orientation.w = 1.0
                
                # Speed text in km/h only
                speed_text_marker.text = f"{speed_kmh:.1f} km/h"
                
                # Appearance
                speed_text_marker.scale.z = 0.6
                if speed_kmh < 7.2:
                    speed_text_marker.color.r = 0.0
                    speed_text_marker.color.g = 1.0
                    speed_text_marker.color.b = 0.0
                elif speed_kmh < 18.0:
                    speed_text_marker.color.r = 1.0
                    speed_text_marker.color.g = 1.0
                    speed_text_marker.color.b = 0.0
                else:
                    speed_text_marker.color.r = 1.0
                    speed_text_marker.color.g = 0.0
                    speed_text_marker.color.b = 0.0
                
                speed_text_marker.color.a = 1.0
                speed_text_marker.lifetime = rospy.Duration(0.5)
                speed_marker_array.markers.append(speed_text_marker)
                
        # Publish marker arrays
        self.marker_pub.publish(label_marker_array)
        self.speed_marker_pub.publish(speed_marker_array)
        
        # Publish box markers on a separate topic
        if not hasattr(self, 'box_marker_pub'):
            self.box_marker_pub = rospy.Publisher('/tracking_boxes_markers', MarkerArray, queue_size=10)
        self.box_marker_pub.publish(box_marker_array)

def main():
    try:
        # Check if paths exist
        if not os.path.exists(POINTCLOUD_FOLDER):
            print(f"Error: Point cloud folder not found: {POINTCLOUD_FOLDER}")
            return
            
        if not os.path.exists(BBOX_FOLDER):
            print(f"Error: Bounding box folder not found: {BBOX_FOLDER}")
            return
            
        print("Starting standalone tracking node with rotation, ROI, and Kalman filter speed estimation...")
        print(f"Point cloud folder: {POINTCLOUD_FOLDER}")
        print(f"Bounding box folder: {BBOX_FOLDER}")
        print(f"Rotation applied: Pitch={PITCH_DEG}°, Roll={ROLL_DEG}°, Yaw={YAW_DEG}°")
        print(f"Original ROI: X[{x_min}, {x_max}], Y[{y_min}, {y_max}], Z[{z_min}, {z_max}]")
        print(f"Rotated ROI bounds: X[{x_min_r:.1f}, {x_max_r:.1f}], Y[{y_min_r:.1f}, {y_max_r:.1f}], Z[{z_min_r:.1f}, {z_max_r:.1f}]")
        print(f"Speed estimation: Frame rate={FRAME_RATE}Hz, Min threshold={MIN_SPEED_THRESHOLD}m/s ({MIN_SPEED_THRESHOLD*3.6:.1f}km/h)")
        print("\nTopics published:")
        print("  - /tracking_bboxes (BoundingBoxArray)")
        print("  - /tracking_markers (MarkerArray) - labels with speed info in km/h")  
        print("  - /tracking_boxes_markers (MarkerArray) - bounding boxes")
        print("  - /speed_markers (MarkerArray) - speed vectors and text in km/h")
        print("  - /pointcloud (PointCloud2) - rotated and ROI filtered")
        print("  - /roi_markers (MarkerArray) - ROI boundary visualization")
        print("\nSpeed visualization (km/h):")
        print("  - Green arrows/text: < 7.2 km/h")
        print("  - Yellow arrows/text: 7.2-18 km/h")
        print("  - Red arrows/text: > 18 km/h")
        print("  - Arrow length proportional to speed")
        print("  - Speed displayed in km/h only")
        print("  - Uses Kalman filter velocity estimates")
        print("\nStart RViz and subscribe to these topics to visualize tracking results with speed.")
        print("The yellow wireframe shows the ROI bounds after rotation is applied.")
        
        node = StandaloneTrackingNode()
        rospy.spin()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()