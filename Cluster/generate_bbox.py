# Modified gpu-detection.py with tracking format output and orientation calculation
import os
import time
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from cuml.cluster import DBSCAN as cuDBSCAN
from cuml.neighbors import NearestNeighbors
import cupy as cp
import pickle
from sklearn.decomposition import PCA

# === Adjustable Rotation Angles (in degrees) ===
PITCH_DEG = 23
ROLL_DEG = 2
YAW_DEG = 2

# === Parameters ===
input_folder = "/media/ie/disk2/Lidar_Data/Construction_Data_Q_Building/7-23-2025/test/lidar/bin/1"
background_file = "/media/ie/disk2/Lidar_Data/Construction_Data_Q_Building/7-23-2025/test/lidar/bin/2/005100.bin"
output_folder = "bbox"  # NEW: folder to save detection results

voxel_size = 0.4
dbscan_eps = 1.0
min_points = 5
max_points = 400
frame_interval = 0.1

x_min, x_max = 15, 70
y_min, y_max = -30, 30
z_min, z_max = -10, 10

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# === Rotation matrix ===
def get_rotation_matrix(pitch_deg=0, roll_deg=0, yaw_deg=0):
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

# === NEW: Function to compute orientation using PCA ===
def compute_bbox_orientation(points):
    """
    Compute the orientation of a point cloud using PCA
    
    Args:
        points: numpy array of 3D points (N, 3)
    
    Returns:
        theta: orientation angle around Z-axis in radians
        center: center of the bounding box
        dimensions: [length, width, height] of the oriented bounding box
    """
    if len(points) < 3:
        # Not enough points for PCA, return axis-aligned
        center = np.mean(points, axis=0)
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        dimensions = max_coords - min_coords
        return 0.0, center, dimensions
    
    # Center the points
    center = np.mean(points, axis=0)
    centered_points = points - center
    
    # Apply PCA to find principal components
    # Only use X and Y coordinates for orientation (ignore Z)
    xy_points = centered_points[:, :2]
    
    if len(np.unique(xy_points, axis=0)) < 2:
        # All points are the same in XY plane
        return 0.0, center, np.array([1.0, 1.0, np.max(centered_points[:, 2]) - np.min(centered_points[:, 2])])
    
    pca = PCA(n_components=2)
    pca.fit(xy_points)
    
    # Get the principal component (longest axis in XY plane)
    principal_component = pca.components_[0]
    
    # Calculate angle of principal component relative to X-axis
    theta = np.arctan2(principal_component[1], principal_component[0])
    
    # Ensure theta is in [-pi, pi]
    theta = np.arctan2(np.sin(theta), np.cos(theta))
    
    # Create rotation matrix for the computed angle
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix_2d = np.array([[cos_theta, sin_theta],
                                   [-sin_theta, cos_theta]])
    
    # Rotate points to aligned coordinate system
    rotated_xy = xy_points @ rotation_matrix_2d.T
    
    # Compute dimensions in the rotated coordinate system
    min_rotated = np.min(rotated_xy, axis=0)
    max_rotated = np.max(rotated_xy, axis=0)
    
    # Length and width in oriented coordinate system
    length = max_rotated[0] - min_rotated[0]  # Along principal component
    width = max_rotated[1] - min_rotated[1]   # Perpendicular to principal component
    
    # Height remains the same (Z-direction)
    height = np.max(centered_points[:, 2]) - np.min(centered_points[:, 2])
    
    # Ensure length >= width (length should be the longer dimension)
    if width > length:
        length, width = width, length
        theta += np.pi/2  # Rotate by 90 degrees
        theta = np.arctan2(np.sin(theta), np.cos(theta))  # Normalize again
    
    dimensions = np.array([length, width, height])
    
    return theta, center, dimensions

# === MODIFIED: Function to convert cluster to tracking format ===
def convert_cluster_to_tracking_format(cluster_points, label_str, score=0.8):
    """
    Convert point cluster to oriented bounding box in tracking format
    
    Args:
        cluster_points: numpy array of 3D points (N, 3)
        label_str: "person" or "vehicle"
        score: confidence score (default 0.8)
    
    Returns:
        box_array: [h, w, l, x, y, z, θ] format
        info_array: [label_id, score] format
    """
    # Compute oriented bounding box
    theta, center, dimensions = compute_bbox_orientation(cluster_points)
    
    # Extract dimensions
    length, width, height = dimensions
    
    # Center coordinates
    x, y, z = center[0], center[1], center[2]
    
    # Convert label string to ID
    label_map = {"person": 0, "vehicle": 2}  # 0=Pedestrian, 2=Car (skipping Cyclist=1)
    label_id = label_map.get(label_str, 0)
    
    # Create arrays in required format
    box_array = np.array([height, width, length, x, y, z, theta])  # [h, w, l, x, y, z, θ]
    info_array = np.array([label_id, score])                       # [label, score]
    
    return box_array, info_array

# === MODIFIED: Function to save detection results ===
def save_detections(frame_name, clusters_list, output_folder):
    """
    Save detection results in format compatible with modelROS.py
    
    Args:
        frame_name: name of the frame file
        clusters_list: list of (cluster_points, label) tuples
        output_folder: folder to save results
    """
    if not clusters_list:
        # Save empty detection
        detection_data = {
            'dets': np.empty((0, 7)),  # Empty array with correct shape
            'info': np.empty((0, 2))
        }
    else:
        # Convert all clusters to tracking format
        box_arrays = []
        info_arrays = []
        
        for cluster_points, label in clusters_list:
            box_arr, info_arr = convert_cluster_to_tracking_format(cluster_points, label)
            box_arrays.append(box_arr)
            info_arrays.append(info_arr)
        
        # Stack into numpy arrays
        detection_data = {
            'dets': np.array(box_arrays),   # Shape: (N, 7) - [h,w,l,x,y,z,θ]
            'info': np.array(info_arrays)   # Shape: (N, 2) - [label,score]
        }
    
    # # Save as pickle file (easier to load)
    # output_file = os.path.join(output_folder, f"{frame_name.replace('.bin', '')}_detections.pkl")
    # with open(output_file, 'wb') as f:
    #     pickle.dump(detection_data, f)
    
    # Also save as human-readable txt file
    txt_file = os.path.join(output_folder, f"{frame_name.replace('.bin', '')}_detections.txt")
    with open(txt_file, 'w') as f:
        f.write(f"# Frame: {frame_name}\n")
        f.write(f"# Format: h w l x y z theta label score\n")
        for i, (box_arr, info_arr) in enumerate(zip(detection_data['dets'], detection_data['info'])):
            f.write(f"{box_arr[0]:.3f} {box_arr[1]:.3f} {box_arr[2]:.3f} "
                   f"{box_arr[3]:.3f} {box_arr[4]:.3f} {box_arr[5]:.3f} "
                   f"{box_arr[6]:.3f} {int(info_arr[0])} {info_arr[1]:.3f}\n")
    
    # print(f"Saved {len(clusters_list)} detections to {output_file}")

# === Function to create oriented bounding box for visualization ===
def create_oriented_bbox_for_visualization(cluster_points):
    """
    Create an oriented bounding box for visualization
    
    Args:
        cluster_points: numpy array of 3D points
        
    Returns:
        oriented_bbox: Open3D OrientedBoundingBox
    """
    if len(cluster_points) < 3:
        # Fall back to axis-aligned bounding box
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(cluster_points)
        return pcd_temp.get_axis_aligned_bounding_box()
    
    # Compute orientation and dimensions
    theta, center, dimensions = compute_bbox_orientation(cluster_points)
    
    # Create oriented bounding box
    oriented_bbox = o3d.geometry.OrientedBoundingBox()
    oriented_bbox.center = center
    oriented_bbox.extent = dimensions
    
    # Create rotation matrix around Z-axis
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix_3d = np.array([[cos_theta, -sin_theta, 0],
                                   [sin_theta,  cos_theta, 0],
                                   [0,          0,         1]])
    
    oriented_bbox.R = rotation_matrix_3d
    
    return oriented_bbox

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

# === Load and voxelize background ===
bg_points = np.fromfile(background_file, dtype=np.float32).reshape(-1, 4)[:, :3]
roi_mask = (
    (bg_points[:, 0] >= x_min) & (bg_points[:, 0] <= x_max) &
    (bg_points[:, 1] >= y_min) & (bg_points[:, 1] <= y_max) &
    (bg_points[:, 2] >= z_min) & (bg_points[:, 2] <= z_max)
)
bg_points = bg_points[roi_mask]
bg_pc = o3d.geometry.PointCloud()
bg_pc.points = o3d.utility.Vector3dVector(bg_points)
bg_pc = bg_pc.voxel_down_sample(voxel_size)
bg_xyz = np.asarray(bg_pc.points)

# === cuML GPU KD-tree (background)
bg_tree = NearestNeighbors(n_neighbors=1, radius=dbscan_eps, output_type='numpy')
bg_tree.fit(bg_xyz)

# === Load frame list ===
frame_files = sorted(f for f in os.listdir(input_folder) if f.endswith(".bin"))
frame_idx = 0

# === Open3D GUI setup ===
gui.Application.instance.initialize()
window = gui.Application.instance.create_window("Detection with Oriented Bounding Boxes", 1280, 720)
scene = gui.SceneWidget()
scene.scene = rendering.Open3DScene(window.renderer)
window.add_child(scene)

pc_material = rendering.MaterialRecord()
pc_material.shader = "defaultUnlit"

bbox_material = rendering.MaterialRecord()
bbox_material.shader = "unlitLine"
bbox_material.line_width = 2.0

scene.scene.set_background(np.array([0, 0, 0, 1], dtype=np.float32))
scene.scene.show_axes(True)

def update_scene():
    global frame_idx
    if frame_idx >= len(frame_files):
        print("All frames complete.")
        return

    # === Load frame
    t0 = time.time()
    file_path = os.path.join(input_folder, frame_files[frame_idx])
    raw_points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    print(f"[Timing] Load frame: {time.time() - t0:.3f}s")

    # === ROI filtering
    t1 = time.time()
    roi_mask = (
        (raw_points[:, 0] >= x_min) & (raw_points[:, 0] <= x_max) &
        (raw_points[:, 1] >= y_min) & (raw_points[:, 1] <= y_max) &
        (raw_points[:, 2] >= z_min) & (raw_points[:, 2] <= z_max)
    )
    curr_points = raw_points[roi_mask]
    print(f"[Timing] ROI filtering: {time.time() - t1:.3f}s")

    # === Background subtraction
    t2 = time.time()
    curr_pc = o3d.geometry.PointCloud()
    curr_pc.points = o3d.utility.Vector3dVector(curr_points)
    curr_pc = curr_pc.voxel_down_sample(voxel_size)
    xyz = np.asarray(curr_pc.points)

    # === cuML GPU background subtraction
    dist, _ = bg_tree.kneighbors(xyz)
    fg_points = xyz[dist.flatten() > voxel_size]
    if len(fg_points) == 0:
        print(f"{frame_idx} → No foreground")
        # Save empty detections
        save_detections(frame_files[frame_idx], [], output_folder)
        
        # Schedule next frame
        frame_idx += 1
        gui.Application.instance.run_one_tick()
        time.sleep(frame_interval)
        gui.Application.instance.post_to_main_thread(window, update_scene)
        return
        
    print(f"[Timing] Background subtraction: {time.time() - t2:.3f}s")

    # === Rotation
    t3 = time.time()
    fg_pc = o3d.geometry.PointCloud()
    clusters_list = []  # Store (cluster_points, label) pairs

    fg_pc.points = o3d.utility.Vector3dVector(fg_points)
    fg_pc.rotate(ROTATION_MATRIX, center=(0, 0, 0))

    # === Apply rotated ROI
    rot_pts = np.asarray(fg_pc.points)
    roi_mask = (
        (rot_pts[:, 0] >= x_min_r) & (rot_pts[:, 0] <= x_max_r) &
        (rot_pts[:, 1] >= y_min_r) & (rot_pts[:, 1] <= y_max_r) &
        (rot_pts[:, 2] >= z_min_r) & (rot_pts[:, 2] <= z_max_r)
    )
    fg_pc.points = o3d.utility.Vector3dVector(rot_pts[roi_mask])
    print(f"[Timing] Rotation: {time.time() - t3:.3f}s")

    # === GPU DBSCAN clustering
    t4 = time.time()
    clustering = cuDBSCAN(eps=dbscan_eps, min_samples=min_points)
    points_np = np.asarray(fg_pc.points)
    points_cp = cp.asarray(points_np, dtype=cp.float32)
    labels = clustering.fit_predict(points_cp).get()

    boxes_for_viz = []  # For visualization

    if labels.size > 0 and labels.max() >= 0:
        for i in range(labels.max() + 1):
            cluster_pts = np.asarray(fg_pc.points)[labels == i]
            if not (min_points <= len(cluster_pts) <= max_points):
                continue

            # Get axis-aligned bounding box for classification heuristic
            cluster_pc = o3d.geometry.PointCloud()
            cluster_pc.points = o3d.utility.Vector3dVector(cluster_pts)
            aa_bbox = cluster_pc.get_axis_aligned_bounding_box()
            extent = aa_bbox.get_extent()
            l, w, h = sorted(extent)[::-1]

            # === Heuristic Classification
            if l < 1.5 and w < 1.5 and h < 1.9:
                label = "person"
                color = (0, 1, 0)  # green
            elif l > 2.5 and w > 1.5 and h > 1.5:
                label = "vehicle"
                color = (1, 0, 0)  # red
            else:
                continue  # skip

            # Store cluster data for saving
            clusters_list.append((cluster_pts, label))
            
            # Create oriented bounding box for visualization
            oriented_bbox = create_oriented_bbox_for_visualization(cluster_pts)
            oriented_bbox.color = color
            boxes_for_viz.append((oriented_bbox, label))
            
    print(f"[Timing] Clustering + Oriented BBox: {time.time() - t4:.3f}s")

    # === NEW: Save detection results with orientation ===
    save_detections(frame_files[frame_idx], clusters_list, output_folder)

    # === Visualization
    t5 = time.time()
    scene.scene.clear_geometry()
    if len(fg_pc.points) > 0:
        scene.scene.add_geometry("foreground", fg_pc, pc_material)
    for idx, (bbox, label) in enumerate(boxes_for_viz):
        scene.scene.add_geometry(f"bbox_{idx}_{label}", bbox, bbox_material)
    print(f"[Timing] Visualization: {time.time() - t5:.3f}s")

    # total processing time 
    print(f"[Timing] Total processing time for each frame: {time.time() - t0:.3f}s\n")

    if frame_idx == 0 and len(fg_pc.points) > 0:
        bounds = fg_pc.get_axis_aligned_bounding_box()
        eye = bounds.get_center() + np.array([0, -30, 15])
        scene.setup_camera(60.0, bounds, eye.astype(np.float32))

    label_list = [label for _, label in boxes_for_viz]
    print(f"[{frame_files[frame_idx]}] Clusters: {len(boxes_for_viz)} →", label_list)

    # Schedule next frame
    frame_idx += 1
    gui.Application.instance.run_one_tick()
    time.sleep(frame_interval)
    gui.Application.instance.post_to_main_thread(window, update_scene)

# === Run GUI ===
gui.Application.instance.post_to_main_thread(window, update_scene)
gui.Application.instance.run()