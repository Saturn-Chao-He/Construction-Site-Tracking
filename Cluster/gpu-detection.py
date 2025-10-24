import os
import time
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from cuml.cluster import DBSCAN as cuDBSCAN
from cuml.neighbors import NearestNeighbors
import cupy as cp

# === Adjustable Rotation Angles (in degrees) ===
PITCH_DEG = 23
ROLL_DEG = 2
YAW_DEG = 2

# === Parameters ===
input_folder = "/media/ie/disk2/Lidar_Data/Construction_Data_Q_Building/7-23-2025/test/lidar/bin/1"
background_file = "/media/ie/disk2/Lidar_Data/Construction_Data_Q_Building/7-23-2025/test/lidar/bin/2/005100.bin"

voxel_size = 0.4
dbscan_eps = 1.0
min_points = 5
max_points = 400
frame_interval = 0.1

x_min, x_max = 15, 70
y_min, y_max = -30, 30
z_min, z_max = -10, 10

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
window = gui.Application.instance.create_window("Tracking with Background Subtraction", 1280, 720)
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
    print(f"[Timing] Background subtraction: {time.time() - t2:.3f}s")

    # === Rotation
    t3 = time.time()
    fg_pc = o3d.geometry.PointCloud()
    boxes = []

    if len(fg_points) > 0:
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
        # labels = clustering.fit_predict(cp.asarray(fg_pc)).get()
        points_np = np.asarray(fg_pc.points)
        points_cp = cp.asarray(points_np, dtype=cp.float32)
        labels = clustering.fit_predict(points_cp).get()

        
        if labels.size > 0 and labels.max() >= 0:
            for i in range(labels.max() + 1):
                cluster_pts = np.asarray(fg_pc.points)[labels == i]
                if not (min_points <= len(cluster_pts) <= max_points):
                    continue

                cluster_pc = o3d.geometry.PointCloud()
                cluster_pc.points = o3d.utility.Vector3dVector(cluster_pts)
                bbox = cluster_pc.get_axis_aligned_bounding_box()

                extent = bbox.get_extent()
                l, w, h = sorted(extent)[::-1]

                # === Heuristic Classification
                if l < 1.5 and w < 1.5 and h < 1.9:
                    label = "person"
                    bbox.color = (0, 1, 0)  # green
                elif l > 2.5 and w > 1.5 and h > 1.5:
                    label = "vehicle"
                    bbox.color = (1, 0, 0)  # red
                else:
                    continue  # skip

                boxes.append((bbox, label))
        print(f"[Timing] Clustering + BBox: {time.time() - t4:.3f}s")

    # === Visualization
    t5 = time.time()
    scene.scene.clear_geometry()
    if len(fg_pc.points) > 0:
        scene.scene.add_geometry("foreground", fg_pc, pc_material)
    for idx, (bbox, label) in enumerate(boxes):
        scene.scene.add_geometry(f"bbox_{idx}_{label}", bbox, bbox_material)
    print(f"[Timing] Visualization: {time.time() - t5:.3f}s")

    # total processing time 
    print(f"[Timing] Total processing time for each frame: {time.time() - t0:.3f}s\n")

    if frame_idx == 0 and len(fg_pc.points) > 0:
        bounds = fg_pc.get_axis_aligned_bounding_box()
        eye = bounds.get_center() + np.array([0, -30, 15])
        scene.setup_camera(60.0, bounds, eye.astype(np.float32))

    label_list = [label for _, label in boxes]
    print(f"[{frame_files[frame_idx]}] Clusters: {len(boxes)} →", label_list)

    # Schedule next frame
    frame_idx += 1
    gui.Application.instance.run_one_tick()
    time.sleep(frame_interval)
    gui.Application.instance.post_to_main_thread(window, update_scene)

# === Run GUI ===
gui.Application.instance.post_to_main_thread(window, update_scene)
gui.Application.instance.run()
