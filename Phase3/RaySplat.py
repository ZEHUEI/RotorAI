import json
import numpy as np
import os
from sklearn.cluster import DBSCAN

"""
use gamma0.5, alpha =0.1
"""
#new rasy splat  built
# --- CONFIGURATION ---
# TRANSFORMS_PATH = "nerfstudio/data/final/transforms.json"
# YOLO_JSON_PATH = "yolo_detections_batch.json"
# OUTPUT_PATH = "clustered_detections_new.json"
#
# # MOTOR GEOMETRY (Adjust these to fit your specific Splat)
# MOTOR_CENTER = np.array([0.45, 0.2, -0.35])
# # Radius is the distance from the center to the surface of the motor.
# # If your motor is roughly 40cm wide, use 0.2.
# MOTOR_RADIUS = 0.22
# CRACK_DEPTH_BIAS = 0.03
#
# def run_mapping_pipeline():
#     if not os.path.exists(TRANSFORMS_PATH):
#         print(f"Error: {TRANSFORMS_PATH} not found.")
#         return
#
#     with open(TRANSFORMS_PATH, 'r') as f:
#         config = json.load(f)
#     with open(YOLO_JSON_PATH, 'r') as f:
#         detections = json.load(f)
#
#     scale_factor = config.get('scale_factor', 1.0)
#     frame_list = config.get('frames', config.get('images', []))
#     frames = {os.path.basename(f.get('file_path', f.get('path', ''))): f for f in frame_list}
#
#     rust_points = []
#     crack_points = []
#
#     # --- PHASE 1: PROJECTION & TYPE SPLITTING ---
#     for det in detections:
#         img_name = det['filename']
#         if img_name not in frames: continue
#
#         c2w = np.array(frames[img_name]['transform_matrix'])
#         cam_origin = c2w[:3, 3]
#
#         vec_to_center = MOTOR_CENTER - cam_origin
#         dist_to_center = np.linalg.norm(vec_to_center)
#         direction_to_center = vec_to_center / dist_to_center
#
#         # APPLY SPECIFIC RADIUS FOR CRACKS
#         is_crack = "rust" not in det['type'].lower()
#         # Subtraction here makes the "surface" smaller, pulling points IN
#         effective_radius = MOTOR_RADIUS - CRACK_DEPTH_BIAS if is_crack else MOTOR_RADIUS
#
#         snap_dist = dist_to_center - effective_radius
#         world_point = (cam_origin + (direction_to_center * snap_dist)) * scale_factor
#
#         point_data = {"pos": world_point, "type": det['type'].lower(), "conf": det['conf']}
#
#         if is_crack:
#             crack_points.append(point_data)
#         else:
#             rust_points.append(point_data)
#
#     final_objects = []
#
#     # --- PHASE 2: CLUSTER RUST (MERGE GREEN) ---
#     if rust_points:
#         coords = np.array([p['pos'] for p in rust_points])
#         # eps=0.1 (10cm). Increase this if green boxes aren't merging enough.
#         clustering = DBSCAN(eps=0.12, min_samples=2).fit(coords)
#
#         for label in set(clustering.labels_):
#             if label == -1: continue
#             indices = [i for i, l in enumerate(clustering.labels_) if l == label]
#             cluster_coords = coords[indices]
#
#             center = np.mean(cluster_coords, axis=0)
#             size = np.max(cluster_coords, axis=0) - np.min(cluster_coords, axis=0)
#
#             final_objects.append({
#                 "type": "rust",
#                 "position": center.tolist(),
#                 "size": size.tolist(),
#             })
#
#     # --- PHASE 3: SOLO CRACKS (CYAN BOXES) ---
#     for i, crack in enumerate(crack_points):
#         # We don't cluster these; each one gets its own box
#         final_objects.append({
#             "type": "crack",
#             "position": crack['pos'].tolist(),
#             "size": [0.03, 0.03, 0.03],  # Small, sharp boxes for cracks
#             "count": 1
#         })
#
#     with open(OUTPUT_PATH, 'w') as f:
#         json.dump(final_objects, f, indent=4)
#
#     print(f"Done! Created {len(final_objects)} total objects.")
#
#
# if __name__ == "__main__":
#     run_mapping_pipeline()


#main
# --- CONFIGURATION ---
TRANSFORMS_PATH = "nerfstudio/data/final/transforms.json"
YOLO_JSON_PATH = "yolo_detections_batch.json"
OUTPUT_PATH = "clustered_detections_new.json"

# MOTOR GEOMETRY (Adjust these to fit your specific Splat)
MOTOR_CENTER = np.array([0.45, 0.2, -0.35])
# Radius is the distance from the center to the surface of the motor.
# If your motor is roughly 40cm wide, use 0.2.
MOTOR_RADIUS = 0.22
CRACK_DEPTH_BIAS = 0.03

def run_mapping_pipeline():
    if not os.path.exists(TRANSFORMS_PATH):
        print(f"Error: {TRANSFORMS_PATH} not found.")
        return

    with open(TRANSFORMS_PATH, 'r') as f:
        config = json.load(f)
    with open(YOLO_JSON_PATH, 'r') as f:
        detections = json.load(f)

    #scale down
    scale_factor = config.get('scale_factor', 1.0)
    frame_list = config.get('frames', config.get('images', []))
    frames = {os.path.basename(f.get('file_path', f.get('path', ''))): f for f in frame_list}

    rust_points = []
    crack_points = []

    # --- PHASE 1: PROJECTION & TYPE SPLITTING ---
    for det in detections:
        img_name = det['filename']
        if img_name not in frames: continue

        c2w = np.array(frames[img_name]['transform_matrix'])
        cam_origin = c2w[:3, 3]

        vec_to_center = MOTOR_CENTER - cam_origin
        dist_to_center = np.linalg.norm(vec_to_center)
        direction_to_center = vec_to_center / dist_to_center

        # APPLY SPECIFIC RADIUS FOR CRACKS
        is_crack = "rust" not in det['type'].lower()
        # Subtraction here makes the "surface" smaller, pulling points IN
        effective_radius = MOTOR_RADIUS - CRACK_DEPTH_BIAS if is_crack else MOTOR_RADIUS

        snap_dist = dist_to_center - effective_radius
        world_point = (cam_origin + (direction_to_center * snap_dist)) * scale_factor

        point_data = {"pos": world_point, "type": det['type'].lower(), "conf": det['conf']}

        if is_crack:
            crack_points.append(point_data)
        else:
            rust_points.append(point_data)

    final_objects = []

    # --- PHASE 2: CLUSTER RUST (MERGE GREEN) ---
    if rust_points:
        coords = np.array([p['pos'] for p in rust_points])
        # eps=0.1 (10cm). Increase this if green boxes aren't merging enough. lower to 0.7
        clustering = DBSCAN(eps=0.12, min_samples=2).fit(coords)

        for label in set(clustering.labels_):
            if label == -1: continue
            indices = [i for i, l in enumerate(clustering.labels_) if l == label]
            cluster_coords = coords[indices]

            center = np.mean(cluster_coords, axis=0)
            size = np.max(cluster_coords, axis=0) - np.min(cluster_coords, axis=0)

            final_objects.append({
                "type": "rust",
                "position": center.tolist(),
                "size": size.tolist(),
            })

    # --- PHASE 3: SOLO CRACKS (CYAN BOXES) ---
    for i, crack in enumerate(crack_points):
        # We don't cluster these; each one gets its own box
        final_objects.append({
            "type": "crack",
            "position": crack['pos'].tolist(),
            "size": [0.03, 0.03, 0.03],  # Small, sharp boxes for cracks
            "count": 1
        })

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(final_objects, f, indent=4)

    print(f"Done! Created {len(final_objects)} total objects.")


if __name__ == "__main__":
    run_mapping_pipeline()


# import json
# import numpy as np
# import os
# from sklearn.cluster import DBSCAN
#
# # --- CONFIGURATION ---
# TRANSFORMS_PATH = "nerfstudio/data/final/transforms.json"
# YOLO_JSON_PATH = "yolo_detections_batch.json"
# OUTPUT_PATH = "clustered_detections_two_tier.json"
#
# # MOTOR GEOMETRY
# MOTOR_CENTER_XZ = np.array([0.45, -0.35])
# MOTOR_RADIUS = 0.22
# # LEVEL 1: The Top Tier (Ribs/Lid)
# MOTOR_TOP_Y = 0.31
# # LEVEL 2: The Side Tier (The Main Body)
# MOTOR_SIDE_Y = 0.15
#
# CRACK_DEPTH_BIAS = 0.03
#
#
# def run_mapping_pipeline():
#     if not os.path.exists(TRANSFORMS_PATH): return
#
#     with open(TRANSFORMS_PATH, 'r') as f:
#         config = json.load(f)
#     with open(YOLO_JSON_PATH, 'r') as f:
#         detections = json.load(f)
#
#     scale_factor = config.get('scale_factor', 1.0)
#     frame_list = config.get('frames', config.get('images', []))
#     frames = {os.path.basename(f.get('file_path', f.get('path', ''))): f for f in frame_list}
#
#     rust_points, crack_points = [], []
#
#     for det in detections:
#         img_name = det['filename']
#         if img_name not in frames: continue
#
#         c2w = np.array(frames[img_name]['transform_matrix'])
#         cam_origin = c2w[:3, 3]
#
#         # 1. HORIZONTAL SNAP (Forces a perfect circle)
#         vec_to_cam_xz = np.array([cam_origin[0] - MOTOR_CENTER_XZ[0], cam_origin[2] - MOTOR_CENTER_XZ[1]])
#         dist_xz = np.linalg.norm(vec_to_cam_xz)
#         dir_xz = vec_to_cam_xz / dist_xz
#
#         is_crack = "rust" not in det['type'].lower()
#         eff_radius = MOTOR_RADIUS - CRACK_DEPTH_BIAS if is_crack else MOTOR_RADIUS
#
#         target_x = MOTOR_CENTER_XZ[0] + (dir_xz[0] * eff_radius)
#         target_z = MOTOR_CENTER_XZ[1] + (dir_xz[1] * eff_radius)
#
#         # 2. THE TWO-TIER FIX (No more staircases)
#         # We classify every shot as either "Top View" or "Side View"
#         # and lock it to that specific height.
#         if cam_origin[1] > 0.26:
#             target_y = MOTOR_TOP_Y
#         else:
#             target_y = MOTOR_SIDE_Y
#
#         world_point = np.array([target_x, target_y, target_z]) * scale_factor
#         point_data = {"pos": world_point, "type": det['type'].lower()}
#
#         if is_crack:
#             crack_points.append(point_data)
#         else:
#             rust_points.append(point_data)
#
#     final_objects = []
#
#     # CLUSTER RUST (GREEN)
#     if rust_points:
#         coords = np.array([p['pos'] for p in rust_points])
#         clustering = DBSCAN(eps=0.08, min_samples=2).fit(coords)
#         for label in set(clustering.labels_):
#             if label == -1: continue
#             idx = [i for i, l in enumerate(clustering.labels_) if l == label]
#             c_coords = coords[idx]
#             final_objects.append({
#                 "type": "rust",
#                 "position": np.mean(c_coords, axis=0).tolist(),
#                 # Keep height slim (Y=0.02) to maintain the "ring" look
#                 "size": [(np.max(c_coords[:, 0]) - np.min(c_coords[:, 0]) + 0.05),
#                          0.02,
#                          (np.max(c_coords[:, 2]) - np.min(c_coords[:, 2]) + 0.05)]
#             })
#
#     # CRACKS (CYAN)
#     for crack in crack_points:
#         final_objects.append({
#             "type": "crack",
#             "position": crack['pos'].tolist(),
#             "size": [0.015, 0.015, 0.015]
#         })
#
#     with open(OUTPUT_PATH, 'w') as f:
#         json.dump(final_objects, f, indent=4)
#     print(f"Success! All detections locked to Y={MOTOR_TOP_Y} or Y={MOTOR_SIDE_Y}.")
#
#
# if __name__ == "__main__":
#     run_mapping_pipeline()

# import json
# import numpy as np
# import os
# from sklearn.cluster import DBSCAN
#
# # --- CONFIGURATION ---
# TRANSFORMS_PATH = "nerfstudio/data/final/transforms.json"
# YOLO_JSON_PATH = "yolo_detections_batch.json"
# OUTPUT_PATH = "clustered_detections_rectangle.json"
#
# # MOTOR RECTANGLE BOUNDS (X and Z floor plane)
# # Adjust these based on your Three.js measurements
# X_MIN, X_MAX = 0.25, 0.65
# Z_MIN, Z_MAX = -0.55, -0.15
#
# # VERTICAL TIERS (Locked Heights)
# MOTOR_TOP_Y = 0.31   # Upper tier (Ribs)
# MOTOR_SIDE_Y = 0.15  # Lower tier (Body)
#
# def run_mapping_pipeline():
#     if not os.path.exists(TRANSFORMS_PATH): return
#
#     with open(TRANSFORMS_PATH, 'r') as f:
#         config = json.load(f)
#     with open(YOLO_JSON_PATH, 'r') as f:
#         detections = json.load(f)
#
#     scale_factor = config.get('scale_factor', 1.0)
#     frame_list = config.get('frames', config.get('images', []))
#     frames = {os.path.basename(f.get('file_path', f.get('path', ''))): f for f in frame_list}
#
#     points = []
#
#     for det in detections:
#         img_name = det['filename']
#         if img_name not in frames: continue
#
#         c2w = np.array(frames[img_name]['transform_matrix'])
#         cam_origin = c2w[:3, 3]
#
#         # 1. RECTANGLE SNAP (X and Z)
#         # We take the camera position and 'clamp' it to the edges of the box
#         target_x = np.clip(cam_origin[0], X_MIN, X_MAX)
#         target_z = np.clip(cam_origin[2], Z_MIN, Z_MAX)
#
#         # 2. VERTICAL SNAP (The Staircase Killer)
#         # Binary height logic: Top or Side.
#         target_y = MOTOR_TOP_Y if cam_origin[1] > 0.26 else MOTOR_SIDE_Y
#
#         points.append({
#             "pos": np.array([target_x, target_y, target_z]) * scale_factor,
#             "type": "crack" if "rust" not in det['type'].lower() else "rust"
#         })
#
#     # --- CLUSTERING & EXPORT ---
#     final_objects = []
#     rust_coords = np.array([p['pos'] for p in points if p['type'] == "rust"])
#     crack_coords = np.array([p['pos'] for p in points if p['type'] == "crack"])
#
#     # Cluster Rust
#     if len(rust_coords) > 0:
#         clusters = DBSCAN(eps=0.08, min_samples=2).fit(rust_coords)
#         for label in set(clusters.labels_):
#             if label == -1: continue
#             c_p = rust_coords[clusters.labels_ == label]
#             final_objects.append({
#                 "type": "rust",
#                 "position": np.mean(c_p, axis=0).tolist(),
#                 "size": (np.max(c_p, axis=0) - np.min(c_p, axis=0) + 0.05).tolist()
#             })
#
#     # Individual Cracks
#     for coord in crack_coords:
#         final_objects.append({
#             "type": "crack", "position": coord.tolist(), "size": [0.015, 0.015, 0.015]
#         })
#
#     with open(OUTPUT_PATH, 'w') as f:
#         json.dump(final_objects, f, indent=4)
#     print(f"Rectangle mapping complete. Saved to {OUTPUT_PATH}.")
#
# if __name__ == "__main__":
#     run_mapping_pipeline()


# import json
# import numpy as np
# import os
#
# # --- CONFIGURATION ---
# TRANSFORMS_PATH = "nerfstudio/data/final/transforms.json"
# YOLO_JSON_PATH = "yolo_detections_batch.json"
# OUTPUT_PATH = "clustered_detections_fixed.json"
#
# # MOTOR GEOMETRY
# MOTOR_CENTER_XZ = np.array([0.45, -0.35])
# MOTOR_RADIUS = 0.22
#
# # THE TWO TIERS (Locking the Y-axis to see the top cracks)
# # Snap to 0.31 for the top lid/ribs, 0.15 for the side walls
# MOTOR_TOP_Y = 0.31
# MOTOR_SIDE_Y = 0.15
#
#
# def run_mapping_pipeline():
#     if not os.path.exists(TRANSFORMS_PATH):
#         print("Transforms file not found.")
#         return
#
#     with open(TRANSFORMS_PATH, 'r') as f:
#         config = json.load(f)
#     with open(YOLO_JSON_PATH, 'r') as f:
#         detections = json.load(f)
#
#     scale_factor = config.get('scale_factor', 1.0)
#
#     # FIX: Robust frame mapping to avoid KeyError: 'path'
#     frames = {}
#     for f in config.get('frames', []):
#         # Check all possible key names for the file path
#         raw_path = f.get('file_path') or f.get('path') or f.get('file_name')
#         if raw_path:
#             frames[os.path.basename(raw_path)] = f
#
#     final_objects = []
#
#     for det in detections:
#         img_name = det['filename']
#         if img_name not in frames: continue
#
#         c2w = np.array(frames[img_name]['transform_matrix'])
#         cam_origin = c2w[:3, 3]
#
#         # 1. HORIZONTAL CIRCLE (X, Z)
#         # Direction from motor center to camera on the floor plane
#         vec_to_cam_xz = np.array([cam_origin[0] - MOTOR_CENTER_XZ[0], cam_origin[2] - MOTOR_CENTER_XZ[1]])
#         dist = np.linalg.norm(vec_to_cam_xz)
#
#         # Avoid division by zero if camera is exactly at center
#         if dist == 0: continue
#         dir_xz = vec_to_cam_xz / dist
#
#         is_crack = "rust" not in det['type'].lower()
#         # Offset cracks slightly inward so they sit on the metal surface
#         eff_radius = MOTOR_RADIUS - 0.03 if is_crack else MOTOR_RADIUS
#
#         target_x = MOTOR_CENTER_XZ[0] + (dir_xz[0] * eff_radius)
#         target_z = MOTOR_CENTER_XZ[1] + (dir_xz[1] * eff_radius)
#
#         # 2. VERTICAL SNAP (The Staircase Killer)
#         # If camera Y is high (> 0.26), it's looking at the top.
#         # Snapping to MOTOR_TOP_Y prevents the boxes from 'sinking'.
#         target_y = MOTOR_TOP_Y if cam_origin[1] > 0.26 else MOTOR_SIDE_Y
#
#         final_objects.append({
#             "type": "crack" if is_crack else "rust",
#             "position": (np.array([target_x, target_y, target_z]) * scale_factor).tolist(),
#             "size": [0.02, 0.02, 0.02] if is_crack else [0.08, 0.04, 0.08]
#         })
#
#     with open(OUTPUT_PATH, 'w') as f:
#         json.dump(final_objects, f, indent=4)
#     print(f"Processed {len(final_objects)} detections. Output saved to {OUTPUT_PATH}")
#
#
# if __name__ == "__main__":
#     run_mapping_pipeline()