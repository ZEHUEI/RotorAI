import json
import numpy as np
import os

# 1. SETUP PATHS (Updated for your environment)
TRANSFORMS_PATH = "nerfstudio/data/final/transforms.json"
YOLO_JSON_PATH = "yolo_detections_batch.json"
OUTPUT_PATH = "final_3d_detections.json"


def project_detections():
    # Load Data
    if not os.path.exists(TRANSFORMS_PATH):
        print(f"Error: Could not find {TRANSFORMS_PATH}")
        return

    with open(TRANSFORMS_PATH, 'r') as f:
        config = json.load(f)

    with open(YOLO_JSON_PATH, 'r') as f:
        detections = json.load(f)

    # DEBUG: Let's see what is actually in your JSON keys
    print(f"JSON Keys found: {list(config.keys())}")

    # Handle different Nerfstudio JSON structures
    if 'frames' in config:
        frame_list = config['frames']
    elif 'images' in config: # Sometimes it uses 'images'
        frame_list = config['images']
    else:
        print("Error: Could not find a 'frames' or 'images' key in your JSON.")
        return

    # Nerfstudio scaling factors
    scale_factor = config.get('scale_factor', 1.0)

    # Create lookup for camera matrices
    frames = {}
    for f in frame_list:
        # Some versions use 'file_path', others just 'path'
        path_key = 'file_path' if 'file_path' in f else 'path'
        if path_key in f:
            frames[os.path.basename(f[path_key])] = f

    print(f"Loaded {len(frames)} camera transforms.")
    final_points = []

    print(f"Mapping {len(detections)} detections...")

    for det in detections:
        img_name = det['filename']
        if img_name not in frames:
            continue

        frame = frames[img_name]
        c2w = np.array(frame['transform_matrix'])  # 4x4 matrix

        # 1. Camera Origin (The eye)
        cam_origin = c2w[:3, 3]

        # 2. Camera Direction (The 'Forward' vector)
        # In Nerfstudio/Splatfacto, the camera looks down the -Z axis (index 2)
        cam_forward = -c2w[:3, 2]

        # 3. Depth Estimation
        # Since we don't have per-pixel depth, we project to a fixed distance
        # from the camera. Adjust 'dist' if points are too far/close.
        dist = 1.2  # 1.2 meters is usually safe for a motor inspection

        # Calculate 3D point
        world_point = cam_origin + (cam_forward * dist)

        # Apply Nerfstudio's global scale to match the .ply exactly
        world_point = world_point * scale_factor

        final_points.append({
            "id": len(final_points),
            "position": world_point.tolist(),
            "type": det['type'],
            "conf": det['conf'],
            "img": img_name
        })

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(final_points, f, indent=4)

    print(f"Success! {len(final_points)} points saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    project_detections()