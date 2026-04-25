import cv2
import numpy as np
import json
import os
from ultralytics import YOLO
from Phase1.PostProcess import detect_rust_and_cracks

# --- CONFIG ---
MODEL_PATH = 'yolo_corrosion/yolov8_corrosion_542026/weights/best.pt'
IMAGE_DIR = "../Phase3/Output/"
OUTPUT_JSON = "yolo_detections_batch.json"
CONF_THRESHOLD = 0.4  # Slightly lower to catch more cracks

# Load Model
model = YOLO(MODEL_PATH)
all_detections = []

# Process all images in directory
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Processing {len(image_files)} images...")

for img_file in image_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    image_bgr = cv2.imread(img_path)
    if image_bgr is None: continue

    h_img, w_img = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 1. YOLO PREDICTION (To find Regions of Interest)
    results = model.predict(image_bgr, conf=CONF_THRESHOLD, imgsz=1024, verbose=False)

    for r in results:
        if r.boxes is None: continue

        for box in r.boxes:
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())

            # ROI Masking
            roi_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            roi_mask[y1:y2, x1:x2] = 255

            # 2. RUN INDEPENDENT PIXEL DETECTION
            # We get two distinct masks back
            rust_mask, cracks_mask = detect_rust_and_cracks(image_rgb, roi_mask)

            # --- PROCESS RUST SEPARATELY ---
            if np.any(rust_mask):
                contours, _ = cv2.findContours(rust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    rx, ry, rw, rh = cv2.boundingRect(cnt)
                    if rw < 5 or rh < 5: continue  # Filter noise

                    all_detections.append({
                        "filename": img_file,
                        "type": "rust",
                        "conf": conf,
                        "box_2d": [rx, ry, rx + rw, ry + rh]
                    })

            # --- PROCESS CRACKS SEPARATELY (The priority fix) ---
            if np.any(cracks_mask):
                contours, _ = cv2.findContours(cracks_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cx, cy, cw, ch = cv2.boundingRect(cnt)

                    # ROPE FILTER: Ignore long thin horizontal crack-like artifacts
                    if cw > (w_img * 0.4) and ch < 30: continue
                    if cw < 3 or ch < 3: continue  # Filter noise

                    # CRITICAL: We save this as a 'crack' type entirely separate from rust
                    all_detections.append({
                        "filename": img_file,
                        "type": "crack",
                        "conf": conf,
                        "box_2d": [cx, cy, cx + cw, cy + ch]
                    })

# Save for the 3D Mapper
with open(OUTPUT_JSON, 'w') as f:
    json.dump(all_detections, f, indent=4)

print(f"Success! Saved {len(all_detections)} independent detections to {OUTPUT_JSON}")