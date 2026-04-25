import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
# Assuming your function is in this file/path
from Phase1.PostProcess import detect_rust_and_cracks

# --- CONFIGURATION ---
MODEL_PATH = '../backend/final_yolo_best_model.pt'
IMAGE_DIR = '../Phase3/Output/'
MASK_DIR = '../Phase3/masks/'

# Load Model
model = YOLO(MODEL_PATH)
os.makedirs(MASK_DIR, exist_ok=True)

image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Generating high-detail masks for {len(image_files)} images...")

for filename in tqdm(image_files):
    img_path = os.path.join(IMAGE_DIR, filename)
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    # 1. YOLO to find the general area
    results = model.predict(img_bgr, conf=0.4, verbose=False)

    # 2. Create a blank "Detection" canvas for this image
    full_image_mask = np.zeros((h, w), dtype=np.uint8)

    for r in results:
        if r.boxes is not None:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)

                # Filter out the rope as you did before
                if (x2 - x1) > (w * 0.8) and (y2 - y1) < (h * 0.1):
                    continue

                # 3. Use YOUR precise detection logic inside this box
                roi_mask = np.zeros((h, w), dtype=np.uint8)
                roi_mask[y1:y2, x1:x2] = 255

                # Get the detailed rust and crack masks from your Phase1 script
                rust_mask, cracks_mask = detect_rust_and_cracks(img_rgb, roi_mask)

                # Combine them into our final training mask
                full_image_mask = cv2.bitwise_or(full_image_mask, rust_mask)
                full_image_mask = cv2.bitwise_or(full_image_mask, cracks_mask)

    # 4. Save the high-detail mask
    cv2.imwrite(os.path.join(MASK_DIR, filename), full_image_mask)

print("Done! You now have a folder of high-detail detection masks.")