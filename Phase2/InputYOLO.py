"""
This is final
"""
import cv2
from ultralytics import YOLO
from Phase1.PostProcess import detect_rust_and_cracks
import numpy as np

# Load YOLO model
model = YOLO('yolo_corrosion/yolov8_corrosion_542026/weights/best.pt')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load image
image_bgr = cv2.imread("../Phase3/Output/IMG_1294.JPG")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
boxed_image = image_bgr.copy()
h_img, w_img = image_rgb.shape[:2]  # Get image dimensions

# Face masking (Keep this, it's good safety)
gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
face_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
for (x, y, w, h) in faces:
    cv2.rectangle(face_mask, (x, y), (x + w, y + h), 255, -1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
face_mask_dilated = cv2.dilate(face_mask, kernel, iterations=3)


def process_roi(roi_mask, img_rgb, debug_name=""):
    final_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_not(face_mask_dilated))

    rust_mask, cracks_mask = detect_rust_and_cracks(img_rgb, final_mask)

    # --- RUST DRAWING ---
    if np.any(rust_mask):
        contours, _ = cv2.findContours(rust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            rx, ry, rw, rh = cv2.boundingRect(cnt)

            # FILTER: Ignore "Rust" that is actually the rope
            aspect_ratio = rw / float(rh)
            if rw > (w_img * 0.5) and aspect_ratio > 5:
                continue  # Skip the rope

            cv2.rectangle(boxed_image, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(boxed_image, "Rust", (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- CRACK DRAWING ---
    if np.any(cracks_mask):
        contours, _ = cv2.findContours(cracks_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)

            # FILTER: Ignore "Cracks" that are actually the rope
            if cw > (w_img * 0.5):
                continue  # Skip long horizontal lines (rope)

            cv2.rectangle(boxed_image, (cx, cy), (cx + cw, cy + ch), (255, 255, 0), 2)
            cv2.putText(boxed_image, "Crack", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


# 3. YOLO PREDICTION
results = model.predict(image_bgr, conf=0.4, imgsz=1024,device="cpu")  # Increased conf slightly
r = results[0]
found_valid_box = False

if r.boxes is not None and len(r.boxes) > 0:
    print(f"YOLO found {len(r.boxes)} boxes. Processing areas...")
    for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)

        # Check if box is just the rope (very wide, very short)
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w > (w_img * 0.8) and box_h < (h_img * 0.1):
            print(f"Skipping Box {i} (Likely Rope)")
            continue

        found_valid_box = True
        box_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        box_mask[y1:y2, x1:x2] = 255
        process_roi(box_mask, image_rgb, f"Box {i}")

cv2.imwrite("../Outcomes/Predictions/testingYOLOframes.png", boxed_image)
print("Saved fixed image.")