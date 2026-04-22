"""
This is final
"""
import cv2
from ultralytics import YOLO
from Phase1.PostProcess import detect_rust_and_cracks
import numpy as np

# 1. SETUP (Global)
model = YOLO('Phase2/yolo_corrosion/yolov8_corrosion_542026/weights/best.pt')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Paths
video_path = "Outcomes/Input/IMG_1291.mp4"
output_path = "Outcomes/Predictions/final.mp4"

cap = cv2.VideoCapture(video_path)

# --- VIDEO WRITER SETUP ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Opened:", cap.isOpened())
print("Resolution:", width, "x", height)

# Persistent variables
frame_idx = 0
last_results = None
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

# def is_too_blurry(img_gray, threshold=80):
#     return cv2.Laplacian(img_gray, cv2.CV_64F).var() < threshold

def process_roi(roi_mask, img_rgb, current_boxed_img, current_face_mask, w_img):
    final_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_not(current_face_mask))
    rust_mask, cracks_mask = detect_rust_and_cracks(img_rgb, final_mask)

    if np.any(rust_mask):
        contours, _ = cv2.findContours(rust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            if rw > (w_img * 0.5) and (rw / float(rh + 1e-5)) > 5:
                continue
            cv2.rectangle(current_boxed_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(current_boxed_img, "Rust", (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if np.any(cracks_mask):
        contours, _ = cv2.findContours(cracks_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            if cw > (w_img * 0.5):
                continue
            cv2.rectangle(current_boxed_img, (cx, cy), (cx + cw, cy + ch), (255, 255, 0), 2)
            cv2.putText(current_boxed_img, "Crack", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


# 3. MAIN VIDEO LOOP
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_bgr = frame
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    boxed_image = image_bgr.copy()
    h_img, w_img = image_rgb.shape[:2]

    # --- FACE MASKING ---
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    face_mask = np.zeros((h_img, w_img), dtype=np.uint8)

    # if is_too_blurry(gray):
    #     out.write(image_bgr)
    #     frame_idx += 1
    #     continue

    for (x, y, w, h) in faces:
        cv2.rectangle(face_mask, (x, y), (x + w, y + h), 255, -1)
    face_mask_dilated = cv2.dilate(face_mask, kernel, iterations=3)

    # --- YOLO PREDICTION (Every 2nd frame) ---
    if frame_idx % 5 == 0:
        # Predict on full resolution — no resize, no scaling needed
        last_results = model.predict(image_bgr, conf=0.4, imgsz=640, device="cpu", iou=0.4, verbose=False)

        # debug_results = model.predict(image_bgr, conf=0.15, imgsz=640, device="cpu", verbose=False)
        # annotated_frame = debug_results[0].plot()
        # cv2.imshow("DEBUG: WHAT YOLO SEES", cv2.resize(annotated_frame, (760, 360)))

    found_valid_box = False
    if last_results is not None and len(last_results[0].boxes) > 0:
        r = last_results[0]
        for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)

            # No scaling — coords are already in full-res space
            box_w = x2 - x1
            box_h = y2 - y1
            if box_w > (w_img * 0.8) and box_h < (h_img * 0.1):
                print(f"Skipping Box {i} (Likely Rope)")
                continue

            found_valid_box = True
            box_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            box_mask[y1:y2, x1:x2] = 255
            process_roi(box_mask, image_rgb, boxed_image, face_mask_dilated, w_img)

    # --- SAVE AND DISPLAY ---
    out.write(boxed_image)
    cv2.imshow("Rotor AI - Processing Video", cv2.resize(boxed_image, (w_img, h_img)))
    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video processing finished. Saved to: {output_path}")