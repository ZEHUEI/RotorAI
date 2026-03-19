import torch
from ultralytics import YOLO
import cv2
import numpy as np
from Phase1.PostProcess import detect_rust_and_cracks

model = YOLO("../backend/yolo_corrosion/yolov8_corrosionV2/weights/best.pt")
model.model.to("cpu")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

cap = cv2.VideoCapture(0)

frame_idx = 0
last_results = None

#helper--------------------------------------------------
def process_roi(roi_mask, img_rgb, current_boxed_img, current_face_mask, w_img):
    final_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_not(current_face_mask))
    rust_mask, cracks_mask = detect_rust_and_cracks(img_rgb, final_mask)
    detections = []
    if np.any(rust_mask):
        contours, _ = cv2.findContours(rust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            if rw > (w_img * 0.5) and (rw / float(rh + 1e-5)) > 5:
                continue
            cv2.rectangle(current_boxed_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(current_boxed_img, "Rust", (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            detections.append({"x":rx,"y":ry,"w":rw,"h":rh,"label":"Rust"})
    if np.any(cracks_mask):
        contours, _ = cv2.findContours(cracks_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            if cw > (w_img * 0.5):
                continue
            cv2.rectangle(current_boxed_img, (cx, cy), (cx + cw, cy + ch), (255, 255, 0), 2)
            cv2.putText(current_boxed_img, "Crack", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            detections.append({"x":cx,"y":cy,"w":cw,"h":ch,"label":"Cracks",})
    return detections

def detect_frame(frame):
    h_img, w_img = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxed_image = frame.copy()

    # Face mask
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    face_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for (x, y, w, h) in faces:
        cv2.rectangle(face_mask, (x, y), (x + w, y + h), 255, -1)
    face_mask_dilated = cv2.dilate(face_mask, kernel, iterations=3)

    # YOLO detection
    results = model.predict(frame, conf=0.4, imgsz=640, device="cpu", verbose=False)[0]

    detections = []
    found_valid_box = False

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            box_w, box_h = x2 - x1, y2 - y1

            if box_w > (w_img * 0.8) and box_h < (h_img * 0.1):
                continue

            roi_face = face_mask_dilated[y1:y2, x1:x2]
            if np.any(roi_face):
                continue

            found_valid_box = True
            box_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            box_mask[y1:y2, x1:x2] = 255
            detections += process_roi(box_mask, img_rgb, boxed_image, face_mask_dilated, w_img)

    # Fallback to center
    if not found_valid_box:
        global_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        mx, my = int(w_img * 0.2), int(h_img * 0.2)
        global_mask[my:h_img - my, mx:w_img - mx] = 255
        detections += process_roi(global_mask, img_rgb, boxed_image, face_mask_dilated, w_img)

    return detections