import torch
from ultralytics import YOLO
import cv2
import numpy as np
from Phase1.PostProcess import detect_rust_and_cracks


#helper
def process_roi(roi_mask,img_rgb,boxed_frame,face_mask_dilated,w_img,h_img):
    final_mask = cv2.bitwise_and(roi_mask,cv2.bitwise_not(face_mask_dilated))
    rust_mask, cracks_mask = detect_rust_and_cracks(img_rgb,final_mask)

    if np.any(rust_mask):
        contours, _ = cv2.findContours(rust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            rx, ry, rw, rh = cv2.boundingRect(cnt)

            # FILTER: Ignore "Rust" that is actually the rope (too wide + too thin)
            aspect_ratio = rw / float(rh + 1e-5)
            # If wider than 50% of screen AND very thin (high aspect ratio)
            if rw > (w_img * 0.5) and aspect_ratio > 5:
                continue

            if np.any(cracks_mask):
                contours, _ = cv2.findContours(cracks_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cx, cy, cw, ch = cv2.boundingRect(cnt)

                    # FILTER: Ignore "Cracks" that are actually the rope (too wide)
                    if cw > (w_img * 0.5):
                        continue

                    cv2.rectangle(boxed_frame, (cx, cy), (cx + cw, cy + ch), (255, 255, 0), 2)
                    cv2.putText(boxed_frame, "Crack", (cx, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.rectangle(boxed_frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(boxed_frame, "Rust", (rx, ry - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def main():
    # Load YOLO model
    model = YOLO("../yolo_corrosion/yolov8_corrosionV2/weights/best.pt")
    device = torch.device("cpu")
    model.model.to(device)
    print(model.device)

    #face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize frame for speed
        h_img, w_img = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        boxed_frame = frame.copy()

        small_frame = cv2.resize(frame, (640, int(640 * h_img / w_img)))

        #detect face in cam frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        face_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for (x, y, w, h) in faces:
            cv2.rectangle(face_mask, (x, y), (x + w, y + h), 255, -1)

        #dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        face_mask_dilated = cv2.dilate(face_mask, kernel, iterations=3)

        # YOLO prediction
        results = model.predict(small_frame, conf=0.05, imgsz=640,device="cpu")
        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:
            scale_x = w_img / small_frame.shape[1]
            scale_y = h_img / small_frame.shape[0]

            for box in r.boxes.xyxy.cpu().numpy():
                # Rescale box coordinates to original frame
                x1, y1, x2, y2 = map(int, [
                    box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y
                ])

                box_w = x2 - x1
                box_h = y2 - y1
                if box_w > (w_img * 0.8) and box_h < (h_img * 0.1):
                    # Skip this box, it's just the wire/rope
                    continue

                # Skip boxes overlapping faces
                roi = face_mask_dilated[y1:y2, x1:x2]
                if np.any(roi):
                    continue

                found_valid_box = False

                if r.boxes is not None and len(r.boxes) > 0:
                    scale_x = w_img / small_frame.shape[1]
                    scale_y = h_img / small_frame.shape[0]

                    for box in r.boxes.xyxy.cpu().numpy():
                        # Rescale box coordinates back to original 1080p/720p frame
                        x1, y1, x2, y2 = map(int, [
                            box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y
                        ])

                        # Check for "Rope-like" YOLO boxes (entire width of screen)
                        box_w = x2 - x1
                        box_h = y2 - y1
                        if box_w > (w_img * 0.8) and box_h < (h_img * 0.1):
                            # Skip this box, it's just the wire/rope
                            continue

                        # Check if box overlaps a face (Safety)
                        roi_face = face_mask_dilated[y1:y2, x1:x2]
                        if np.any(roi_face):
                            continue

                        found_valid_box = True

                        # Create mask for this specific box
                        box_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                        box_mask[y1:y2, x1:x2] = 255

                        # Run detection on this box
                        process_roi(box_mask, image_rgb, boxed_frame, face_mask_dilated, w_img, h_img)

                if not found_valid_box:
                    # Create a mask for the center 60% of the screen
                    global_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                    margin_x = int(w_img * 0.2)
                    margin_y = int(h_img * 0.2)
                    global_mask[margin_y:h_img - margin_y, margin_x:w_img - margin_x] = 255

                    # Run detection on the center area
                    process_roi(global_mask, image_rgb, boxed_frame, face_mask_dilated, w_img, h_img)

                    # --- E. DISPLAY ---
                    # Resize for display convenience
                scale = 0.5
                display_frame = cv2.resize(boxed_frame, (int(w_img * scale), int(h_img * scale)))
                cv2.imshow("Rotor AI - Live", display_frame)

                # Press 'f' to exit
                if cv2.waitKey(1) & 0xFF == ord('f'):
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
