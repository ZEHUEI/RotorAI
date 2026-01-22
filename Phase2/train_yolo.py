from ultralytics import YOLO
import cv2
import numpy as np
from Phase1.PostProcess import detect_rust_and_cracks

def main():
    # Load YOLO model
    model = YOLO("../yolo_corrosion/yolov8_corrosionV2/weights/best.pt")
    print(model.device)

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
        small_frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))

        # YOLO prediction
        results = model.predict(small_frame, conf=0.05, imgsz=1024)
        r = results[0]

        boxed_frame = frame.copy()

        # Only process if boxes are detected
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)

                # Create a mask for post-processing
                box_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                box_mask[y1:y2, x1:x2] = 255

                rust_mask, cracks_mask = detect_rust_and_cracks(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), box_mask
                )

                # Draw Rust boxes
                if np.any(rust_mask):
                    rx, ry, rw, rh = cv2.boundingRect(rust_mask)
                    cv2.rectangle(boxed_frame, (rx, ry), (rx+rw, ry+rh), (0,255,0), 2)
                    cv2.putText(boxed_frame, "Rust", (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                # Draw Crack boxes
                if np.any(cracks_mask):
                    contours, _ = cv2.findContours(cracks_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        cx, cy, cw, ch = cv2.boundingRect(cnt)
                        cv2.rectangle(boxed_frame, (cx, cy), (cx+cw, cy+ch), (255,255,0), 2)
                        cv2.putText(boxed_frame, "Crack", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        # Display frame smaller
        scale = 0.3
        display_frame = cv2.resize(boxed_frame, (int(boxed_frame.shape[1]*scale), int(boxed_frame.shape[0]*scale)))
        cv2.imshow("YOLO Camera Test", display_frame)

        # Press 'f' to exit
        if cv2.waitKey(1) & 0xFF == ord('f'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
