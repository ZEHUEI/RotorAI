from sympy.printing.pretty.pretty_symbology import annotated
from ultralytics import YOLO
import cv2

def main():
    # test model
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error line 11: Cannot Open Camera")
        return
    while True:
        ret,frame = cap.read()
        if not ret:
            print("Error line 16: Failed to read frame")
            break


        results = model(frame)

        # this is boxes
        annotated_frame = results[0].plot()

        #camera settings
        scale = 0.3
        height,width = annotated_frame.shape[:2]
        annotated_frame = cv2.resize(annotated_frame, (int(width * scale), int(height * scale)))

        cv2.imshow("YOLO Camera Test",annotated_frame)

        #key binddings
        if cv2.waitKey(1) & 0xFF == ord('f'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()