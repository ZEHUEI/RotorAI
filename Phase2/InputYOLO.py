from ultralytics import YOLO

model = YOLO('yolo_corrosion/yolov8_corrosion/weights/best.pt')
results = model.predict(source="Outcomes/Input/motor3.jpg", save=True,conf=0.25,project="Outcomes",name="Predictions")
print("save here",results[0].save_dir)
