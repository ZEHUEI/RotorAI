import os
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# -----------------------------
# GPU info
# -----------------------------
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.current_device())
    print("CUDA name:", torch.cuda.get_device_name(0))

# -----------------------------
# Paths & Settings
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_img_dir = os.path.join(BASE_DIR, "../Data/trainYOLO/images")
val_img_dir   = os.path.join(BASE_DIR, "../Data/validYOLO/images")

train_json    = os.path.join(BASE_DIR, "../Data/trainYOLO/_annotations.coco.json")
val_json      = os.path.join(BASE_DIR, "../Data/validYOLO/_annotations.coco.json")

train_labels_dir = os.path.join(BASE_DIR, "../Data/trainYOLO/labels")
val_labels_dir   = os.path.join(BASE_DIR, "../Data/validYOLO/labels")

target_labels = ["corrosion"]
batch_size = 8
img_size = 768
epochs = 300

# -----------------------------
# Check that images exist
# -----------------------------
def check_folder(folder):
    imgs = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.png'))]
    if not imgs:
        raise RuntimeError(f"No images found in {folder}")
    print(f"{len(imgs)} images found in {folder}")
    return imgs

train_images = check_folder(train_img_dir)
val_images = check_folder(val_img_dir)

# -----------------------------
# Convert COCO JSON -> YOLO txt
# -----------------------------
def coco_bbox_to_yolo(coco_json, yolo_label_dir, target_labels):
    os.makedirs(yolo_label_dir, exist_ok=True)

    with open(coco_json) as f:
        data = json.load(f)

    # Safer mapping (important)
    cat_name_to_id = {name: i for i, name in enumerate(target_labels)}

    img_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    img_id_to_info = {img['id']: img for img in data['images']}

    for ann in data['annotations']:
        # Get category name
        cat = next(c for c in data['categories'] if c['id'] == ann['category_id'])
        cat_name = cat['name']

        if cat_name not in target_labels:
            continue

        img_file = img_id_to_filename[ann['image_id']]
        txt_file = os.path.join(yolo_label_dir, os.path.splitext(img_file)[0] + '.txt')

        img_info = img_id_to_info[ann['image_id']]
        W, H = img_info['width'], img_info['height']

        # COCO bbox: [x, y, width, height]
        x, y, w, h = ann['bbox']

        # Convert to YOLO format
        x_center = (x + w / 2) / W
        y_center = (y + h / 2) / H
        w /= W
        h /= H

        with open(txt_file, 'a') as f:
            f.write(f"{cat_name_to_id[cat_name]} {x_center} {y_center} {w} {h}\n")

def coco_seg_to_yolo(coco_json, img_dir, yolo_label_dir, target_labels):
    os.makedirs(yolo_label_dir, exist_ok=True)
    with open(coco_json) as f:
        data = json.load(f)

    cat_name_to_id = {cat['name']: i for i, cat in enumerate(data['categories']) if cat['name'] in target_labels}
    img_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    img_id_to_info = {img['id']: img for img in data['images']}

    for ann in data['annotations']:
        cat_name = [cat['name'] for cat in data['categories'] if cat['id']==ann['category_id']][0]
        if cat_name not in target_labels:
            continue

        if "segmentation" not in ann:
            continue

        img_file = img_id_to_filename[ann['image_id']]
        txt_file = os.path.join(yolo_label_dir, os.path.splitext(img_file)[0] + '.txt')

        img_info = img_id_to_info[ann['image_id']]
        W, H = img_info['width'], img_info['height']

        for seg in ann['segmentation']:
            pts = np.array(seg,dtype=np.float32).reshape(-1, 2)
            pts[:, 0] /= W
            pts[:, 1] /= H

            flat = pts.flatten()
            with open(txt_file, 'a') as f:
                f.write(str(cat_name_to_id[cat_name]) + " " + " ".join(map(str, flat)) + "\n")

#change this func
# coco_seg_to_yolo(train_json, train_img_dir, train_labels_dir, target_labels)
# coco_seg_to_yolo(val_json, val_img_dir, val_labels_dir, target_labels)
coco_bbox_to_yolo(train_json, train_labels_dir, target_labels)
coco_bbox_to_yolo(val_json, val_labels_dir, target_labels)
# -----------------------------
# Check labels exist
# -----------------------------
def check_labels(folder):
    labels = [f for f in os.listdir(folder) if f.endswith(".txt")]
    if not labels:
        raise RuntimeError(f"No label txt files found in {folder}")
    print(f"{len(labels)} label files found in {folder}")
    return labels

check_labels(train_labels_dir)
check_labels(val_labels_dir)

# -----------------------------
# Create YOLO dataset YAML
# -----------------------------
yolo_dataset_yaml = f"""
train: {os.path.join(BASE_DIR, '../Data/trainYOLO/images')}
val: {os.path.join(BASE_DIR, '../Data/validYOLO/images')}

nc: 1
names: ['corrosion']
"""


yaml_path = os.path.join(BASE_DIR, "yolo_dataset.yaml")
with open(yaml_path, "w") as f:
    f.write(yolo_dataset_yaml)
print(f"YOLO dataset YAML created at {yaml_path}")

# -----------------------------
# Train YOLOv8
# -----------------------------
# YOLO('yolov8s.pt')
model = YOLO('yolov8m.pt')

print("Starting YOLO training...")
model.train(
    data=yaml_path,
    epochs=epochs,
    batch=batch_size,
    imgsz=img_size,
    device=0,
    #leartning rate
    lr0=3e-4,
    lrf=0.01,
    warmup_epochs=5,
    cos_lr=True,
    weight_decay=5e-4,

    mosaic=1.0,
    mixup=0.15,
    degrees=15.0,
    shear=5.0,
    perspective=0.0005,
    fliplr=0.5,
    flipud=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    scale=0.5,

    dropout=0.1,
    label_smoothing=0.05,

    patience=50,
    save_period=10,
    cache='disk',
    workers=8,

    project='yolo_corrosion',
    name='yolov8_corrosion_542026'
)

metrics = model.val()
print(metrics.box.map)
print(metrics.box.map50)
