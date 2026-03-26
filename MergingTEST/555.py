import os
import json
import random
import shutil

# 1. Setup Paths specifically for Dataset 5
# CHANGE THIS to the actual path of your human dataset folder
base_dataset_5 = "../Data/5"

train_dir = os.path.join(base_dataset_5, "train")
valid_dir = os.path.join(base_dataset_5, "valid")
train_json_path = os.path.join(train_dir, "_annotations.coco.json")
valid_json_path = os.path.join(valid_dir, "_annotations.coco.json")

# Create valid directory if it doesn't exist
os.makedirs(valid_dir, exist_ok=True)

# 2. Load the JSON from the Folder 5 Train directory
if not os.path.exists(train_json_path):
    print(f"Error: Could not find JSON at {train_json_path}")
    exit()

with open(train_json_path, 'r') as f:
    data = json.load(f)

# 3. Decide which images to move (20% split)
images = data['images']
random.seed(42) # Keeps the split consistent if you run it twice
random.shuffle(images)
num_valid = int(len(images) * 0.20)

valid_images_list = images[:num_valid]
train_images_list = images[num_valid:]

valid_img_ids = {img['id'] for img in valid_images_list}

# 4. Filter Annotations
valid_anns = [ann for ann in data['annotations'] if ann['image_id'] in valid_img_ids]
train_anns = [ann for ann in data['annotations'] if ann['image_id'] not in valid_img_ids]

# 5. Physical Move
print(f"Moving {len(valid_images_list)} images from {train_dir} to {valid_dir}...")
for img in valid_images_list:
    src = os.path.join(train_dir, img['file_name'])
    dst = os.path.join(valid_dir, img['file_name'])
    if os.path.exists(src):
        shutil.move(src, dst)
    else:
        print(f"Warning: File {img['file_name']} not found in train folder.")

# 6. Save the split JSONs back into Folder 5
def save_json(path, img_list, ann_list):
    out_data = {
        "info": data.get("info"),
        "licenses": data.get("licenses"),
        "categories": data.get("categories"),
        "images": img_list,
        "annotations": ann_list
    }
    with open(path, 'w') as f:
        json.dump(out_data, f)

save_json(train_json_path, train_images_list, train_anns)
save_json(valid_json_path, valid_images_list, valid_anns)

print(f"Split Complete for Dataset 5!")
print(f"Train: {len(train_images_list)} images | Valid: {len(valid_images_list)} images")