import json
import os
from shutil import copy2

# Base folder where your dataset folders 1,2,3 live
base_dir = r"C:\Users\Ze Huei\PycharmProjects\RotorAI\Data"

datasets = ['1', '2', '3']
splits = ['train', 'valid', 'test']  # merge all splits at once

for split in splits:
    output_dir = os.path.join(base_dir, split)
    os.makedirs(output_dir, exist_ok=True)

    merged_json = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_id_offset = 0
    annotation_id_offset = 0
    category_map = {}

    for ds in datasets:
        json_path = os.path.join(base_dir, ds, split, '_annotations.coco.json')
        if not os.path.exists(json_path):
            print(f"No JSON found for {json_path}, skipping...")
            continue

        with open(json_path) as f:
            data = json.load(f)

        # merge categories
        for cat in data['categories']:
            if cat['id'] not in category_map:
                new_id = len(category_map) + 1
                category_map[cat['id']] = new_id
                cat_copy = cat.copy()
                cat_copy['id'] = new_id
                merged_json['categories'].append(cat_copy)

        # merge images
        for img in data['images']:
            old_id = img['id']
            img['id'] = img['id'] + image_id_offset

            # Rename image to avoid filename conflicts
            new_file_name = f"{ds}_{img['file_name']}"
            img['file_name'] = new_file_name
            merged_json['images'].append(img)

            src_image_path = os.path.join(base_dir, ds, split, img['file_name'].split("_",1)[1])
            dst_image_path = os.path.join(output_dir, new_file_name)
            os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
            if os.path.exists(src_image_path):
                copy2(src_image_path, dst_image_path)
            else:
                print(f"Image not found: {src_image_path}")

        # merge annotations
        for ann in data['annotations']:
            ann['id'] = ann['id'] + annotation_id_offset
            ann['image_id'] = ann['image_id'] + image_id_offset
            ann['category_id'] = category_map[ann['category_id']]
            merged_json['annotations'].append(ann)

        # update offsets
        if merged_json['images']:
            image_id_offset = max([img['id'] for img in merged_json['images']]) + 1
        if merged_json['annotations']:
            annotation_id_offset = max([ann['id'] for ann in merged_json['annotations']]) + 1

    # --- SAVE THE MERGED JSON ---
    merged_json_path = os.path.join(output_dir, '_annotations.coco.json')
    with open(merged_json_path, 'w') as f:
        json.dump(merged_json, f)
    print(f"Merged COCO JSON saved to {merged_json_path}")
