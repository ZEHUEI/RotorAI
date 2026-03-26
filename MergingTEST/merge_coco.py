import json
import os
from shutil import copy2

# Base folder where your dataset folders 1,2,3 live
base_dir = r"C:\Users\Ze Huei\PycharmProjects\RotorAI\Data"

datasets = ['1', '2', '3','4','5']
splits = ['train', 'valid', 'test']  # merge all splits at once

for split in splits:
    output_dir = os.path.join(base_dir, split)
    os.makedirs(output_dir, exist_ok=True)

    merged_json = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id":1,"name":"corrosion","supercategory":"none"}
        ]
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
        # for cat in data['categories']:
        #     if cat['id'] not in category_map:
        #         new_id = len(category_map) + 1
        #         category_map[cat['id']] = new_id
        #         cat_copy = cat.copy()
        #         cat_copy['id'] = new_id
        #         merged_json['categories'].append(cat_copy)

        # merge images
        for img in data['images']:
            img_copy = img.copy()
            img_copy['id'] = img['id'] + image_id_offset
            new_file_name = f"{ds}_{img['file_name']}"
            img_copy['file_name'] = new_file_name
            merged_json['images'].append(img_copy)

            # File copy logic
            src_image_path = os.path.join(base_dir, ds, split, img['file_name'])
            dst_image_path = os.path.join(output_dir, new_file_name)
            if os.path.exists(src_image_path):
                copy2(src_image_path, dst_image_path)

        # merge annotations & categories
        if ds not in ['4','5'] :
            # Create a local map for this dataset to match our "corrosion" ID 1
            local_cat_map = {}
            for cat in data['categories']:
                if cat['name'].lower() == "corrosion":
                    local_cat_map[cat['id']] = 1

            for ann in data['annotations']:
                # Only keep annotations if they are for 'corrosion'
                if ann['category_id'] in local_cat_map:
                    ann_copy = ann.copy()
                    ann_copy['id'] = ann['id'] + annotation_id_offset
                    ann_copy['image_id'] = ann['image_id'] + image_id_offset
                    ann_copy['category_id'] = 1  # Force to our master Corrosion ID
                    merged_json['annotations'].append(ann_copy)

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
