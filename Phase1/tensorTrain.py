#train with tensorflow and keras
import cv2
import tensorflow as tf
import json
import os
import numpy as np
from pycocotools.coco import COCO
from tensorflow.keras import layers, models


#COCO files
train_img="../Data/images/train"
val_img="../Data/images/validation"
train_json = "../Data/annotations/train"
val_json = "../Data/annotations/validation"

TARGET_SIZE = (512, 512)
TARGET_LABELS = ["Crack", "Rust"]
NUM_TARGET_CLASSES = len(TARGET_LABELS)

def rasterize_polygon_to_mask(json_data_str, target_labels, target_size):
    python_target_labels = [s.decode('utf-8') for s in target_labels.numpy()]
    python_target_size = target_size.numpy()
    data = json.loads(json_data_str.numpy().decode('utf-8'))
    H, W = data['imageHeight'], data['imageWidth']

    mask = np.zeros((H, W, len(target_labels)), dtype=np.uint8)
    label_to_channel = {label: i for i, label in enumerate(python_target_labels)}
    for shape in data['shapes']:
        label = shape['label']

        if label in label_to_channel:
            channel_index = label_to_channel[label]
            points = np.array(shape['points'], dtype=np.int32)

            temp_mask = np.zeros((H, W), dtype=np.uint8)

            cv2.fillPoly(temp_mask, [points.reshape((-1, 1, 2))], color=1)

            mask[:, :, channel_index] = np.maximum(mask[:, :, channel_index], temp_mask)

    resized_mask = cv2.resize(
        mask,
        (python_target_size[1], python_target_size[0]),  # cv2 uses (W, H)
        interpolation=cv2.INTER_NEAREST
    )
    return resized_mask


def map_func(image_path, json_data_str):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, TARGET_SIZE)
    img = img / 255.0

    mask_tensor = tf.py_function(
        rasterize_polygon_to_mask,
        [json_data_str, tf.constant(TARGET_LABELS), tf.constant(TARGET_SIZE)],
        tf.uint8
    )

    mask_tensor.set_shape([TARGET_SIZE[0], TARGET_SIZE[1], NUM_TARGET_CLASSES])
    mask_tensor = tf.cast(mask_tensor, tf.float32)

    return img, mask_tensor

def collect_data_paths(img_dir, json_dir):
    image_paths = []
    json_data_strings = []

    image_filenames = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg'))]

    for filename in image_filenames:
        base_name = os.path.splitext(filename)[0]
        json_filename = base_name + '.json'

        image_path = os.path.join(img_dir, filename)
        json_path = os.path.join(json_dir, json_filename)

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    json_content_string = f.read()

                image_paths.append(image_path)
                json_data_strings.append(json_content_string)
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
        else:
            print(f"Warning: No JSON found for image {filename}")

    return image_paths, json_data_strings

print("Collecting training data paths and JSON content...")
train_image_paths, train_json_data_strings = collect_data_paths(train_img, train_json)
print(f"Found {len(train_image_paths)} image-JSON pairs for training.")

train_dataset = tf.data.Dataset.from_tensor_slices((
    train_image_paths,
    train_json_data_strings
))

train_dataset = train_dataset.map(
    map_func,
    num_parallel_calls=tf.data.AUTOTUNE
).batch(4).prefetch(tf.data.AUTOTUNE)

def unet_model(input_size=TARGET_SIZE + (3,), num_classes=NUM_TARGET_CLASSES):
    inputs = tf.keras.Input(input_size)

    # --- Encoder (Downsampling Path) ---
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # --- Bottleneck ---
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)

    # --- Decoder (Upsampling Path) ---
    u4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = layers.concatenate([u4, c2]) # Skip connection
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)

    u5 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c1]) # Skip connection
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c5)

    # Output Layer: NUM_TARGET_CLASSES channels (2: Crack, Rust)
    # Use Sigmoid because it's multi-label (a pixel can potentially be both)
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c5)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

model = unet_model()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy',tf.keras.metrics.MeanIoU(num_classes=NUM_TARGET_CLASSES)])

EPOCHS =20
print(f"\nStarting model training for {EPOCHS} epochs...")

history = model.fit(
    train_dataset,
    epochs=EPOCHS
)

# Save the weights so you can reload the model later without retraining
model.save_weights("unet_crack_rust_dacl10k_weights.weights.h5")
print("Training finished and weights saved.")
