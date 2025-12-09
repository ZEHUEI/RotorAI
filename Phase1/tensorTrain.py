#train with tensorflow and keras
import cv2
import tensorflow as tf
import json
import os
import numpy as np
from pycocotools.coco import COCO
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import ResNet50

#COCO files
train_img="../Data/images/train"
val_img="../Data/images/validation"
train_json = "../Data/annotations/train"
val_json = "../Data/annotations/validation"

TARGET_SIZE = (512, 512)
TARGET_LABELS = ["Crack", "Rust"]
NUM_TARGET_CLASSES = len(TARGET_LABELS)
base = ResNet50(weights='imagenet', include_top=False, input_shape=TARGET_SIZE + (3,))

def rasterize_polygon_to_mask(json_data_str, target_labels, target_size):
    python_target_labels = [s.decode('utf-8') for s in target_labels.numpy()]
    python_target_size = target_size.numpy()
    data = json.loads(json_data_str.numpy().decode('utf-8'))
    H, W = data['imageHeight'], data['imageWidth']

    mask = np.zeros((H, W, len(target_labels)), dtype=np.uint8)
    label_to_channel = {label: i for i, label in enumerate(python_target_labels)}
    try:
        for shape in data['shapes']:
            label = shape['label']

            if label in label_to_channel:
                channel_index = label_to_channel[label]
                points = np.array(shape['points'], dtype=np.int32)

                temp_mask = np.zeros((H, W), dtype=np.uint8)

                cv2.fillPoly(temp_mask, [points.reshape((-1, 1, 2))], color=1)

                mask[:, :, channel_index] = np.maximum(mask[:, :, channel_index], temp_mask)
#----------------------------------------------------------------------------------------------------------
    except Exception as e:
        print(f"CRITICAL RASTERIZATION ERROR in {data.get('imagePath', 'unknown_file')}: {e}")
    # Return an empty mask if failure occurs
        return np.zeros(python_target_size + (len(target_labels),), dtype=np.uint8)

    # --- DEBUG: Check mask content before resizing ---
    # total_mask_pixels = np.sum(mask > 0)
    # if total_mask_pixels == 0 and len(data['shapes']) > 0:
    #     print(f"DEBUG WARNING: {data.get('imagePath')} had {len(data['shapes'])} shapes but mask is empty!")
    # else:
    #     print(f"DEBUG: {data.get('imagePath')} rasterized {total_mask_pixels} pixels.")
#----------------------------------------------------------------------------------------------------------
    resized_mask = cv2.resize(
        mask,
        (python_target_size[1], python_target_size[0]),  # cv2 uses (W, H)
        interpolation=cv2.INTER_NEAREST
    )
    return resized_mask

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def mean_iou_custom(y_true, y_pred, smooth=1e-6,threshold=0.5):
    # Threshold the predicted probabilities to get binary masks (0 or 1)
    y_pred = tf.cast(y_pred > threshold, tf.float32)

    # Flatten the tensors for easier computation: (Batch * H * W, Num_Classes)
    y_true_f = tf.reshape(y_true, [-1, NUM_TARGET_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, NUM_TARGET_CLASSES])

    # Calculate intersection and union per class
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f + y_pred_f, axis=0) - intersection

    # IoU for each class
    iou_per_class = (intersection + smooth) / (union + smooth)

    # Mean IoU across all classes
    return tf.reduce_mean(iou_per_class)

geometric_augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),
    layers.RandomRotation(0.1),
])

photometric_augment = tf.keras.Sequential([
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.1),
])

def map_func(image_path, json_data_str):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, TARGET_SIZE)

    mask_tensor = tf.py_function(
        rasterize_polygon_to_mask,
        [json_data_str, tf.constant(TARGET_LABELS), tf.constant(TARGET_SIZE)],
        tf.uint8
    )

    mask_tensor.set_shape([TARGET_SIZE[0], TARGET_SIZE[1], NUM_TARGET_CLASSES])
    mask_tensor = tf.cast(mask_tensor, tf.float32)

    stacked_img_mask = tf.concat([img, mask_tensor], axis=-1)
    augmented_stack = geometric_augment(stacked_img_mask)

    augmented_img = augmented_stack[:, :, :3]
    augmented_mask = augmented_stack[:, :, 3:]

    augmented_img = photometric_augment(augmented_img)
    augmented_img = augmented_img / 255

    return augmented_img, augmented_mask

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
).batch(16).prefetch(tf.data.AUTOTUNE)

#-------------------------------------------------------------------
# REVISE USING RES50 Model
#-------------------------------------------------------------------

def unet_model(input_size=TARGET_SIZE + (3,), num_classes=NUM_TARGET_CLASSES):
    inputs = base.input

    def conv_block(input_tensor, num_filters):
        x = layers.Conv2D(num_filters, (3, 3), padding='same',kernel_initializer='he_normal')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(num_filters, (3, 3), padding='same',kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    #---Encoder (RES50) downsampling
    c4 = base.get_layer('conv4_block6_out').output
    c3 = base.get_layer('conv3_block4_out').output
    c2 = base.get_layer('conv2_block3_out').output
    c1 = base.get_layer('conv1_relu').output
    bottleneck = base.output

    # --- Encoder (Upsampling Path) ---
    u1 = layers.UpSampling2D((2,2))(bottleneck)
    u1 = layers.concatenate([u1,c4])
    d1 = conv_block(u1,512)

    u2 = layers.UpSampling2D((2, 2))(d1)
    u2 = layers.concatenate([u2, c3])
    d2 = conv_block(u2, 256)

    u3 = layers.UpSampling2D((2, 2))(d2)
    u3 = layers.concatenate([u3, c2])
    d3 = conv_block(u3, 128)

    u4 = layers.UpSampling2D((2, 2))(d3)
    u4 = layers.concatenate([u4, c1])
    d4 = conv_block(u4, 64)

    u5 = layers.UpSampling2D((2,2))(d4)
    d5 = conv_block(u5,32)

    # Output Layer: NUM_TARGET_CLASSES channels (2: Crack, Rust)
    # Use Sigmoid because it's multi-label (a pixel can potentially be both)
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(d5)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

model = unet_model()
model.compile(optimizer='adam',loss=bce_dice_loss,metrics=['accuracy',mean_iou_custom])

EPOCHS =100

#--validation usage---------------------------------------------------------------------------------------
print("Collecting validation data paths and JSON content...")
val_image_paths, val_json_data_strings = collect_data_paths(val_img, val_json)
print(f"Found {len(val_image_paths)} image-JSON pairs for validation.")

val_dataset = tf.data.Dataset.from_tensor_slices((
    val_image_paths,
    val_json_data_strings
)).map(
    map_func,
    num_parallel_calls=tf.data.AUTOTUNE
).batch(16).prefetch(tf.data.AUTOTUNE)
#--end

#callbacks
callbacks=[
    EarlyStopping(patience=10,verbose=1,monitor='val_mean_io_u_custom',mode='max'),
    ModelCheckpoint('best_unet_weights.h5',verbose=1,monitor='val_mean_io_u_custom',save_best_only=True,mode='max')
]
#----Start
print(f"\nStarting model training for {EPOCHS} epochs...")

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data = val_dataset,
    callbacks=callbacks
)

# Save the weights so you can reload the model later without retraining
model.save_weights("unet3_crack_rust_dacl10k_weights.weights.h5")
print("Training finished and weights saved.")
