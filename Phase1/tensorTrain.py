#train with tensorflow and keras
import cv2
import tensorflow as tf
import json
import os
import numpy as np
from pycocotools.coco import COCO
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models.losses import DiceLoss, BinaryFocalLoss

#Use GPU RTX 5070 nvdia
print("GPU Checker")
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"using GPU {len(gpu_devices)} tensorflow")
    for gpu in gpu_devices:
        print(f"name : {gpu.name}")
        tf.config.experimental.set_memory_growth(gpu,True)
    print("train on GPU")
else:
    print("Warning nop gpu  runb on cpu")
    print("------------------------------------\n")


#COCO files
train_img="../Data/images/train"
val_img="../Data/images/validation"
train_json = "../Data/annotations/train"
val_json = "../Data/annotations/validation"

TARGET_SIZE = (1024, 1024)
TARGET_LABELS = ["Crack", "Rust"]
NUM_TARGET_CLASSES = len(TARGET_LABELS)
BACKBONE = 'resnet50'
preprocess_input = sm.get_preprocessing(BACKBONE)
base = Unet(backbone_name=BACKBONE, encoder_weights='imagenet', classes=NUM_TARGET_CLASSES, activation='sigmoid')

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
        interpolation=cv2.INTER_AREA
    )
    resized_mask = (resized_mask > 0.5).astype(np.uint8)
    return resized_mask

def mean_iou_custom(y_true, y_pred, smooth=1e-6,threshold=0.35):
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

def crack_iou(y_true, y_pred, threshold=0.25):
    """IoU specifically for crack detection"""
    y_pred = tf.cast(y_pred[..., 0] > threshold, tf.float32)
    y_true = y_true[..., 0]

    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    return (intersection + smooth) / (union + smooth)


def rust_iou(y_true, y_pred, threshold=0.25):
    """IoU specifically for rust detection"""
    y_pred = tf.cast(y_pred[..., 1] > threshold, tf.float32)
    y_true = y_true[..., 1]

    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    return (intersection + smooth) / (union + smooth)


def weighted_loss(y_true, y_pred):
    """Give more weight to cracks"""
    crack_weight = 8.0  # 3x more important
    rust_weight = 1.0

    dice = DiceLoss()
    focal = BinaryFocalLoss()
    bce = tf.keras.losses.BinaryCrossentropy()

    # Split channels
    yt_c = y_true[..., 0:1]
    yp_c = y_pred[..., 0:1]
    yt_r = y_true[..., 1:2]
    yp_r = y_pred[..., 1:2]

    crack_loss = 0.4 * dice(yt_c, yp_c) + 0.4 * focal(yt_c, yp_c) + 0.2 * bce(yt_c, yp_c)
    rust_loss = 0.5 * dice(yt_r, yp_r) + 0.5 * bce(yt_r, yp_r)

    return crack_weight * crack_loss + rust_weight * rust_loss

#-----------------------------------------------------------------
geometric_augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomZoom((-0.1, 0.0)),
    layers.RandomRotation(0.03),
])

photometric_augment = tf.keras.Sequential([
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.1),
])
#--------------------------------------------------------------------

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
    augmented_img = preprocess_input(augmented_img)

    return augmented_img, augmented_mask
#-----------------------------------------------
'''
Validation no augmentated
'''
def val_map_func(image_path,json_data_str):
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

    img = preprocess_input(img)
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

    if len(image_paths) == 0:
        raise ValueError(f"No valid image-JSON pairs found in {img_dir}")

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
# Removed
#-------------------------------------------------------------------

# def unet_model(input_size=TARGET_SIZE + (3,), num_classes=NUM_TARGET_CLASSES):
#     inputs = base.input
#
#     def conv_block(input_tensor, num_filters):
#         x = layers.Conv2D(num_filters, (3, 3), padding='same',kernel_initializer='he_normal')(input_tensor)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.Conv2D(num_filters, (3, 3), padding='same',kernel_initializer='he_normal')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         return x
#
#     #---Encoder (RES50) downsampling
#     c4 = base.get_layer('conv4_block6_out').output
#     c3 = base.get_layer('conv3_block4_out').output
#     c2 = base.get_layer('conv2_block3_out').output
#     c1 = base.get_layer('conv1_relu').output
#     bottleneck = base.output
#
#     # --- Encoder (Upsampling Path) ---
#     u1 = layers.UpSampling2D((2,2))(bottleneck)
#     u1 = layers.concatenate([u1,c4])
#     d1 = conv_block(u1,512)
#
#     u2 = layers.UpSampling2D((2, 2))(d1)
#     u2 = layers.concatenate([u2, c3])
#     d2 = conv_block(u2, 256)
#
#     u3 = layers.UpSampling2D((2, 2))(d2)
#     u3 = layers.concatenate([u3, c2])
#     d3 = conv_block(u3, 128)
#
#     u4 = layers.UpSampling2D((2, 2))(d3)
#     u4 = layers.concatenate([u4, c1])
#     d4 = conv_block(u4, 64)
#
#     u5 = layers.UpSampling2D((2,2))(d4)
#     d5 = conv_block(u5,32)
#
#     # Output Layer: NUM_TARGET_CLASSES channels (2: Crack, Rust)
#     # Use Sigmoid because it's multi-label (a pixel can potentially be both)
#     outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(d5)
#
#     model = models.Model(inputs=[inputs], outputs=[outputs])
#     return model

# model = unet_model()
model = Unet(backbone_name=BACKBONE,encoder_weights='imagenet',classes=NUM_TARGET_CLASSES,activation='sigmoid',input_shape=TARGET_SIZE+(3,))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss=weighted_loss,metrics=['accuracy',mean_iou_custom,crack_iou,rust_iou])

EPOCHS =100

#--validation usage---------------------------------------------------------------------------------------
print("Collecting validation data paths and JSON content...")
val_image_paths, val_json_data_strings = collect_data_paths(val_img, val_json)
print(f"Found {len(val_image_paths)} image-JSON pairs for validation.")

val_dataset = tf.data.Dataset.from_tensor_slices((
    val_image_paths,
    val_json_data_strings
)).map(
    val_map_func,
    num_parallel_calls=tf.data.AUTOTUNE
).batch(16).prefetch(tf.data.AUTOTUNE)
#--end

#callbacks
callbacks=[
    EarlyStopping(patience=15,verbose=1,monitor='val_crack_iou',mode='max',restore_best_weights=True),
    ModelCheckpoint('bestLinux_unet_weights.h5',verbose=1,monitor='val_crack_iou',save_best_only=True,mode='max')
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
model.save_weights("unetLinux_crack_rust_dacl10k_weights.weights.h5")
print("Training finished and weights saved.")
