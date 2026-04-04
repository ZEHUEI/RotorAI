#train with tensorflow and keras
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["SM_FRAMEWORK"] = "tf.keras"
import cv2
import tensorflow as tf
tf.keras.backend.clear_session()

import json
import numpy as np
from pycocotools.coco import COCO
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models.losses import DiceLoss, BinaryFocalLoss
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
from keras.layers import Lambda

tf.config.optimizer.set_jit(True)
# mixed_precision.set_global_policy('mixed_float16')

#-------------------------------
#Use GPU RTX 5070 nvdia
#-------------------------------

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



policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
#-------------------------------
#COCO files
#-------------------------------
train_img="../Data/train"
val_img="../Data/valid"
train_json = "../Data/train/_annotations.coco.json"
val_json   = "../Data/valid/_annotations.coco.json"

with open("../Data/train/_annotations.coco.json") as f:
    coco = json.load(f)

total_images = len(coco['images'])
images_with_annotations = len(set(ann['image_id'] for ann in coco['annotations']))
empty_images = total_images - images_with_annotations

print(f"Total images: {total_images}")
print(f"Images WITH rust annotations: {images_with_annotations}")
print(f"Images with NO annotation (hard negatives): {empty_images}")
print(f"Hard negative ratio: {empty_images/total_images*100:.1f}%")

TARGET_SIZE = (768, 768)
BATCH_SIZE = 4
TARGET_LABELS = ["corrosion"]
NUM_TARGET_CLASSES = len(TARGET_LABELS)
BACKBONE = 'resnet50'
preprocess_input = sm.get_preprocessing(BACKBONE)

#-------------------------------
# Mask resterization
#-------------------------------
def rasterize_coco_annotations(anns, target_size, target_labels):
    H, W = target_size
    mask = np.zeros((H, W, len(target_labels)), dtype=np.uint8)
    label_to_channel = {label: i for i, label in enumerate(target_labels)}

    if not anns:
        return mask

    for ann in anns:
        category_name = ann['category_name']
        if category_name in label_to_channel:
            channel_idx = label_to_channel[category_name]
            for seg in ann['segmentation']:
                pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                temp_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(temp_mask, [pts], 1)
                mask[:, :, channel_idx] = np.maximum(mask[:, :, channel_idx], temp_mask)

    return mask

#------------------------------
# Metric loss custom
#------------------------------
dice = DiceLoss()
focal = BinaryFocalLoss(gamma=2.0,alpha=0.25)
bce = tf.keras.losses.BinaryCrossentropy()

def mean_iou_custom(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.65, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred,axis=[1,2,3])
    union = tf.reduce_sum(y_true,axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3]) - intersection
    iou=(intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)

def weighted_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce_loss = tf.reduce_mean(bce_loss)
    #change to 10:5::5
    return (10.0 * dice(y_true, y_pred)) + (5.0 * focal(y_true, y_pred)) + (8.0 * bce_loss)

#-----------------------------------------------------------------
#Augment
#-----------------------------------------------------------------
geometric_augment = tf.keras.Sequential([
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip("horizontal"),
    layers.RandomZoom((-0.1, 0.0)),
    layers.RandomRotation(0.03),
])

photometric_augment = tf.keras.Sequential([
    #brightness,contrast and gamma(shadow)
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.15),
layers.Lambda(lambda x: tf.image.random_hue(x, 0.03)),
layers.Lambda(lambda x: tf.image.adjust_gamma(x, gamma=tf.cast(tf.random.uniform([], 0.5, 1.5),x.dtype))),
layers.Lambda(lambda x: tf.cond(
        tf.random.uniform([]) > 0.5,
        lambda: tf.nn.avg_pool2d(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME'),
        lambda: x
    ))
])

#-----------------------------------------------------------------
#map functions
#-----------------------------------------------------------------
def resize_and_pad_numpy(img_np, mask_np, target_h, target_w):
    """Pure numpy resize+pad — avoids TF's off-by-one bug completely."""
    h, w = img_np.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h = min(int(np.floor(h * scale)), target_h)
    new_w = min(int(np.floor(w * scale)), target_w)

    # Resize with OpenCV (stays in numpy, no TF involved)
    img_resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Pad to target size
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top  = pad_h // 2
    pad_left = pad_w // 2

    img_padded  = np.pad(img_resized,  [[pad_top, pad_h - pad_top], [pad_left, pad_w - pad_left], [0, 0]])
    # mask may be (H, W) or (H, W, C) — handle both
    if mask_resized.ndim == 2:
        mask_padded = np.pad(mask_resized, [[pad_top, pad_h - pad_top], [pad_left, pad_w - pad_left]])
    else:
        mask_padded = np.pad(mask_resized, [[pad_top, pad_h - pad_top], [pad_left, pad_w - pad_left], [0, 0]])

    return img_padded, mask_padded

def apply_clahe(img_np):
    # Convert RGB to LAB color space
    # We only apply CLAHE to the 'Lightness' channel (L) so we don't ruin the rust colors
    img_uint8 = np.clip(img_np, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Create CLAHE object (clipLimit 3.0 is a good strong baseline for shadows)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)


def map_func(image_path, anns):
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((*TARGET_SIZE, 3), dtype=np.float32), \
               np.zeros((*TARGET_SIZE, NUM_TARGET_CLASSES), dtype=np.float32)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img.shape[:2]
    mask_np = rasterize_coco_annotations(anns, (h_orig, w_orig), TARGET_LABELS)

    # Resize + pad entirely in numpy — no TF involved, no off-by-one bugs
    img_padded, mask_padded = resize_and_pad_numpy(img, mask_np, TARGET_SIZE[0], TARGET_SIZE[1])

    # CLAHE on the padded uint8 image
    img_clahe = apply_clahe(img_padded)

    # Preprocess & cast
    img_final  = preprocess_input(img_clahe.astype(np.float32))
    mask_final = (mask_padded > 0.5).astype(np.float32)

    return tf.convert_to_tensor(img_final,  dtype=tf.float32), \
           tf.convert_to_tensor(mask_final, dtype=tf.float32)

def val_map_func(image_path, anns):
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((*TARGET_SIZE, 3), dtype=np.float32), \
            np.zeros((*TARGET_SIZE, NUM_TARGET_CLASSES), dtype=np.float32)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img.shape[:2]
    mask_np = rasterize_coco_annotations(anns, (h_orig, w_orig), TARGET_LABELS)

    # Resize + pad entirely in numpy — no TF involved, no off-by-one bugs
    img_padded, mask_padded = resize_and_pad_numpy(img, mask_np, TARGET_SIZE[0], TARGET_SIZE[1])

    # CLAHE on the padded uint8 image
    img_clahe = apply_clahe(img_padded)

    # Preprocess & cast
    img_final = preprocess_input(img_clahe.astype(np.float32))
    mask_final = (mask_padded > 0.5).astype(np.float32)

    return tf.convert_to_tensor(img_final, dtype=tf.float32), \
        tf.convert_to_tensor(mask_final, dtype=tf.float32)


def tf_train_map_func(img_path):
    def py_func(img_path_tensor):
        img_path_str = img_path_tensor.numpy().decode('utf-8')
        anns = train_ann_dict[img_path_str]
        img, mask = map_func(img_path_str, anns)

        return img,mask

    img, mask = tf.py_function(py_func, [img_path], [tf.float32, tf.float32])
    img = tf.reshape(img, [TARGET_SIZE[0], TARGET_SIZE[1], 3])
    mask = tf.reshape(mask, [TARGET_SIZE[0], TARGET_SIZE[1], NUM_TARGET_CLASSES])
    return img, mask


def tf_val_map_func(img_path):
    def py_func(img_path_tensor):
        img_path_str = img_path_tensor.numpy().decode('utf-8')
        anns = val_ann_dict[img_path_str]
        img, mask = val_map_func(img_path_str, anns)

        return img,mask

    img, mask = tf.py_function(py_func, [img_path], [tf.float32, tf.float32])
    img = tf.reshape(img, [TARGET_SIZE[0], TARGET_SIZE[1], 3])
    mask = tf.reshape(mask, [TARGET_SIZE[0], TARGET_SIZE[1], NUM_TARGET_CLASSES])
    return img, mask


#----------------------------------------
#Collect paths
#----------------------------------------
def collect_coco_data(img_dir, coco_json_path):
    coco = COCO(coco_json_path)
    image_paths = []
    annotations = []

    # Map category_id -> category_name
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}

    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        img_path = os.path.join(img_dir, img_filename)
        if os.path.exists(img_path):
            image_paths.append(img_path)
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            # Add category_name to each annotation
            for ann in anns:
                ann['category_name'] = cat_id_to_name[ann['category_id']]

            annotations.append(anns)
        else:
            print(f"Warning: Image not found: {img_path}")

    print(f"Collected {len(image_paths)} images from COCO JSON")
    return image_paths, annotations


#--------------
#prepare datasets
#-----------------
def apply_augmentations(img, mask):
    # img: (H, W, 3), mask: (H, W, num_classes)
    # Concatenate image and mask along channels to apply same geometric transform
    combined = tf.concat([img, mask], axis=-1)
    combined = geometric_augment(tf.expand_dims(combined, 0), training=True)
    combined = tf.squeeze(combined, 0)

    img_aug = combined[..., :3]
    mask_aug = combined[..., 3:]

    img_aug = photometric_augment(tf.expand_dims(img_aug, 0), training=True)
    img_aug = tf.squeeze(img_aug, 0)

    img_aug.set_shape([TARGET_SIZE[0], TARGET_SIZE[1], 3])
    mask_aug.set_shape([TARGET_SIZE[0], TARGET_SIZE[1], NUM_TARGET_CLASSES])

    return img_aug, mask_aug

print("Collecting training data paths and JSON content...")
train_image_paths, train_annotations  = collect_coco_data(train_img, train_json)
train_ann_dict = {path: anns for path, anns in zip(train_image_paths, train_annotations)}
train_image_paths = list(train_ann_dict.keys())
print(f"Found {len(train_image_paths)} image-JSON pairs for training.")


train_dataset = tf.data.Dataset.from_tensor_slices((
    train_image_paths
))

train_dataset = train_dataset.shuffle(buffer_size=1000)

train_dataset = train_dataset.repeat()

train_dataset = train_dataset.map(
    tf_train_map_func,
    num_parallel_calls=tf.data.AUTOTUNE
)
train_dataset = train_dataset.map(
    apply_augmentations,
    num_parallel_calls=tf.data.AUTOTUNE
)
train_dataset=train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

#-------------
#model
#------------
base_model = Unet(
    backbone_name=BACKBONE,
    encoder_weights='imagenet',
    classes=NUM_TARGET_CLASSES,
    activation='sigmoid',
    input_shape=TARGET_SIZE + (3,)
)
x = layers.SpatialDropout2D(0.2)(base_model.layers[-2].output)
base_model.trainable = True
outputs = base_model.layers[-1](x)

outputs = Lambda(lambda t: tf.cast(t, tf.float32))(outputs)

model = tf.keras.Model(
    inputs=base_model.input,
    outputs=outputs
)

PREVIOUS_WEIGHTS = "best_lastdance8.h5"
if os.path.exists(PREVIOUS_WEIGHTS):
    print(f"Loading weights from {PREVIOUS_WEIGHTS} to continue training...")
    model.load_weights(PREVIOUS_WEIGHTS, by_name=True, skip_mismatch=True)
    for layer in model.layers:
        # Only freeze encoder stages 1-3, nothing else
        should_freeze = (
                layer.name.startswith('stage1') or
                layer.name.startswith('stage2')
                # or
                # layer.name.startswith('stage3')
        )
        # Explicitly never freeze decoder no matter what
        if 'decoder' in layer.name:
            layer.trainable = True
        elif should_freeze:
            layer.trainable = False
        else:
            layer.trainable = True

    print(f"Trainable layers: {sum(1 for l in model.layers if l.trainable)}")
    print(f"Frozen layers: {sum(1 for l in model.layers if not l.trainable)}")

    print("\n--- Layer Freeze Check ---")
    for layer in model.layers:
        if not layer.trainable:
            print(f"FROZEN: {layer.name}")
    print("--- End Check ---\n")

else:
    print("No previous weights found. Starting training from scratch.")


#optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6)
optimizer = mixed_precision.LossScaleOptimizer(optimizer)
model.compile(optimizer=optimizer,loss=weighted_loss,metrics=[mean_iou_custom])

EPOCHS =100

#--------------
#validation usage
#------------------

print("Collecting validation data paths and JSON content...")
val_image_paths, val_annotations  = collect_coco_data(val_img, val_json)
val_ann_dict   = {path: anns for path, anns in zip(val_image_paths, val_annotations)}
print(f"Found {len(val_image_paths)} image-JSON pairs for validation.")

val_dataset = tf.data.Dataset.from_tensor_slices((
    val_image_paths
)).map(
    tf_val_map_func,
    num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


#-------------
#callbacks
#-------------
#unet 7 now! 27/3/2026 00:44AM
callbacks=[
    EarlyStopping(patience=10,verbose=1,monitor='val_mean_iou_custom',mode='max',restore_best_weights=True),
    ModelCheckpoint('best_lastdance9.h5',verbose=1,monitor='val_mean_iou_custom',save_best_only=True,mode='max'),
    ReduceLROnPlateau(monitor='val_mean_iou_custom', factor=0.3, patience=5, min_lr=1e-6, verbose=1,mode='max')
]

#---------
#start train
#----------
def train():
    print(f"\nStarting model training for {EPOCHS} epochs...")

    class_weights = {0: 1.0, 1: 10.0}
    STEPS = len(train_image_paths) // BATCH_SIZE
    VAL_STEPS = len(val_image_paths) // BATCH_SIZE

    history = model.fit(
        train_dataset,
        steps_per_epoch=STEPS,
        epochs=EPOCHS,
        validation_data = val_dataset,
        validation_steps=VAL_STEPS,
        callbacks=callbacks
    )

    # Save the weights so you can reload the model later without retraining
    model.save_weights("lastdance9.h5")
    print("Training finished and weights saved.")

    print("Generating training plots...")
    epochs_range = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(15, 5))

    # Plot 1: Mean IoU (Confidence/Accuracy)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['mean_iou_custom'], label='Training IoU')
    plt.plot(epochs_range, history.history['val_mean_iou_custom'], label='Validation IoU')
    plt.title('Model Mean IoU (Confidence Check)')
    plt.xlabel('Epochs')
    plt.ylabel('IoU Score')
    plt.legend()

    # Plot 2: Loss (The "Pain" Score)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], label='Training Loss')
    plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
    plt.title('Model Weighted Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()

    plt.tight_layout()
    plt.savefig('lastdance9.png')
    print("Graphs saved as 'lastdance9.png'. Check this to see the learning curve!")

if __name__ == "__main__":
    train()
