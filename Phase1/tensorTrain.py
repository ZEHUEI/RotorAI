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
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models.losses import DiceLoss, BinaryFocalLoss
from tensorflow.keras import mixed_precision
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

#-------------------------------
#COCO files
#-------------------------------
train_img="../Data/train"
val_img="../Data/valid"
train_json = "../Data/train/_annotations.coco.json"
val_json   = "../Data/valid/_annotations.coco.json"

TARGET_SIZE = (512, 512)
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

    for ann in anns:
        category_name = ann['category_name']
        if category_name in label_to_channel:
            channel_idx = label_to_channel[category_name]
            for seg in ann['segmentation']:
                pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                temp_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(temp_mask, [pts], 1)
                kernel = np.ones((3, 3), np.uint8)
                mask[:, :, channel_idx] = cv2.dilate(mask[:, :, channel_idx], kernel, iterations=1)
                mask[:, :, channel_idx] = np.maximum(mask[:, :, channel_idx], temp_mask)

    return mask

#------------------------------
# Metric loss custom
#------------------------------
dice = DiceLoss()
focal = BinaryFocalLoss()
bce = tf.keras.losses.BinaryCrossentropy()

def mean_iou_custom(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def weighted_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    return dice(y_true, y_pred) + (2.0 * focal(y_true, y_pred))

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
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.1),
layers.Lambda(lambda x: tf.cond(
        tf.random.uniform([]) > 0.5,
        lambda: tf.image.gaussian_blur(x, kernel_size=(3, 3), sigma=0.5),
        lambda: x
    ))
])

#-----------------------------------------------------------------
#map functions
#-----------------------------------------------------------------
def map_func(image_path, anns):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, TARGET_SIZE)

    # Rasterize annotations into mask
    mask_np = rasterize_coco_annotations(anns, TARGET_SIZE, TARGET_LABELS)
    mask_tensor = tf.convert_to_tensor(mask_np, dtype=tf.float32)

    img = preprocess_input(img)
    mask_tensor = tf.cast(mask_tensor, tf.float32)

    return img, mask_tensor

def val_map_func(image_path, anns):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SIZE)

    mask_np = rasterize_coco_annotations(anns, TARGET_SIZE, TARGET_LABELS)

    img = preprocess_input(img.astype(np.float32))
    mask = mask_np.astype(np.float32)

    return img, mask



def tf_train_map_func(img_path):
    def py_func(img_path_tensor):
        img_path_str = img_path_tensor.numpy().decode('utf-8')
        anns = train_ann_dict[img_path_str]
        return map_func(img_path_str, anns)

    img, mask = tf.py_function(py_func, [img_path], [tf.float32, tf.float32])
    img.set_shape([TARGET_SIZE[0], TARGET_SIZE[1], 3])
    mask.set_shape([TARGET_SIZE[0], TARGET_SIZE[1], NUM_TARGET_CLASSES])
    return img, mask


def tf_val_map_func(img_path):
    def py_func(img_path_tensor):
        img_path_str = img_path_tensor.numpy().decode('utf-8')
        anns = val_ann_dict[img_path_str]
        return val_map_func(img_path_str, anns)

    img, mask = tf.py_function(py_func, [img_path], [tf.float32, tf.float32])
    img.set_shape([TARGET_SIZE[0], TARGET_SIZE[1], 3])
    mask.set_shape([TARGET_SIZE[0], TARGET_SIZE[1], NUM_TARGET_CLASSES])
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
    return img_aug, mask_aug

print("Collecting training data paths and JSON content...")
train_image_paths, train_annotations  = collect_coco_data(train_img, train_json)
train_ann_dict = {path: anns for path, anns in zip(train_image_paths, train_annotations)}
print(f"Found {len(train_image_paths)} image-JSON pairs for training.")


train_dataset = tf.data.Dataset.from_tensor_slices((
    train_image_paths
))

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

outputs = Lambda(lambda t: tf.cast(t, tf.float32))(base_model.output)

model = tf.keras.Model(
    inputs=base_model.input,
    outputs=outputs
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss=weighted_loss,metrics=[mean_iou_custom])

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
callbacks=[
    EarlyStopping(patience=15,verbose=1,monitor='val_mean_iou_custom',mode='max',restore_best_weights=True),
    ModelCheckpoint('best_unet2_corrosion.h5',verbose=1,monitor='val_mean_iou_custom',save_best_only=True,mode='max')
]

#---------
#start train
#----------
def train():
    print(f"\nStarting model training for {EPOCHS} epochs...")

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data = val_dataset,
        callbacks=callbacks
    )

    # Save the weights so you can reload the model later without retraining
    model.save_weights("unet2_corrosion_model.h5")
    print("Training finished and weights saved.")

if __name__ == "__main__":
    train()
