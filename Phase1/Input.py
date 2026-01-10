import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models as sm
from segmentation_models import Unet
from keras.layers import Lambda
from tensorflow.keras import layers, models

TARGET_SIZE = (512, 512)
TARGET_LABELS = ["Crack", "Rust"]
NUM_TARGET_CLASSES = 2
BACKBONE = 'resnet50'
#old
# NUM_TARGET_CLASSES = len(TARGET_LABELS)

# old
# def unet_model(input_size=TARGET_SIZE + (3,), num_classes=NUM_TARGET_CLASSES):
#     inputs = tf.keras.Input(input_size)
#
#     # --- Encoder (Downsampling Path) ---
#     c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
#     c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
#     p1 = layers.MaxPooling2D((2, 2))(c1)
#
#     c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
#     c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
#     p2 = layers.MaxPooling2D((2, 2))(c2)
#
#     # --- Bottleneck ---
#     c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
#     c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
#
#     # --- Decoder (Upsampling Path) ---
#     u4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
#     u4 = layers.concatenate([u4, c2])  # Skip connection
#     c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u4)
#     c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
#
#     u5 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c4)
#     u5 = layers.concatenate([u5, c1])  # Skip connection
#     c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u5)
#     c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c5)
#
#     # Output Layer
#     outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c5)
#
#     model = models.Model(inputs=[inputs], outputs=[outputs])
#     return model
#
# model = unet_model()

#----------------------new
base_model = Unet(
    backbone_name=BACKBONE,
    encoder_weights=None,  # IMPORTANT when loading trained weights
    classes=NUM_TARGET_CLASSES,
    activation='sigmoid',
    input_shape=TARGET_SIZE + (3,)
)

outputs = Lambda(lambda t: tf.cast(t, tf.float32))(base_model.output)

model = tf.keras.Model(
    inputs=base_model.input,
    outputs=outputs
)
#---------------------------
#unet3_crack_rust_dacl10k_weights
WEIGHTS_FILE = "unetLinux_crack_rust_dacl10k_weights.h5"
try:
    model.load_weights(WEIGHTS_FILE)
    print(f"Successfully loaded trained weights from {WEIGHTS_FILE}.")
except tf.errors.NotFoundError:
    print(f"Error: Weights file '{WEIGHTS_FILE}' not found.")
    print("Please ensure you run tensorTrain.py first to generate the weights file.")
    exit() # Exit the script if weights aren't found

# old TEST
# def predict_image(model, image_path, target_size=TARGET_SIZE, threshold=0.2):
#     img = tf.io.read_file(image_path)
#     img = tf.image.decode_jpeg(img, channels=3)
#     original_img = img.numpy()
#
#     img = tf.image.resize(img, target_size)
#     img = tf.expand_dims(img, axis=0)
#     img = tf.cast(img, tf.float32) / 255.0
#
#     print(f"Predicting mask for {os.path.basename(image_path)}...")
#     prediction = model.predict(img)[0]
#
#     binary_mask = (prediction > threshold).astype(np.uint8)
#
#     mask_resized = cv2.resize(
#         binary_mask,
#         (original_img.shape[1], original_img.shape[0]),
#         interpolation=cv2.INTER_NEAREST
#     )
#
#     crack_mask = mask_resized[:, :, 0]
#     rust_mask = mask_resized[:, :, 1]
#
#     return original_img, crack_mask, rust_mask


#new TEST
preprocess_input = sm.get_preprocessing(BACKBONE)


def predict_image(model, image_path, target_size=TARGET_SIZE, threshold=0.2):  # Default threshold lowered to 0.2
    # 1. Load original image
    img_raw = tf.io.read_file(image_path)
    img_decoded = tf.image.decode_jpeg(img_raw, channels=3)
    original_img = img_decoded.numpy()

    # 2. Resize to the size the model expects
    img_resized = tf.image.resize(img_decoded, target_size).numpy()

    # 3. CRITICAL: Use ResNet50 preprocessing instead of /255.0
    img_preprocessed = preprocess_input(img_resized)
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    print(f"Predicting mask for {os.path.basename(image_path)}...")
    prediction = model.predict(img_batch)[0]

    # 4. Use a lower threshold (0.2 is better for thin cracks)
    binary_mask = (prediction > threshold).astype(np.uint8)

    # 5. Resize mask back to original size for overlay
    mask_resized = cv2.resize(
        binary_mask,
        (original_img.shape[1], original_img.shape[0]),
        interpolation=cv2.INTER_NEAREST  # Keep thin lines sharp
    )

    crack_mask = mask_resized[:, :, 0]
    rust_mask = mask_resized[:, :, 1]

    return original_img, crack_mask, rust_mask

# --- Test Execution Block-----------------------------------------------------------

TESTING123 = "../Outcomes/Input/motor3.jpg"
#0078 rust and 0089 cracks
# TEST_IMAGE_PATH = "../Data/images/testdev/dacl10k_v2_testdev_0089.jpg"

if os.path.exists(TESTING123):
    print("\n--- Starting Inference ---")
    original_image, crack_mask, rust_mask = predict_image(model, TESTING123)

    print(f"Test Image: {os.path.basename(TESTING123)}")
    print(f"Predicted Crack Pixels: {np.sum(crack_mask)}")
    print(f"Predicted Rust Pixels: {np.sum(rust_mask)}")

    # --- Visualization ---
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))

    # Plot 1: Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")

    # Plot 2: Crack Detection (Overlaying red)
    crack_overlay = original_image.copy()
    crack_overlay[crack_mask > 0] = [0, 255, 255]
    plt.subplot(1, 3, 2)
    plt.imshow(crack_overlay)
    plt.title("Cracks Detected (Red)")

    # Plot 3: Rust Detection (Overlaying green)
    rust_overlay = original_image.copy()
    rust_overlay[rust_mask > 0] = [0, 255, 0]
    plt.subplot(1, 3, 3)
    plt.imshow(rust_overlay)
    plt.title("Rust Detected (Green)")

    plt.tight_layout()
    plt.show()
else:
    print(f"Error: Test image not found at {TESTING123}")