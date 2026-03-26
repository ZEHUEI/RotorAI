import os
from Phase1.PostProcess import detect_rust_and_cracks
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
TARGET_LABELS = ["corrosion"]
NUM_TARGET_CLASSES = len(TARGET_LABELS)
BACKBONE = 'resnet50'

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
WEIGHTS_FILE = "best_unet2_with_faces_corrosion.h5"
try:
    model.load_weights(WEIGHTS_FILE)
    print("Model output test:", model(np.zeros((1, 512, 512, 3))).numpy().mean())
    print(f"Successfully loaded trained weights from {WEIGHTS_FILE}.")
except tf.errors.NotFoundError:
    print(f"Error: Weights file '{WEIGHTS_FILE}' not found.")
    print("Please ensure you run tensorTrain.py first to generate the weights file.")
    exit() # Exit the script if weights aren't found

#new TEST
preprocess_input = sm.get_preprocessing(BACKBONE)


def predict_image(model, image_path, target_size=TARGET_SIZE, threshold=0.1):
    # 1. Load original image
    img_raw = tf.io.read_file(image_path)
    img_decoded = tf.image.decode_jpeg(img_raw, channels=3)
    original_img = img_decoded.numpy()

    # 2. Resize to the size the model expects
    img_resized = tf.image.resize(img_decoded, target_size).numpy()

    # 3. Preprocess for ResNet backbone
    img_preprocessed = preprocess_input(img_resized)
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    # 4. Predict corrosion mask
    pred_mask = model.predict(img_batch)[0, :, :, 0]
    print(f"\n--- Model Output Stats ---")
    print(f"Max confidence: {pred_mask.max():.4f}")
    print(f"Min confidence: {pred_mask.min():.4f}")
    print(f"Mean confidence: {pred_mask.mean():.4f}")
    corrosion_mask = (pred_mask > threshold).astype(np.uint8)

    # 5. Resize back to original image size
    corrosion_mask_resized = cv2.resize(
        corrosion_mask,
        (original_img.shape[1], original_img.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # 6. Post-processing to get rust and cracks
    rust_mask, crack_mask = detect_rust_and_cracks(original_img, corrosion_mask_resized)

    return original_img, corrosion_mask_resized, rust_mask, crack_mask


# --- Test Execution Block-----------------------------------------------------------

TESTING123 = "../Outcomes/Input/motor2.jpg"
TEST_IMAGE_PATH = "../Data/test/1_58_jpg.rf.926f79e868a36f37b8bbf79c3e4d4fa6.jpg"

if os.path.exists(TEST_IMAGE_PATH):
    print("\n--- Starting Inference ---")
    original_image, corrosion_mask, rust_mask, crack_mask = predict_image(model, TEST_IMAGE_PATH)


    print(f"Test Image: {os.path.basename(TEST_IMAGE_PATH)}")
    print(f"Predicted Crack Pixels: {np.sum(crack_mask)}")
    print(f"Predicted Rust Pixels: {np.sum(rust_mask)}")

    # --- Visualization ---
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 6))

    # Plot 1: Original Image
    plt.subplot(2, 4, 1)
    plt.imshow(original_image)
    plt.title("IMG 1")

    # Plot 2: Rust Detection (Overlaying green)
    plt.subplot(2, 4, 2)
    img_raw = tf.io.read_file(TEST_IMAGE_PATH)
    img_resized = tf.image.resize(tf.image.decode_jpeg(img_raw, channels=3), TARGET_SIZE)
    raw_pred = model.predict(np.expand_dims(preprocess_input(img_resized.numpy()), axis=0))[0, :, :, 0]
    plt.imshow(original_image)
    plt.imshow(cv2.resize(raw_pred, (original_image.shape[1], original_image.shape[0])), cmap='jet', alpha=0.5)
    plt.title("Raw AI Heatmap")

    # Plot 3: Rust Detection (Overlaying green)
    rust_overlay = original_image.copy()
    rust_overlay[rust_mask > 0] = [0, 255, 0]
    plt.subplot(2, 4, 3)
    plt.imshow(rust_overlay)
    plt.title("Rust Detected (Green)")

    # Plot 4: Crack Detection (Overlaying cyan)
    crack_overlay = original_image.copy()
    crack_overlay[crack_mask > 0] = [0, 255, 255]
    plt.subplot(2, 4, 4)
    plt.imshow(crack_overlay)
    plt.title("Cracks Detected (Cyan)")

    original_image, corrosion_mask, rust_mask, crack_mask = predict_image(model, TESTING123)

    # Plot 1: Original Image
    plt.subplot(2, 4, 5)
    plt.imshow(original_image)
    plt.title("IMG 2")

    # Plot 2: Rust Detection (Overlaying green)
    plt.subplot(2, 4, 6)
    img_raw = tf.io.read_file(TESTING123)
    img_resized = tf.image.resize(tf.image.decode_jpeg(img_raw, channels=3), TARGET_SIZE)
    raw_pred = model.predict(np.expand_dims(preprocess_input(img_resized.numpy()), axis=0))[0, :, :, 0]
    plt.imshow(original_image)
    plt.imshow(cv2.resize(raw_pred, (original_image.shape[1], original_image.shape[0])), cmap='jet', alpha=0.5)
    plt.title("Raw AI Heatmap")

    # Plot 3: Rust Detection (Overlaying green)
    rust_overlay = original_image.copy()
    rust_overlay[rust_mask > 0] = [0, 255, 0]
    plt.subplot(2, 4, 7)
    plt.imshow(rust_overlay)
    plt.title("Rust Detected (Green)")

    # Plot 4: Crack Detection (Overlaying cyan)
    crack_overlay = original_image.copy()
    crack_overlay[crack_mask > 0] = [0, 255, 255]
    plt.subplot(2, 4, 8)
    plt.imshow(crack_overlay)
    plt.title("Cracks Detected (Cyan)")

    plt.subplots_adjust(hspace=0.8, wspace=0.3)
    plt.show()
else:
    print(f"Error: Test image not found at {TEST_IMAGE_PATH}")