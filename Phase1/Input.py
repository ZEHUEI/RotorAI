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

TARGET_SIZE = (768, 768)
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
WEIGHTS_FILE = "best_unet9_with_faces_corrosion.h5"
try:
    model.load_weights(WEIGHTS_FILE)
    print("Model output test:", model(np.zeros((1, 768, 768, 3))).numpy().mean())
    print(f"Successfully loaded trained weights from {WEIGHTS_FILE}.")
except tf.errors.NotFoundError:
    print(f"Error: Weights file '{WEIGHTS_FILE}' not found.")
    print("Please ensure you run tensorTrain.py first to generate the weights file.")
    exit() # Exit the script if weights aren't found

#new TEST
preprocess_input = sm.get_preprocessing(BACKBONE)

def apply_clahe(img_np):
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def predict_image(model, image_path, target_size=TARGET_SIZE, threshold=0.2):
    img_raw = tf.io.read_file(image_path)
    img_decoded = tf.image.decode_jpeg(img_raw, channels=3)
    original_img = img_decoded.numpy()

    img_resized = tf.image.resize(img_decoded, target_size).numpy()

    #-------
    #Clahe Method
    #-------
    # img_resized = tf.image.resize(img_decoded, target_size)
    # img_resized_uint8 = tf.cast(img_resized, tf.uint8).numpy()
    # img_clahe = apply_clahe(img_resized_uint8)
    # img_preprocessed = preprocess_input(img_clahe.astype(np.float32))

    img_preprocessed = preprocess_input(img_resized.astype(np.float32))
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    pred_mask = model.predict(img_batch)[0, :, :, 0]
    corrosion_mask = (pred_mask > threshold).astype(np.uint8)

    corrosion_mask_resized = cv2.resize(
        corrosion_mask,
        (original_img.shape[1], original_img.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    rust_mask, crack_mask = detect_rust_and_cracks(original_img, corrosion_mask_resized)

    # Return pred_mask too so visualization reuses it
    return original_img, corrosion_mask_resized, rust_mask, crack_mask, pred_mask


# --- Test Execution Block-----------------------------------------------------------

TESTING123 = "../Outcomes/Input/motor2.jpg"
TEST_IMAGE_PATH = "../Data/test/1_58_jpg.rf.926f79e868a36f37b8bbf79c3e4d4fa6.jpg"

if os.path.exists(TESTING123):
    print("\n--- Starting Inference ---")
    original_image, corrosion_mask, rust_mask, crack_mask,raw_pred_1  = predict_image(model, TESTING123)


    print(f"Test Image: {os.path.basename(TESTING123)}")
    print(f"Predicted Crack Pixels: {np.sum(crack_mask)}")
    print(f"Predicted Rust Pixels: {np.sum(rust_mask)}")

    # --- Visualization ---
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 6))

    # ori
    plt.subplot(2, 4, 1)
    plt.imshow(original_image)
    plt.title("IMG")

    # heatmap
    plt.subplot(2, 4, 2)
    plt.imshow(original_image)
    heatmap_1 = cv2.resize(raw_pred_1, (original_image.shape[1], original_image.shape[0]))
    plt.imshow(heatmap_1, cmap='jet',alpha=0.5)
    plt.colorbar(label='Confidence Score')
    plt.title("Raw AI Heatmap (no clache)")

    plt.subplots_adjust(hspace=0.8, wspace=0.3)
    plt.show()
else:
    print(f"Error: Test image not found at {TESTING123}")