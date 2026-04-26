"""
Monte Carlo Testing for Corrosion Segmentation Model
"""

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import cv2
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from segmentation_models import Unet
from keras.layers import Lambda

from Phase1.PostProcess import detect_rust_and_cracks

# ================= CONFIG =================
TARGET_SIZE = (768, 768)
BACKBONE = 'resnet50'
NUM_CLASSES = 1
WEIGHTS_FILE = "../backend/final_TF_model.h5"
TEST_IMAGE = "../Outcomes/Input/IMG_1341.jpg"
MC_RUNS = 50

# ================= MODEL =================
base_model = Unet(
    backbone_name=BACKBONE,
    encoder_weights=None,
    classes=NUM_CLASSES,
    activation='sigmoid',
    input_shape=TARGET_SIZE + (3,)
)

outputs = Lambda(lambda t: tf.cast(t, tf.float32))(base_model.output)
model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

model.load_weights(WEIGHTS_FILE)
print("✅ Model loaded")

preprocess_input = sm.get_preprocessing(BACKBONE)

# ================= AUGMENTATIONS =================
def augment_image(img):
    img = img.copy()

    # brightness
    if np.random.rand() < 0.5:
        factor = np.random.uniform(0.7, 1.3)
        img = np.clip(img * factor, 0, 255).astype(np.uint8)

    # gaussian noise
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, 10, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # blur
    if np.random.rand() < 0.3:
        k = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img


def apply_clahe_random(img):
    clip = np.random.uniform(1.0, 3.0)
    grid = np.random.choice([4, 8, 16])

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

# ================= CORE PIPELINE =================
def predict_from_array(model, img, threshold):
    original_img = img.copy()

    # resize + pad (same as training)
    h, w = img.shape[:2]
    scale = min(TARGET_SIZE[0] / h, TARGET_SIZE[1] / w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    img_resized = cv2.resize(img, (new_w, new_h))

    pad_h = TARGET_SIZE[0] - new_h
    pad_w = TARGET_SIZE[1] - new_w

    pad_top = pad_h // 2
    pad_left = pad_w // 2

    img_padded = np.pad(
        img_resized,
        [[pad_top, pad_h - pad_top],
         [pad_left, pad_w - pad_left],
         [0, 0]]
    )

    # CLAHE (randomized)
    img_clahe = apply_clahe_random(img_padded)

    # preprocess
    img_pre = preprocess_input(img_clahe.astype(np.float32))
    img_batch = np.expand_dims(img_pre, axis=0)

    pred_mask = model.predict(img_batch, verbose=0)[0, :, :, 0]

    # crop back
    pred_cropped = pred_mask[
        pad_top:pad_top + new_h,
        pad_left:pad_left + new_w
    ]

    pred_original = cv2.resize(
        pred_cropped,
        (original_img.shape[1], original_img.shape[0])
    )

    corrosion_mask = (pred_original > threshold).astype(np.uint8)

    rust_mask, crack_mask = detect_rust_and_cracks(original_img, corrosion_mask)

    return corrosion_mask, rust_mask, crack_mask, pred_original


# ================= MONTE CARLO =================
def monte_carlo_test(model, image_path, runs=50):
    base_img = cv2.imread(image_path)
    base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

    results = []
    pred_stack = []

    for i in range(runs):
        img_aug = augment_image(base_img)
        threshold = np.random.uniform(0.5, 0.8)

        corrosion_mask, rust_mask, crack_mask, raw_pred = predict_from_array(
            model,
            img_aug,
            threshold
        )

        pred_stack.append(raw_pred)

        results.append({
            "corrosion_pixels": np.sum(corrosion_mask),
            "rust_pixels": np.sum(rust_mask),
            "crack_pixels": np.sum(crack_mask),
            "mean_conf": np.mean(raw_pred)
        })

        print(f"Run {i+1}/{runs} done")

    pred_stack = np.array(pred_stack)

    return results, pred_stack


# ================= ANALYSIS =================
def analyze_results(results):
    corrosion = [r["corrosion_pixels"] for r in results]
    rust = [r["rust_pixels"] for r in results]
    crack = [r["crack_pixels"] for r in results]

    print("\n===== MONTE CARLO RESULTS =====")
    print(f"Corrosion Pixels: mean={np.mean(corrosion):.2f}, std={np.std(corrosion):.2f}")
    print(f"Rust Pixels: mean={np.mean(rust):.2f}, std={np.std(rust):.2f}")
    print(f"Crack Pixels: mean={np.mean(crack):.2f}, std={np.std(crack):.2f}")


def visualize_uncertainty(pred_stack):
    import matplotlib.pyplot as plt

    mean_map = np.mean(pred_stack, axis=0)
    std_map = np.std(pred_stack, axis=0)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Mean Prediction")
    plt.imshow(mean_map, cmap='jet')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Uncertainty (Std Dev)")
    plt.imshow(std_map, cmap='hot')
    plt.colorbar()

    plt.show()


# ================= RUN =================
if __name__ == "__main__":
    if not os.path.exists(TEST_IMAGE):
        print("❌ Test image not found")
        exit()

    results, pred_stack = monte_carlo_test(model, TEST_IMAGE, MC_RUNS)

    analyze_results(results)
    visualize_uncertainty(pred_stack)