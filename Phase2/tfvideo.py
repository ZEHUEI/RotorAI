import cv2
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from segmentation_models import Unet
from keras.layers import Lambda
from tensorflow.keras import layers
from Phase1.PostProcess import detect_rust_and_cracks
import os

os.environ["SM_FRAMEWORK"] = "tf.keras"

# --- MODEL SETUP ---
last_pred_original = None
TARGET_SIZE = (768, 768)
BACKBONE = 'resnet50'
NUM_TARGET_CLASSES = 1
THRESHOLD = 0.6

preprocess_input = sm.get_preprocessing(BACKBONE)

base_model = Unet(
    backbone_name=BACKBONE,
    encoder_weights=None,
    classes=NUM_TARGET_CLASSES,
    activation='sigmoid',
    input_shape=TARGET_SIZE + (3,)
)
outputs = Lambda(lambda t: tf.cast(t, tf.float32))(base_model.output)
model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

WEIGHTS_FILE = "../backend/final_TF_model.h5"
model.load_weights(WEIGHTS_FILE)
print(f"Loaded weights from {WEIGHTS_FILE}")

# --- FACE CASCADE ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

# --- PATHS ---
video_path = "../Outcomes/Input/IMG_1150.mp4"
output_path = "../Outcomes/Predictions/processed_video.mp4"

cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#motion blur
def apply_sharpen(img_np):
    # Unsharp mask — sharpens blurry frames
    gaussian = cv2.GaussianBlur(img_np, (0, 0), 2.0)
    sharpened = cv2.addWeighted(img_np, 1.5, gaussian, -0.5, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def apply_clahe(img_np):
    img_uint8 = np.clip(img_np, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)


def predict_frame(img_rgb):
    global last_pred_original
    """Run TF UNet inference on a single frame."""
    h_orig, w_orig = img_rgb.shape[:2]

    # Pad don't squash — match training preprocessing
    scale = min(TARGET_SIZE[0] / h_orig, TARGET_SIZE[1] / w_orig)
    new_h = min(int(np.floor(h_orig * scale)), TARGET_SIZE[0])
    new_w = min(int(np.floor(w_orig * scale)), TARGET_SIZE[1])
    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = TARGET_SIZE[0] - new_h
    pad_w = TARGET_SIZE[1] - new_w
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    img_padded = np.pad(img_resized, [[pad_top, pad_h - pad_top], [pad_left, pad_w - pad_left], [0, 0]])

    img_sharpened = apply_sharpen(img_padded)

    img_clahe = apply_clahe(img_sharpened)
    img_preprocessed = preprocess_input(img_clahe.astype(np.float32))
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    pred_mask = model.predict(img_batch, verbose=0)[0, :, :, 0]

    # Crop padding back out
    pred_cropped = pred_mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    pred_original = cv2.resize(pred_cropped, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    # Smooth between frames — weighted average with previous
    if last_pred_original is not None and last_pred_original.shape == pred_original.shape:
        pred_original = 0.6 * pred_original + 0.4 * last_pred_original
    last_pred_original = pred_original

    pred_max = pred_original.max()
    adaptive_threshold = max(0.35, pred_max * 0.6)
    corrosion_mask = (pred_original > adaptive_threshold).astype(np.uint8)
    return corrosion_mask


def process_roi(roi_mask, img_rgb, current_boxed_img, current_face_mask, w_img):
    final_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_not(current_face_mask))
    rust_mask, cracks_mask = detect_rust_and_cracks(img_rgb, final_mask)

    if np.any(rust_mask):
        merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
        rust_merged = cv2.dilate(rust_mask, merge_kernel, iterations=2)
        rust_merged = cv2.erode(rust_merged, merge_kernel, iterations=2)

        contours, _ = cv2.findContours(rust_merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            if rw > (w_img * 0.5) and (rw / float(rh + 1e-5)) > 5:
                continue
            cv2.rectangle(current_boxed_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(current_boxed_img, "Rust", (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if np.any(cracks_mask):
        contours, _ = cv2.findContours(cracks_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            if cw > (w_img * 0.5):
                continue
            cv2.rectangle(current_boxed_img, (cx, cy), (cx + cw, cy + ch), (255, 255, 0), 2)
            cv2.putText(current_boxed_img, "Crack", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


# --- MAIN VIDEO LOOP ---
frame_idx = 0
last_corrosion_mask = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_bgr = frame
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    boxed_image = image_bgr.copy()
    h_img, w_img = image_rgb.shape[:2]

    # --- FACE MASKING ---
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    face_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for (x, y, w, h) in faces:
        cv2.rectangle(face_mask, (x, y), (x + w, y + h), 255, -1)
    face_mask_dilated = cv2.dilate(face_mask, kernel, iterations=3)

    # --- TF PREDICTION (Every 3rd frame for speed) ---
    if frame_idx % 3 == 0:
        last_corrosion_mask = predict_frame(image_rgb)

    if last_corrosion_mask is not None:
        process_roi(last_corrosion_mask, image_rgb, boxed_image, face_mask_dilated, w_img)

    out.write(boxed_image)
    cv2.imshow("Rotor AI - Processing Video", cv2.resize(boxed_image, (w_img, h_img)))
    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video processing finished. Saved to: {output_path}")