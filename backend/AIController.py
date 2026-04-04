import os
#--Google Cloud--
from google.cloud import storage

YOLO_MODEL_PATH = "/tmp/best.pt"
TENSOR_MODEL_PATH ="/tmp/final_TF_model.h5"

API_KEY = os.environ.get("API_SECRET")

def download_model():
    client = storage.Client()
    bucket = client.bucket("rotor-ai-models")

    if not os.path.exists(YOLO_MODEL_PATH):
        print("Downloading YOLO model...")
        blob = bucket.blob("best.pt")
        blob.download_to_filename(YOLO_MODEL_PATH)

    if not os.path.exists(TENSOR_MODEL_PATH):
        print("Downloading Tensor model...")
        blob = bucket.blob("final_TF_model.h5")  # 🔁 match your uploaded file name
        blob.download_to_filename(TENSOR_MODEL_PATH)

    print("Model downloaded!")

download_model()


os.environ["SM_FRAMEWORK"] = "tf.keras"

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras import layers, models
import cv2
from PIL import Image
import segmentation_models as sm
from segmentation_models import Unet
from keras.layers import Lambda
import base64
from io import BytesIO
from ultralytics import YOLO

from Phase1.PostProcess import detect_rust_and_cracks
from Phase2.YOLOapi import detect_frame

def check_api_key():
    client_key = request.headers.get("x-api-key")
    if client_key != API_KEY:
        return False
    return True

#-- google cloud path
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_model.to("cpu")
WEIGHTS_PATH = TENSOR_MODEL_PATH
#--local path
# WEIGHTS_PATH = "Phase1/best_unet2_corrosion.h5"
# yolo_model = YOLO('yolo_corrosion/yolov8_corrosionV2/weights/best.pt')

app = Flask(__name__)
CORS(app, origins=["https://rotor-ai.vercel.app"])

def image_to_base64(img_np):
    img_pil = Image.fromarray(img_np)
    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def apply_clahe(img_np):
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)


TARGET_SIZE = (512, 512)
BACKBONE = "resnet50"
THRESHOLD = 0.1

preprocess_input = sm.get_preprocessing(BACKBONE)

base_model = Unet(
    backbone_name=BACKBONE,
    encoder_weights='imagenet',
    classes=1,
    activation='sigmoid',
    input_shape=TARGET_SIZE + (3,)
)

outputs = Lambda(lambda t: tf.cast(t, tf.float32))(base_model.output)

model = tf.keras.Model(
    inputs=base_model.input,
    outputs=outputs
)

model.load_weights(WEIGHTS_PATH)
print("loaded success")

# ---------------- API ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if not check_api_key():
        return jsonify({"error": "unauthorized"}), 401
    if "image" not in request.files:
        return jsonify({"error": "no images provided"}),400

    file = request.files["image"]

    image = Image.open(file).convert("RGB")
    original_img = np.array(image)

    resized_img = cv2.resize(original_img, TARGET_SIZE)
    preprocessed_img = preprocess_input(resized_img)
    img_batch = np.expand_dims(preprocessed_img, axis=0)

    pred_mask = model.predict(img_batch)[0, :, :, 0]
    corrosion_mask = (pred_mask > THRESHOLD).astype(np.uint8)

    corrosion_mask_resized = cv2.resize(
        corrosion_mask,
        (original_img.shape[1], original_img.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    rust_mask, crack_mask = detect_rust_and_cracks(
        original_img,
        corrosion_mask_resized
    )

    crack_mask = (crack_mask > 0).astype(np.uint8) * 255

    #currently black and white
    crack_overlay = original_img.copy()
    crack_overlay[crack_mask == 255] = [0, 255, 255]

    rust_overlay = original_img.copy()
    rust_overlay[rust_mask > 0] = [0, 255, 0]

    return jsonify({
        "rust_pixels": int(np.sum(rust_mask)),
        "crack_pixels": int(np.sum(crack_mask)),

        "original_image": image_to_base64(original_img),
        "corrosion_mask": image_to_base64(corrosion_mask_resized * 255),
        "crack_overlay": image_to_base64(crack_overlay),
        "rust_overlay": image_to_base64(rust_overlay),
    })

#------YOLO API
def decode_image(data_url):
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

@app.route("/detect", methods=["POST"])
def detect():
    if not check_api_key():
        return jsonify({"error": "unauthorized"}), 401

    data = request.json
    img = decode_image(data["image"])
    print("Image shape:", img.shape)
    detections = detect_frame(img,yolo_model)
    print("Detections:", detections)

    return jsonify(detections)

#this is for 3DGS
import uuid

def create_job_id():
    return f"job_{uuid.uuid4().hex[:8]}"

@app.route("/process-video",methods=["POST"])
def process_video():
    if not check_api_key():
        return jsonify({"error": "unauth"}),401
    if "video" not in request.files:
        return jsonify({"error":"no video provided"}),400

    file = request.files["video"]

    job_id = create_job_id()

    local_path = f"/tmp/{job_id}.mp4"
    file.save(local_path)

    client = storage.Client()
    bucket = client.bucket("rotor-ai-jobs")

    blob = bucket.blob(f"jobs/{job_id}/input.mp4")
    blob.upload_from_filename(local_path)

    return jsonify({
        "message": "video uploaded",
        "job_id": job_id
    })

# #processing
# @app.route("/run-job/<job_id>", methods=["POST"])
# def run_job(job_id):
#     if not check_api_key():
#         return jsonify({"error": "unauthorized"}), 401
#
#     client = storage.Client()
#     bucket = client.bucket("rotor-ai-jobs")
#
#     # Download video
#     blob = bucket.blob(f"jobs/{job_id}/input.mp4")
#     local_video = f"/tmp/{job_id}.mp4"
#     blob.download_to_filename(local_video)
#
#     # Create working dirs
#     base_path = f"/tmp/{job_id}"
#     frames_dir = f"{base_path}/images"
#     masks_dir = f"{base_path}/masks"
#
#     os.makedirs(frames_dir, exist_ok=True)
#     os.makedirs(masks_dir, exist_ok=True)
#
#     # 1. Extract frames
#     extract_frames(local_video, frames_dir, fps=3)
#
#     # 2. Run AI
#     process_frames(frames_dir, masks_dir)
#
#     return jsonify({"status": "frames + AI done"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)