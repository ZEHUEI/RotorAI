import os

from Phase2.YOLOapi import detect_frame
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

app = Flask(__name__)
CORS(app)

def image_to_base64(img_np):
    img_pil = Image.fromarray(img_np)
    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


TARGET_SIZE = (512, 512)
BACKBONE = "resnet50"
THRESHOLD = 0.275

WEIGHTS_PATH = r"/Phase1/Ignore/best_unet3_corrosion.h5"
yolo_model = YOLO('../yolo_corrosion/yolov8_corrosionV2/weights/best.pt')

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
    if "image" not in request.files:
        return jsonify({"error": "no images provided"}),400

    file = request.files["image"]

    image = Image.open(file).convert("RGB")
    original_img = np.array(image)

    resized_img = cv2.resize(original_img, TARGET_SIZE)
    preprocessed_img = preprocess_input(resized_img)
    img_batch = np.expand_dims(preprocessed_img, axis=0)

    pred_mask = model.predict(img_batch)[0, :, :, 0]
    pred_mask = (pred_mask > THRESHOLD).astype(np.uint8)
    # corrosion_mask = (pred_mask > THRESHOLD).astype(np.uint8)

    corrosion_mask_resized = cv2.resize(
        pred_mask,
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
    data = request.json
    img = decode_image(data["image"])
    print("Image shape:", img.shape)
    detections = detect_frame(img)
    print("Detections:", detections)

    return jsonify(detections)

#3D-Gaussian we expect video only mp4
@app.route("/gaussian",methods=["POST"])
def gaussian():
    #request ddata which is video, mp4

    #decode it to like 3 frames a sec then make it and scan for rust / cracks like predict

    #Get 3D model and maybe toggle rust/cracks or smt

    #maybe return to interactive moving where u can rotate or smt but return 0 for now
    return 0

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3001, debug=True)