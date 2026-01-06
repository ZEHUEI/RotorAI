from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras import layers, models

app = Flask(__name__)

TARGET_SIZE = (512, 512)
TARGET_LABELS = ["Crack", "Rust"]
NUM_TARGET_CLASSES = len(TARGET_LABELS)

WEIGHTS_PATH = r"C:\Users\Ze Huei\PycharmProjects\RotorAI\Phase1\best_unet_weights.h5"

def unet_model(input_size=TARGET_SIZE + (3,), num_classes=NUM_TARGET_CLASSES):
    inputs = tf.keras.Input(input_size)

    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)

    u4 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(c4)

    u5 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(c5)

    return models.Model(inputs, outputs)

model = unet_model()
model.load_weights(WEIGHTS_PATH)

# ---------------- API ----------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    image = Image.open(file).convert("RGB")
    image = image.resize(TARGET_SIZE)
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]

    crack_mask = (prediction[:, :, 0] > 0.9).astype(np.uint8)
    rust_mask  = (prediction[:, :, 1] > 0.9).astype(np.uint8)

    return jsonify({
        "crack_pixels": int(np.sum(crack_mask)),
        "rust_pixels": int(np.sum(rust_mask))
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3001, debug=True)