from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model("unet3_crack_rust_dacl10k_weights.weights.h5")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    image = Image.open(file).resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = prediction.tolist()

    return jsonify({"prediction": result})
