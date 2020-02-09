""" Use this to detect the plant disease by passing in a filename

Example:

python src/models/predict_model.py models/Corn.h5 --image-filename 1621.JPG --classes-filename data/processed/Corn/labels.json

This should output the disease (or "healthy")
"""

import argparse
import os
import json
import flask

import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
from tensorflow import keras

from ikapati.data.io import read_metadata, NumpyEncoder
from ikapati.data.image_processing import preprocess_image, preprocess_raw_image

app = flask.Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict_disease():
    image_file = flask.request.files["image"]
    input_data = prepare_input_data(image_file)
    
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]["index"])
 
    label_id = get_label_id(predictions[0])
    probability = get_probability(predictions[0])

    data = {
        "class_name": class_names[label_id],
    }

    return flask.jsonify(data)


def prepare_input_data(image_file):
    image_bytes = image_file.read()
    im = Image.open(BytesIO(image_bytes))
    resized_im = im.resize((256,256))
    image_raw = resized_im.tobytes()
    image = preprocess_raw_image(image_raw)

    # This is done because tf expects the input to be in a list
    input_data = tf.reshape(image, [1, 256, 256, 3])
    return input_data


def get_class_names(model_dir):
    metadata_file_path = os.path.join(model_dir, "metadata.json")
    metadata = read_metadata(metadata_file_path)
    species = metadata["dataset"]["species"]
    print(f"{species}")
    return metadata["dataset"]["class_names"]


def get_label_id(prediction):
    return np.argmax(prediction)


def get_probability(prediction):
    return np.max(prediction)


def load_tflite_model(model_dir):
    model_path = os.path.join(model_dir, "model.tflite")
    print(f"Loading {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=5000)
    parser.add_argument("--model_dir")
    
    args = parser.parse_args()
    
    # Load class names
    class_names = get_class_names(args.model_dir)
    
    # Load saved tflite model
    interpreter = load_tflite_model(args.model_dir)
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    app.run(host=args.host, debug=True, port=args.port)
