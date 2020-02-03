""" Use this to detect the plant disease by passing in a filename

Example:

python src/models/predict_model.py models/Corn.h5 --image-filename 1621.JPG --classes-filename data/processed/Corn/labels.json

This should output the disease (or "healthy")
"""

import os
import json
import click

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ikapati.data.io import read_metadata
from ikapati.data.image_processing import preprocess_image


def prepare_input_data(input_filename):
    image = preprocess_image(input_filename)
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


def load_tflite_model(model_dir):
    model_path = os.path.join(model_dir, "model.tflite")
    print(f"Loading {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


@click.command()
@click.argument("model_dir", type=click.Path())
@click.option("--input_filename", type=click.Path())
def main(model_dir, input_filename):
    # Load class names
    class_names = get_class_names(model_dir)

    # Load saved tflite model
    interpreter = load_tflite_model(model_dir)

    # Preprocess image
    input_data = prepare_input_data(input_filename)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make a prediction
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]["index"])
    label_id = get_label_id(predictions[0])

    # Show result
    print(f"Detected: {class_names[label_id]}")


if __name__ == "__main__":
    main()
