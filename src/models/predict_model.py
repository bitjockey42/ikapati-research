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

from src.data.image_processing import preprocess_image


def get_labels(filename):
    with open(filename, "r") as f:
        labels = f.readlines()
        return list(map(str.strip, labels))


def get_label_id(prediction):
    return np.argmax(prediction)


def load_tflite_model(filename):
    interpreter = tf.lite.Interpreter(model_path=filename)
    interpreter.allocate_tensors()
    return interpreter


@click.command()
@click.argument("model_filename", type=click.Path())
@click.option("--input-filename", type=click.Path())
@click.option("--labels-filename", type=click.Path())
def main(model_filename, input_filename, labels_filename):
    # Load labels
    labels = get_labels(labels_filename)

    # Load saved tflite model
    interpreter = load_tflite_model(model_filename)

    # Preprocess image
    input_data = preprocess_image(input_filename, normalize=True, standardize=True)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make a prediction
    interpreter.set_tensor(input_details[0]["index"], np.array([input_data]))
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]["index"])
    label_id = get_label_id(predictions[0])

    # Show result
    print(f"Detected: {labels[label_id]}")


if __name__ == "__main__":
    main()
