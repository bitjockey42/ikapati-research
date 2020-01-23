""" Use this to detect the plant disease by passing in a filename

Example:

python src/models/predict_model.py models/Corn.h5 --image-filename 1621.JPG --classes-filename data/processed/Corn/labels.json

This should output the disease (or "healthy")
"""

import os
import json
import click

import numpy as np
from tensorflow import keras

from src.data.image_processing import preprocess_image


def get_classes(classes_filename):
    with open(classes_filename) as f:
        classes = json.load(f)
    return classes


def get_label_id(prediction):
    label_id = np.argmax(prediction)
    return str(label_id)


def load_model(filename):
    return keras.models.load_model(filename)


def predict(predictor, input_data, classes):
    predictions = predictor.predict(np.array([input_data]))
    prediction = predictions[0]
    return classes[get_label_id(prediction)]


@click.command()
@click.argument('model_filename', type=click.Path())
@click.option('--image-filename', type=click.Path())
@click.option('--classes-filename', type=click.Path())
def main(model_filename, image_filename, classes_filename):
    classes = get_classes(classes_filename)
    predictor = load_model(model_filename)
    input_data = preprocess_image(image_filename, normalize=True, standardize=True)
    prediction = predict(predictor, input_data, classes)
    print(f"CLASS: {prediction}")


if __name__ == '__main__':
    main()
