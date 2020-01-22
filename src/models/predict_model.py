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


@click.command()
@click.argument('model_filename', type=click.Path())
@click.option('--image-filename', type=click.Path())
@click.option('--classes-filename', type=click.Path())
def main(model_filename, image_filename, classes_filename):
    classes = get_classes(classes_filename)
    predictor = load_model(model_filename)
    preprocessed_image = preprocess_image(image_filename, normalize=True, standardize=True)
    predictions = predictor.predict(np.array([preprocessed_image]))
    prediction = predictions[0]
    print(classes[get_label_id(prediction)])


if __name__ == '__main__':
    main()
