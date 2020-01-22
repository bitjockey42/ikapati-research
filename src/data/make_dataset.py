# -*- coding: utf-8 -*-
import click
import os
import logging
import glob
import random

from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from PIL import Image

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def prepare_dataset(data_dir, species=None, file_ext="JPG"):
    """ Prepare the dataset for saving """
    logger.info(f"preparing dataset for {species}")

    preprocessed_images = []
    labels = []
    data = {
        "Species": [],
        "Filename": [],
        "Disease": [],
        "Label": [],
    }

    folder_paths = get_folder_paths(data_dir, species)

    for label_id, folder_path in enumerate(folder_paths):
        species, disease = get_species_disease(folder_path)
        filenames = find_image_files(folder_path, file_ext=file_ext)
        for filename in filenames:
            data["Species"].append(species)
            data["Disease"].append(disease)
            data["Filename"].append(os.path.basename(filename))
            data["Label"].append(label_id)
            labels.append(label_id)
            preprocessed_image = preprocess_image(filename)
            preprocessed_images.append(preprocessed_image)

    classes = get_classes(labels)
    num_classes = len(classes)
    labels_enc = convert_to_one_hot_labels(labels, num_classes)

    train_data, test_data, train_labels_enc, test_labels_enc = train_test_split(preprocessed_images, labels_enc, test_size=0.2, random_state=13)
    train_data, eval_data, train_labels_enc, eval_labels_enc = train_test_split(train_data, train_labels_enc, test_size=0.2, random_state=13)

    return {
        "train_data": train_data,
        "eval_data": eval_data,
        "test_data": test_data,
        "train_labels": train_labels_enc,
        "eval_labels": eval_labels_enc,
        "test_labels": test_labels_enc,
    }


def get_folder_paths(data_dir, species=None):
    if species:
        return sorted(glob.glob(os.path.join(data_dir, f"{species}*/")))
    return sorted(os.listdir(data_dir))


def get_species_disease(folder_path):
    folder_name = os.path.basename(os.path.dirname(folder_path))
    species, disease = folder_name.split("___")
    return species, disease


def find_image_files(folder_path, file_ext="JPG"):
    """ Find image files with file_ext in data_dir """
    return glob.glob(os.path.join(folder_path, f"*.{file_ext}"))


def get_classes(labels):
    """ Get the classes """
    return np.unique(labels)


def convert_to_one_hot_labels(labels, num_classes):
    """ Convert labels (e.g. "2") to one hot encoding (e.g [0, 0, 1])"""
    return keras.utils.to_categorical(labels, num_classes=num_classes)


def preprocess_images(filenames):
    """ Load and preprocess images before feeding into CNN """
    return np.asarray(list(map(preprocess_image, filenames)))


def preprocess_image(filename, normalize=True, standardize=True):
    image = load_image(filename)
    pixel_values = convert_to_pixel_values(image, normalize, standardize)
    pixel_values.reshape(-1, image.size[0], image.size[1], 1)
    return pixel_values


def load_image(filename):
    """ Load image and optionaly split into different channels """
    image = Image.open(filename)
    logger.info(f"LOADED {filename}\n{image.format} {image.mode} {image.size}")
    return image


def get_image_filepath(data_dir, row):
    """ Get image filepath from row """
    return os.path.join(data_dir, f"{row.Species}___{row.Label}", row.Filename)


def convert_to_pixel_values(image, normalize, standardize):
    """ Convert image to array of pixel values """
    pixels = np.asarray(image)
    if normalize:
        pixels = normalize_pixel_values(pixels)
    if standardize:
        pixels = standardize_pixel_values(pixels)
    return pixels


def normalize_pixel_values(pixels):
    """ Normalize pixel values to be in the range [0, 1] """
    pixels = pixels.astype('float32')
    pixels /= 255.0
    return pixels


def standardize_pixel_values(pixels):
    """ Globally standardize pixel values to positive """
    mean, std = pixels.mean(), pixels.std()
    pixels = (pixels - mean) / std
    pixels = np.clip(pixels, -1.0, 1.0)
    pixels = (pixels + 1.0) / 2.0
    return pixels


def save_dataset(project_dir, dataset):
    """ Save dataset as *.npy to """
    output_dir = os.path.join(project_dir, "data", "processed")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for key, data in dataset.items():
        file_path = os.path.join(output_dir, f"{key}.npy")
        logger.info(f"Saving {key} to {file_path}")
        np.save(file_path, data)


@click.command()
@click.argument('data_dir', type=click.Path())
@click.argument('output_dir', type=click.Path())
@click.option('--species', help='The name of the plant species, e.g. Corn')
@click.option('--file-ext', default="JPG", help='The file extension of images, e.g. JPG')
def main(data_dir, output_dir, species, file_ext):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info('making final data set from raw data')
    dataset = prepare_dataset(data_dir, species, file_ext)
    save_dataset(project_dir, dataset)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
