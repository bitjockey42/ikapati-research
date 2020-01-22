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


def prepare_dataset(data_dir, species, file_ext="JPG"):
    """ Prepare the dataset for saving """
    logger.info(f"preparing dataset for {species}")

    preprocessed_images = []
    labels = []
    data = {
        "Species": [],
        "Filename": [],
        "Disease": [],
    }

    folder_paths = get_folder_paths(data_dir, species)

    for label_id, folder_path in enumerate(folder_paths):
        species, disease = get_species_disease(folder_path)
        filenames = find_image_files(folder_path, file_ext=file_ext)
        for filename in filenames:
            data["Species"].append(species)
            data["Disease"].append(disease)
            data["Filename"].append(os.path.basename(filename))
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


def get_folder_paths(data_dir, species):
    return sorted(glob.glob(os.path.join(data_dir, f"{species}*/")))


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
    print('Data Type: %s' % pixels.dtype)
    print('BEFORE NORMALIZATION Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
    pixels = pixels.astype('float32')
    pixels /= 255.0
    print('AFTER NORMALIZATION Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
    return pixels


def standardize_pixel_values(pixels):
    """ Globally standardize pixel values to positive """
    mean, std = pixels.mean(), pixels.std()
    print('BEFORE STANDARDIZATION Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    pixels = (pixels - mean) / std
    pixels = np.clip(pixels, -1.0, 1.0)
    pixels = (pixels + 1.0) / 2.0
    mean, std = pixels.mean(), pixels.std()
    print('AFTER STANDARDIZATION Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    print('AFTER STANDARDIZATION Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
    return pixels


def save_dataset(data_dir, dataset):
    """ Save dataset as *.npy to """
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    for key, data in dataset.items():
        file_path = os.path.join(data_dir, f"{key}.npy")
        logger.info(f"Saving {key} to {file_path}")
        np.save(file_path, dataset[key])


@click.command()
@click.argument('data_dir', type=click.Path())
def main(data_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info('making final data set from raw data')
    #dataset = download_dataset()
    #prepared_dataset = prepare_dataset(dataset)
    #save_dataset(data_dir, prepared_dataset)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
