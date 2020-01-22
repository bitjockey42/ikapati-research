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
from tensorflow import keras
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def download_dataset():
    """ Download data from fashion MNIST set """
    logger.info('downloading dataset')
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }


def find_image_files(data_dir, file_ext="JPG"):
    return glob.glob(os.path.join(data_dir, f"*.{file_ext}"))


def prepare_dataset(project_dir, data_dir, species):
    """ Prepare the dataset for saving """
    logger.info(f"preparing dataset for {species}")

    data_dirs = glob.glob(os.path.join(data_dir, f"{species}*"))
    mapping = create_mapping(data_dirs, os.path.join(project_dir, "data", "processed", species, "labels.csv"))

    classes = get_classes(dataset["train_labels"])
    num_classes = len(classes)
    logger.info('num_classes {}'.format(num_classes))
    
    train_data = preprocess_images(dataset["train_images"])
    test_data = preprocess_images(dataset["test_images"])
    
    train_labels_enc = convert_to_one_hot_labels(dataset["train_labels"], num_classes=num_classes)
    test_labels_enc = convert_to_one_hot_labels(dataset["test_labels"], num_classes=num_classes)
    
    train_data, eval_data, train_labels_enc, eval_labels_enc = train_test_split(train_data, train_labels_enc, test_size=0.2, random_state=13)
    
    return {
        "train_data": train_data,
        "eval_data": eval_data,
        "test_data": test_data,
        "train_labels": train_labels_enc,
        "eval_labels": eval_labels_enc, 
        "test_labels": test_labels_enc
    }


def create_mapping(data_dirs, filename):
    """ Create mapping of filenames, labels, and species from data dir names """
    dir_path = os.path.dirname(filename)
    logger.info(f"Creating mapping under {dir_path}")
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    data = {
        "Filename": [],
        "Label": [],
        "Species": [],
    }

    for data_dir in data_dirs:
        image_files = list(map(os.path.basename, find_image_files(data_dir)))
        species, label = data_dir.split("___")
        labels = [label for _ in range(len(image_files))]
        species = [species for _ in range(len(image_files))]
        
        data["Filename"].extend(image_files)
        data["Label"].extend(labels)
        data["Species"].extend(species)

    mapping = pd.DataFrame(data)
    logger.info(f"Saving CSV to {filename}")
    mapping.to_csv(filename, index=False)

    return mapping


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
