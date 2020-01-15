# -*- coding: utf-8 -*-
import click
import os
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

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


def prepare_dataset(dataset):
    """ Prepare the dataset for saving """
    logger.info('preparing dataset')
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


def get_classes(labels):
    """ Get the classes """
    return np.unique(labels)


def preprocess_images(images):
    """ Scale the values of the images to a range of 0 to 1 before feeding into CNN """
    # reshape each image from 28,28 to a 28,28,1
    logger.info('preprocessing images')
    preprocessed_images = images.reshape(-1, 28,28, 1)
    # convert type to float32
    preprocessed_images = preprocessed_images.astype('float32')
    return preprocessed_images / 255


def convert_to_one_hot_labels(labels, num_classes):
    """ Convert labels (e.g. "2") to one hot encoding (e.g [0, 0, 1])"""
    return keras.utils.to_categorical(labels, num_classes=num_classes)


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
    dataset = download_dataset()
    prepared_dataset = prepare_dataset(dataset)
    save_dataset(data_dir, prepared_dataset)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
