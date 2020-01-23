# -*- coding: utf-8 -*-
import click
import os
import logging
import glob
import random
import json

from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from PIL import Image

import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split

from src.data.image_processing import preprocess_image
from src.data import utils

logger = logging.getLogger(__name__)

DELIM = "___"


def prepare_dataset(data_dir, output_dir, species, file_ext="JPG"):
    logger.info(f"Preparing {species} dataset")
    folder_paths = utils.get_folder_paths(data_dir, species)
    labels = []

    for label_id, folder_path in enumerate(folder_paths):
        species, disease = utils.get_species_disease(folder_path)
        filenames = utils.find_image_files(folder_path)
        labels.append(disease)
        for file_id, filename in enumerate(filenames):
            image_data = preprocess_image(filename)
            dest_path = os.path.join(output_dir, f"{label_id}{DELIM}{file_id}.npy")
            np.save(dest_path, image_data)
    
    with open(os.path.join(output_dir, f"labels.txt"), "w") as labels_file:
        labels_file.write("\n".join(labels))


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

    if species is not None:
        output_dir = os.path.join(output_dir, species)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    prepare_dataset(data_dir, output_dir, species, file_ext)
    utils.split_dataset(output_dir)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
