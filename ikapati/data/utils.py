import sys
import glob
import os
import numpy as np
import pathlib
import sys

import tensorflow as tf

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from tensorflow import keras

DELIM = "___"


def progress(count, total, status=""):
    # From https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)

    sys.stdout.write("[%s] %s%s ...%s \r" % (bar, percents, "%", status))
    sys.stdout.flush()


def get_folder_paths(data_dir_path: str, species: str) -> List[pathlib.Path]:
    """ Get folder paths that match the species """
    data_dir = pathlib.Path(data_dir_path)
    return list(data_dir.glob(f"{species}*/"))


def get_image_paths(
    data_dir_path: str, species: str, file_ext: str = "JPG"
) -> List[str]:
    """ Get image file paths that match species and file extension """
    data_dir = pathlib.Path(data_dir_path)
    return list(map(str, data_dir.glob(f"{species}*/*.{file_ext}")))


def get_label(file_path: str, class_names: List[str]) -> Tuple[List[int], str]:
    """ Get label and label text from file path """
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    label_text = parts[-2]
    label = np.where(label_text == class_names)[0]
    return label, label_text.numpy()


def get_class_names(data_dir_path: str, species: str) -> List[str]:
    """ Get the class names from the data dir paths """
    data_dir = pathlib.Path(data_dir_path)
    return np.array(sorted([item.name for item in data_dir.glob(f"{species}*")]))


def convert_to_one_hot_labels(labels, num_classes):
    """ Convert labels (e.g. "2") to one hot encoding (e.g [0, 0, 1])"""
    return keras.utils.to_categorical(labels, num_classes=num_classes)


def get_dataset_map(image_file_paths: List[str], class_names: List[str]) -> dict:
    labels = [get_label(file_path, class_names)[0] for file_path in image_file_paths]
    train_files, test_files, train_labels, test_labels = train_test_split(
        image_file_paths, labels, test_size=0.2, random_state=13
    )
    test_files, eval_files, test_labels, eval_labels = train_test_split(
        test_files, test_labels, test_size=0.5, random_state=13
    )
    return {
        "train": train_files,
        "test": test_files,
        "eval": eval_files,
    }