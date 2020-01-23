import sys
import glob
import os
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow import keras

DELIM = "___"


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


def save_dataset(output_dir, dataset):
    """ Save dataset as *.npy to """
    for key, data in dataset.items():
        file_path = os.path.join(output_dir, f"{key}.npy")
        np.save(file_path, data)


def split_dataset(output_dir):
    """ Split dataset into train, eval, and test sets """
    file_paths = sorted(glob.glob(os.path.join(output_dir, f"**{DELIM}**.npy")))
    labels = [get_label_id(file_path) for file_path in file_paths]
    num_classes = len(get_classes(labels))
    train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=13)
    train_files, eval_files, train_labels, eval_labels = train_test_split(train_files, train_labels, test_size=0.2, random_state=13)

    train_data = [np.load(train_file) for train_file in train_files]
    np.save(os.path.join(output_dir, "train_data.npy"), train_data)
    np.save(os.path.join(output_dir, "train_labels.npy"), convert_to_one_hot_labels(train_labels, num_classes=num_classes))
    train_data = train_labels = None

    eval_data = [np.load(eval_file) for eval_file in eval_files]
    np.save(os.path.join(output_dir, "eval_data.npy"), eval_data)
    np.save(os.path.join(output_dir, "eval_labels.npy"), convert_to_one_hot_labels(eval_labels, num_classes=num_classes))
    eval_data = eval_labels = None

    test_data = [np.load(test_file) for test_file in test_files]
    np.save(os.path.join(output_dir, "test_data.npy"), test_data)
    np.save(os.path.join(output_dir, "test_labels.npy"), convert_to_one_hot_labels(test_labels, num_classes=num_classes))
    test_data = test_labels = None


def get_label_id(file_path):
    label_id, _ = os.path.basename(file_path).split(DELIM)
    return int(label_id)
