import glob
import os
import numpy as np

from tensorflow import keras


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
