"""
Utility functions for processing images
"""
import os

import tensorflow as tf
import numpy as np
from PIL import Image


def preprocess_image(filename, channels=3):
    image_raw = open(filename, "rb").read()
    image = tf.image.decode_jpeg(image_raw, channels=channels)
    image = tf.reshape(image, [image.shape[0], image.shape[1], channels])
    image, _ = normalize(image, None)
    return image


def load_image(filename):
    """ Load image and optionaly split into different channels """
    image = Image.open(filename)
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
    pixels = pixels.astype("float32")
    pixels /= 255.0
    return pixels


def standardize_pixel_values(pixels):
    """ Globally standardize pixel values to positive """
    mean, std = pixels.mean(), pixels.std()
    pixels = (pixels - mean) / std
    pixels = np.clip(pixels, -1.0, 1.0)
    pixels = (pixels + 1.0) / 2.0
    return pixels


def normalize(image, label):
    """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
    image = tf.cast(image, tf.float32) * (1.0 / 255) - 0.5
    return image, label


def image_example(file_path, label, label_text):
    """ Form TFExample obj to be added to a TFRecord """
    image_string = open(file_path, "rb").read()
    # Define the features of the tfrecord
    feature = {
        "image_raw": _bytes_feature(tf.compat.as_bytes(image_string)),
        "label": _int64_feature(int(label)),
        "label_text": _bytes_feature(tf.compat.as_bytes(label_text)),
    }
    # Serialize to string and write to file
    return tf.train.Example(features=tf.train.Features(feature=feature))


# Helper functions from the official documentation for TFData
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
