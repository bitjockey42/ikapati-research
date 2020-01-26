import os
import json
import pathlib
from uuid import uuid4
from datetime import datetime
from typing import List

import tensorflow as tf
import numpy as np

from src.data.image_processing import preprocess_image, image_example, normalize
from src.data import utils

AUTOTUNE = tf.data.experimental.AUTOTUNE

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def write_dataset(
    record_file_path: str,
    image_file_paths: List[str],
    class_names: List[str],
    file_ext: str = "JPG",
):
    total = len(image_file_paths)
    with tf.io.TFRecordWriter(record_file_path) as writer:
        for i, image_file_path in enumerate(image_file_paths):
            label, label_text = utils.get_label(image_file_path, class_names)
            example = image_example(image_file_path, label, label_text)
            writer.write(example.SerializeToString())
            utils.progress(i, total)


def write_metadata(
    metadata_file_path: str, dataset_map: dict, species: str, class_names: List[str]
) -> dict:
    current_datetime = datetime.utcnow()
    num_classes = len(class_names)
    file_counts = dict(
        [(key, len(file_paths)) for key, file_paths in dataset_map.items()]
    )

    metadata = {
        "id": str(uuid4()),
        "species": species,
        "num_classes": num_classes,
        "class_names": list(map(str, class_names)),
        "created_date": current_datetime.strftime("%Y-%m-%d %T%z"),
        "file_counts": file_counts,
    }

    with open(metadata_file_path, "w") as json_file:
        json.dump(metadata, json_file)

    return metadata


def read_dataset(
    record_file_path: str,
    batch_size: int,
    num_classes: int,
    num_parallel_calls=AUTOTUNE,
):
    dataset = tf.data.TFRecordDataset(record_file_path)
    dataset = dataset.map(parse_function(num_classes), num_parallel_calls)
    dataset = dataset.map(normalize)
    dataset = dataset.shuffle(batch_size + 3 * batch_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset


def read_metadata(metadata_file_path: str) -> dict:
    with open(metadata_file_path) as json_file:
        return json.load(json_file)


def parse_function(num_classes):
    def _parse_function(example_proto):
        # Define feature description
        image_feature_description = {
            "image_raw": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
            "label_text": tf.io.FixedLenFeature([], tf.string),
        }

        # Load one example
        parsed_features = tf.io.parse_single_example(
            example_proto, image_feature_description
        )

        # Parse image
        image = tf.image.decode_jpeg(parsed_features["image_raw"], channels=3)
        image = tf.reshape(image, [256, 256, 3])

        # Convert label
        label = tf.one_hot(parsed_features["label"], num_classes)

        return image, label

    return _parse_function
