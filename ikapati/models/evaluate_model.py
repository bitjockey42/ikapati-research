import os
import pathlib
import shutil
import tempfile
import argparse
import csv

import tensorflow as tf
import numpy as np

from ikapati.visualization import visualize as vz
from ikapati.data import io


def load_metadata(model_dir):
    metadata_file_path = model_dir.joinpath("metadata.json")
    metadata = io.read_metadata(str(metadata_file_path))
    return metadata


def evaluate_model(model_file_path, data_dir, metadata):
    batch_size = metadata["arguments"]["batch_size"]
    
    test_data_path = data_dir.joinpath("test.tfrecord")
    test_dataset = io.read_dataset(str(test_data_path), batch_size, metadata["dataset"]["num_classes"])
    
    steps = metadata["dataset"]["file_counts"]["test"] // batch_size
    model = tf.keras.models.load_model(str(model_file_path))
    result = model.evaluate(test_dataset, steps=steps)
    
    return model, result


def write_evaluation_to_file(file_path, model_file_path, results, metadata):
    file_exists = file_path.exists()
    
    evaluation = {
        "id": metadata["id"],
        "model_file_path": str(model_file_path),
        **metadata["arguments"],
        **results,
    }
    
    header = list(evaluation.keys())

    with open(str(file_path), "a+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)

        if not file_exists:
            writer.writeheader()

        writer.writerow(evaluation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--filename", type=str)
    
    args = parser.parse_args()
    
    model_dir = pathlib.Path(args.model_dir)
    data_dir = pathlib.Path(args.data_dir)
    file_path = pathlib.Path(args.filename)
    model_file_path = model_dir.joinpath("final.h5")
    
    metadata = load_metadata(model_dir)
    model, result = evaluate_model(model_file_path, data_dir, metadata)
    
    results = dict(zip(model.metrics_names, result))
    print(results)
    
    write_evaluation_to_file(file_path, model_file_path, results, metadata)
    
    model = None
