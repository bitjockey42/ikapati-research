import os
import pathlib
import shutil
import tempfile
import argparse

import tensorflow as tf
import numpy as np

from ikapati.visualization import visualize as vz
from ikapati.data import io


def evaluate_model(model_dir, data_dir):
    model_file_path = model_dir.joinpath("final.h5")
    metadata_file_path = model_dir.joinpath("metadata.json")
    metadata = io.read_metadata(str(metadata_file_path))
    batch_size = metadata["arguments"]["batch_size"]
    
    test_data_path = data_dir.joinpath("test.tfrecord")
    test_dataset = io.read_dataset(str(test_data_path), batch_size, metadata["dataset"]["num_classes"])
    
    steps = metadata["dataset"]["file_counts"]["test"] // batch_size
    
    model = tf.keras.models.load_model(str(model_file_path))
    
    result = model.evaluate(test_dataset, steps=steps)
    
    return model, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--data_dir", type=str)
    
    args = parser.parse_args()
    
    model_dir = pathlib.Path(args.model_dir)
    data_dir = pathlib.Path(args.data_dir)
    
    model, result = evaluate_model(model_dir, data_dir)
    
    print(dict(zip(model.metrics_names, result)))
    
    model = None
