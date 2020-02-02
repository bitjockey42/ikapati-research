"""
Script to train the model both locally and on AWS SageMaker: 
https://sagemaker.readthedocs.io/en/stable/using_tf.html#prepare-a-script-mode-training-script

If you want to train using your GPU, set the CUDA_VISIBLE_DEVICES to 0.

export CUDA_VISIBLE_DEVICES=0
python src/models/train_model.py --train path/to/where/your/data/is
"""

import argparse
import os
import numpy as np
import json
import pathlib
import shutil
import tempfile

from datetime import datetime
from uuid import uuid4

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import (
    layers,
    models,
    activations,
    losses,
    optimizers,
    callbacks,
)

from ikapati.data import io
from ikapati.models import architectures


def model(num_classes, architecture="alexnet", learning_rate=0.001, activation="linear", padding="same"):
    """ Select an architecture and create a network from that """
    if architecture == "alexnet":
        model = architectures.alexnet.model(num_classes, learning_rate=learning_rate, activation=activation, padding=padding)
    else:
        raise NotImplementedError(f"{architecture} not implemented")

    model.compile(
        loss=losses.categorical_crossentropy,
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    model.summary()

    return model


def _get_callbacks(start_time, logdir, monitor, model_dir, model_id):
    return [
        tfdocs.modeling.EpochDots(),
        keras.callbacks.EarlyStopping(
            # Stop training when monitor (e.g. `val_loss`) is no longer improving
            monitor=monitor,
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=1e-2,
            # "no longer improving" being further defined as "for at least 3 epochs"
            patience=5,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, model_id, start_time, "{epoch}.h5"),
            save_best_only=True,
            monitor=monitor,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(logdir),
    ]


def load_dataset(data_dir, dataset_name, batch_size, num_classes):
    record_file_path = os.path.join(data_dir, f"{dataset_name}.tfrecord")
    return io.read_dataset(record_file_path, batch_size, num_classes)


def save_model(model, model_dir, model_id, start_time):
    model_filepath = os.path.join(model_dir, model_id, start_time, "final.h5")
    print(f"Saving model {model_filepath}")
    model.save(model_filepath)

    # Save as tflite model as well
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(
        os.path.join(model_dir, model_id, start_time, "model.tflite"), "wb"
    ) as converted_model_file:
        converted_model_file.write(tflite_model)


def train(
    architecture,
    train_dir,
    model_dir,
    batch_size,
    epochs,
    monitor,
    start_time,
    learning_rate=0.001,
    activation="linear",
    early_stopping=False,
):

    # Set up logs
    logdir = pathlib.Path(tempfile.mkdtemp()) / "tensorboard_logs"
    shutil.rmtree(logdir, ignore_errors=True)

    # Load metadata
    metadata_file_path = os.path.join(train_dir, "metadata.json")
    metadata = io.read_metadata(metadata_file_path)
    model_id = metadata["id"]
    num_classes = metadata["num_classes"]

    # Create directories
    model_dir_path = os.path.join(model_dir, model_id, start_time)
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)

    # Write to log
    with open(os.path.join(model_dir, model_id, "training.log"), "a+") as log_file:
        log_file.write(f"{activation}\t{model_dir_path}\n")

    # Load data
    train_dataset = load_dataset(train_dir, "train", batch_size, num_classes)
    eval_dataset = load_dataset(train_dir, "eval", batch_size, num_classes)

    # Create the model
    classifier = model(num_classes, architecture=architecture, learning_rate=learning_rate, activation=activation)

    # Steps
    steps = metadata["file_counts"]["train"] // batch_size
    validation_steps = metadata["file_counts"]["eval"] // batch_size

    callbacks = None
    if early_stopping:
        callbacks = _get_callbacks(start_time, logdir, monitor, model_dir, model_id)

    # Train
    history = classifier.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps,
        validation_data=eval_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    write_metadata(
        architecture=architecture,
        model_dir=model_dir,
        model_id=model_id,
        batch_size=batch_size,
        epochs=epochs,
        monitor=monitor,
        dataset_metadata=metadata,
        history=history,
        start_time=start_time,
        activation=activation,
        early_stopping=early_stopping,
    )

    return classifier, history, model_id


def write_metadata(
    architecture,
    model_dir,
    model_id,
    batch_size,
    epochs,
    monitor,
    dataset_metadata,
    history,
    start_time,
    activation,
    early_stopping,
):
    print("Write metadata for model")
    metadata_file_path = os.path.join(model_dir, model_id, start_time, "metadata.json")

    metadata = {
        "id": model_id,
        "start_time": start_time,
        "arguments": {
            "batch_size": batch_size,
            "epochs": epochs,
            "monitor": monitor,
            "activation": activation,
            "early_stopping": early_stopping,
            "architecture": architecture,
        },
        "dataset": dataset_metadata,
        "history": history.history,
    }

    with open(metadata_file_path, "w") as json_file:
        json.dump(metadata, json_file, cls=io.NumpyEncoder)

    return metadata


def _parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    # The activation function to try, e.g "linear"
    parser.add_argument("--activation", type=str)

    # Whether to stop when the monitor reaches a threshold
    parser.add_argument("--early_stopping", action="store_true", default=False)

    # Specify neural network architecture to use
    parser.add_argument("--architecture", type=str, default="alexnet")

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument("--model_dir", type=str)
    parser.add_argument(
        "--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument(
        "--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING")
    )

    # https://github.com/aws/sagemaker-containers#sm-hosts
    parser.add_argument(
        "--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS"))
    )
    parser.add_argument(
        "--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST")
    )
    parser.add_argument("--eval", type=str, default=os.environ.get("SM_CHANNEL_EVAL"))
    parser.add_argument("--monitor", type=str, default="val_loss")

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()
    start_time = datetime.utcnow().strftime("%Y-%m-%d__%H_%M%S")

    classifier, history, model_id = train(
        architecture=args.architecture,
        train_dir=args.train,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        monitor=args.monitor,
        start_time=start_time,
        learning_rate=args.learning_rate,
        activation=args.activation,
        early_stopping=args.early_stopping,
    )

    # save model
    save_model(classifier, args.model_dir, model_id, start_time)