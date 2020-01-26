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

from datetime import datetime
from uuid import uuid4

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

from src.data import io


def model(num_classes):
    """Generate a simple model"""
    model = keras.Sequential(
        [
            layers.Conv2D(
                32,
                kernel_size=(3, 3),
                activation="linear",
                input_shape=(256, 256, 3),
                padding="same",
            ),
            layers.LeakyReLU(alpha=0.1),
            layers.MaxPooling2D((2, 2), padding="same"),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation="linear", padding="same"),
            layers.LeakyReLU(alpha=0.1),
            layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
            layers.Dropout(0.25),
            layers.Conv2D(128, (3, 3), activation="linear", padding="same"),
            layers.LeakyReLU(alpha=0.1),
            layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
            layers.Dropout(0.4),
            layers.Flatten(),
            layers.Dense(128, activation="linear"),
            layers.LeakyReLU(alpha=0.1),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        loss=losses.categorical_crossentropy,
        optimizer=optimizers.Adam(),
        metrics=["accuracy"],
    )

    model.summary()

    return model


def _get_callbacks(monitor, model_dir, model_id):
    return [
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
            filepath=os.path.join(model_dir, model_id, "{epoch}.h5"),
            save_best_only=True,
            monitor=monitor,
            verbose=1,
        ),
    ]


def load_dataset(data_dir, dataset_name, batch_size, num_classes):
    record_file_path = os.path.join(data_dir, f"{dataset_name}.tfrecord")
    return io.read_dataset(record_file_path, batch_size, num_classes)


def save_model(model, model_dir, model_id):
    print(f"Saving model {model_id}")
    model_filepath = os.path.join(model_dir, model_id, "final.h5")
    model.save(model_filepath)

    # Save as tflite model as well
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(
        os.path.join(model_dir, model_id, "model.tflite"), "wb"
    ) as converted_model_file:
        converted_model_file.write(tflite_model)


def train(train_dir, model_dir, batch_size, epochs, monitor):
    # Load metadata
    metadata_file_path = os.path.join(train_dir, "metadata.json")
    metadata = io.read_metadata(metadata_file_path)
    model_id = metadata["id"]
    num_classes = metadata["num_classes"]

    # Create directories
    model_dir_path = os.path.join(model_dir, model_id)
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)

    # Load data
    train_dataset = load_dataset(train_dir, "train", batch_size, num_classes)
    eval_dataset = load_dataset(train_dir, "eval", batch_size, num_classes)

    # Create the model
    classifier = model(num_classes)

    # Steps
    steps = metadata["file_counts"]["train"] // batch_size
    validation_steps = metadata["file_counts"]["eval"] // batch_size

    # Train
    history = classifier.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps,
        validation_data=eval_dataset,
        validation_steps=validation_steps,
        callbacks=_get_callbacks(monitor, model_dir, model_id),
    )

    return classifier, history, model_id


def write_metadata(model_dir, model_id, batch_size, epochs, monitor):
    print("Write metadata for model")
    metadata_file_path = os.path.join(model_dir, model_id, "metadata.json")
    current_datetime = datetime.utcnow()

    metadata = {
        "id": model_id,
        "created_date": current_datetime.strftime("%Y-%m-%d %T%z"),
        "arguments": {"batch_size": batch_size, "epochs": epochs, "monitor": monitor},
    }

    with open(metadata_file_path, "w") as json_file:
        json.dump(metadata, json_file)

    return metadata


def evaluate(model, dataset, batch_size, file_count):
    steps = file_count // batch_size
    loss, acc = model.evaluate(dataset, verbose=0, steps=steps)
    return loss, acc


def _parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)

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

    classifier, history, model_id = train(
        train_dir=args.train,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        monitor=args.monitor,
    )

    # save model
    save_model(classifier, args.model_dir, model_id)

    # write metadata
    write_metadata(
        model_dir=args.model_dir,
        model_id=model_id,
        batch_size=args.batch_size,
        epochs=args.epochs,
        monitor=args.monitor,
    )
