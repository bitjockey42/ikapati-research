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

from uuid import uuid4

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import (
    layers,
    models,
    activations,
    losses,
    optimizers,
)


def model(num_classes):
    """Generate a simple model"""
    model = keras.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(256,256,3),padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2),padding='same'),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation='linear',padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D(pool_size=(2, 2),padding='same'),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation='linear',padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D(pool_size=(2, 2),padding='same'),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(128, activation='linear'),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        loss=losses.categorical_crossentropy,
        optimizer=optimizers.Adam(),
        metrics=['accuracy']
    )

    model.summary()

    return model


def _load_training_data(base_dir):
    """Load training data"""
    x_train = np.load(os.path.join(base_dir, 'train_data.npy'))
    y_train = np.load(os.path.join(base_dir, 'train_labels.npy'))
    return x_train, y_train


def _load_testing_data(base_dir):
    """Load testing data"""
    x_test = np.load(os.path.join(base_dir, 'eval_data.npy'))
    y_test = np.load(os.path.join(base_dir, 'eval_labels.npy'))
    return x_test, y_test


def _save_model(model, model_dir):
    model_id = str(uuid4())
    print(f"Saving model {model_id}")
    model_filepath = os.path.join(model_dir, f"{model_id}.h5")
    model.save(model_filepath)
    # Save as tflite model as well
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(f"{model_id}.tflite", "wb") as converted_model_file:
        converted_model_file.write(tflite_model)


def _parse_args():
    parser = argparse.ArgumentParser()
    
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.1)

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    
    # https://github.com/aws/sagemaker-containers#sm-hosts
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--eval', type=str, default=os.environ.get('SM_CHANNEL_EVAL'))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    # load data
    train_data, train_labels = _load_training_data(args.train)
    eval_data, eval_labels = _load_testing_data(args.train)

    # get num_classes
    num_classes = train_labels[0].shape[0]

    # create the model
    classifier = model(num_classes)

    # train
    classifier.fit(train_data, train_labels, batch_size=args.batch_size, epochs=args.epochs, validation_data=(eval_data, eval_labels))

    # save model
    _save_model(classifier, args.model_dir)
