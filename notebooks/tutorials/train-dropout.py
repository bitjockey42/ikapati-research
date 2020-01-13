# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.import tensorflow as tf

import argparse
import os
import numpy as np
import json

from tensorflow import keras
from tensorflow.keras import (
    layers,
    models,
    activations,
    losses,
    optimizers,
)

NUM_CLASSES = 10


def model(x_train, y_train, x_test, y_test):
    """Generate a simple model"""
    model = keras.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'),
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
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ])

    model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adam(), metrics=['accuracy'])
    model.fit(x_train, y_train)
    model.evaluate(x_test, y_test)

    print(model.summary())
    return model


def _load_training_data(base_dir):
    """Load MNIST training data"""
    x_train = np.load(os.path.join(base_dir, 'train_data.npy'))
    y_train = np.load(os.path.join(base_dir, 'train_labels.npy'))
    return x_train, y_train


def _load_testing_data(base_dir):
    """Load MNIST testing data"""
    x_test = np.load(os.path.join(base_dir, 'eval_data.npy'))
    y_test = np.load(os.path.join(base_dir, 'eval_labels.npy'))
    return x_test, y_test


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    train_data, train_labels = _load_training_data(args.train)
    eval_data, eval_labels = _load_testing_data(args.train)

    classifier = model(train_data, train_labels, eval_data, eval_labels)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        classifier.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')
