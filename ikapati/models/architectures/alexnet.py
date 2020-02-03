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


def model(num_classes, learning_rate=0.1, activation="linear", padding="same", channels=3, dropout=False):
    """
    This is a convolutional neural network based on the AlexNet implementations here:
    - https://engmrk.com/alexnet-implementation-using-keras/
    - https://github.com/tensorpack/benchmarks/blob/master/other-wrappers/keras.alexnet.py
    """
    print(f"learning_rate: {learning_rate} - activation: {activation}")

    convolution_layers = [
        # Layer 1
        layers.Conv2D(
            96,
            kernel_size=(11, 11),
            input_shape=(256, 256, channels),
            strides=(4, 4),
            padding=padding,
            activation=activation,
        ),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding=padding),
        # Layer 2
        layers.Conv2D(
            256,
            kernel_size=(5, 5),
            strides=(2, 2),
            activation=activation,
            padding=padding,
        ),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding=padding),
        # Layer 3
        layers.Conv2D(
            384,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
        ),
        # Layer 4
        layers.Conv2D(
            384,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
        ),
        # Layer 5
        layers.Conv2D(
            256,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
        ),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding=padding),
    ]

    fully_connected_layers = [
        # Pass to Fully Connected (FC) Layers
        layers.Flatten(),
        # FC 1
        layers.Dense(4096, activation=activation),
        # FC 2
        layers.Dense(4096, activation=activation),
        # Output Layer
        layers.Dense(num_classes, activation="softmax"),
    ]

    if dropout:
        fully_connected_layers = [
            # Pass to Fully Connected (FC) Layers
            layers.Flatten(),
            # FC 1
            layers.Dense(4096),
            layers.Dropout(0.5),
            layers.Activation(activation),
            # FC 2
            layers.Dense(4096),
            layers.Dropout(0.5),
            layers.Activation(activation),
            # Output Layer
            layers.Dense(num_classes),
            layers.Activation("softmax"),
        ]

    _layers = convolution_layers + fully_connected_layers

    model = keras.Sequential(_layers)

    return model
