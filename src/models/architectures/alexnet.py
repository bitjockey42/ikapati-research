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


def model(num_classes, learning_rate=0.1, activation="linear", padding="same"):
    """
    This is a convolutional neural network based on the AlexNet implementations here:
    - https://engmrk.com/alexnet-implementation-using-keras/
    - https://github.com/tensorpack/benchmarks/blob/master/other-wrappers/keras.alexnet.py
    """
    print(f"learning_rate: {learning_rate} - activation: {activation}")

    model = keras.Sequential(
        [
            # Layer 1
            layers.Conv2D(
                64,
                kernel_size=(11, 11),
                input_shape=(256, 256, 3),
                strides=(4, 4),
                padding=padding,
                activation=activation,
            ),
            layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding=padding),
            # Layer 2
            layers.Conv2D(
                192,
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
                activation=activation,
                padding=padding,
            ),
            # Layer 4
            layers.Conv2D(
                256,
                kernel_size=(3, 3),
                activation=activation,
                padding=padding,
            ),
            # Layer 5
            layers.Conv2D(
                256,
                kernel_size=(3, 3),
                activation=activation,
                padding=padding,
            ),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding=padding),
            # Pass to Fully Connected (FC) Layers
            layers.Flatten(),
            # FC 1
            layers.Dense(4096, activation=activation),
            layers.Dropout(0.5),
            # FC 2
            layers.Dense(4096, activation=activation),
            layers.Dropout(0.5),
            # FC 3
            layers.Dense(4096, activation=activation),
            # Output Layer
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
