from tensorflow import keras

import numpy as np


def convert_prediction(prediction, num_classes):
    return keras.utils.to_categorical(np.argmax(prediction, axis=0), num_classes)
