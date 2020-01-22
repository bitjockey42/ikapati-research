import argparse
import os
import json

import numpy as np
from tensorflow import keras


def convert_prediction(prediction, num_classes):
    return keras.utils.to_categorical(np.argmax(prediction, axis=0), num_classes)


def model(filename):
    return keras.models.load_model(filename)


def _parse_args():
    parser = argparse.ArgumentParser()


if __name__ == "__main__":
    pass