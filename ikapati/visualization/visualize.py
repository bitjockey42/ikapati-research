import numpy as np
import pandas as pd
import seaborn as sns

from typing import List
from matplotlib import pyplot as plt


def create_metrics_for_dataset_type(
    history: List[dict], dataset_type: str
) -> pd.DataFrame:
    loss_key = "loss"
    accuracy_key = "accuracy"

    if dataset_type == "validation":
        loss_key = "val_loss"
        accuracy_key = "val_accuracy"

    data = {"loss": history[loss_key], "accuracy": history[accuracy_key]}

    df = pd.DataFrame(data)
    df["epoch"] = df.index
    df["name"] = dataset_type

    return df


def combine_dataframes(data_frames: List[pd.DataFrame]):
    combined_df = pd.concat(data_frames)
    combined_df = combined_df.reset_index()
    return combined_df


def create_metrics_dataframe(history: List[dict]) -> pd.DataFrame:
    training_df = create_metrics_for_dataset_type(history, "train")
    validation_df = create_metrics_for_dataset_type(history, "validation")

    return combine_dataframes([training_df, validation_df])


def learning_curves(metric: str, metrics_df: pd.DataFrame):
    return sns.lineplot(x="epoch", y=metric, hue="name", data=metrics_df, legend="full")


def save_plot(plot, filename: str):
    plot.get_figure().savefig(filename)
