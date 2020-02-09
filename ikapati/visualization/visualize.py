import numpy as np
import pandas as pd
import seaborn as sns
import pathlib

from typing import List
from matplotlib import pyplot as plt

from ikapati.data.io import read_metadata


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
    df["dataset"] = dataset_type

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
    return sns.lineplot(
        x="epoch", y=metric, hue="dataset", data=metrics_df, legend="full"
    )


def save_plot(plot, filename: str):
    plot.get_figure().savefig(filename)


def save_metrics_plots_for_training(training_csv_filename: str, file_ext: str = "png"):
    training_df = pd.read_csv(training_csv_filename)
    training_df = training_df.fillna(0)
    project_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = project_dir.joinpath("reports", "figures")

    for _, row in training_df.iterrows():
        metadata_file_path = project_dir.joinpath(row.model_dir_path, "metadata.json")
        metadata = read_metadata(str(metadata_file_path))
        filename_template = f"{row.start_time}.{row.activation}.learning_rate___{row.learning_rate}.dropout___{row.dropout}.batch_size___{row.batch_size}.epochs___{row.epochs}"

        model_figures_dir = figures_dir.joinpath(metadata["id"])
        if not model_figures_dir.exists():
            model_figures_dir.mkdir(parents=True)

        history = metadata["history"]
        metrics_df = create_metrics_dataframe(history)

        loss_plot_file_path = model_figures_dir.joinpath(
            f"LOSS.{filename_template}.{file_ext}"
        )
        loss_plot = learning_curves("loss", metrics_df)
        save_plot(loss_plot, str(loss_plot_file_path))
        plt.clf()

        acc_plot_file_path = model_figures_dir.joinpath(
            f"ACCURACY.{filename_template}.{file_ext}"
        )
        acc_plot = learning_curves("accuracy", metrics_df)
        save_plot(acc_plot, str(figures_dir.joinpath(acc_plot_file_path)))
        plt.clf()
