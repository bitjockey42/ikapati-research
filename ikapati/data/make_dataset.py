# -*- coding: utf-8 -*-
import click
import os

from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split

from ikapati.data import io
from ikapati.data import utils


def prepare_dataset(
    data_dir_path: str, output_dir_path: str, species: str, test_size: float, file_ext: str = "JPG"
):
    image_file_paths = utils.get_image_paths(data_dir_path, species)
    metadata_file_path = os.path.join(output_dir_path, "metadata.json")

    class_names = utils.get_class_names(data_dir_path, species)
    dataset_map = utils.get_dataset_map(image_file_paths, class_names, test_size)

    for key, file_paths in dataset_map.items():
        print(f"Writing {key} record")
        record_file_path = os.path.join(output_dir_path, f"{key}.tfrecord")
        io.write_dataset(record_file_path, file_paths, class_names, file_ext)

    print(f"Writing metadata to {metadata_file_path}")
    io.write_metadata(metadata_file_path, dataset_map, species, class_names)


@click.command()
@click.argument("species", nargs=-1)
@click.option("--data_dir", type=click.Path())
@click.option("--output_dir", type=click.Path())
@click.option("--test_size", type=click.FLOAT, default=0.4)
@click.option("--overwrite/--no_overwrite", default=False)
@click.option(
    "--file_ext", default="JPG", help="The file extension of images, e.g. JPG"
)
def main(species, data_dir, output_dir, test_size, overwrite, file_ext):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    print(species)

    species_names = "_".join(sorted(species))
    data_dir = Path(data_dir)
    output_dir = Path(output_dir, species_names)

    if output_dir.exists() and not overwrite:
        raise ValueError(f"{str(output_dir)}. Use the --overwrite flag to overwrite existing data")

    if not output_dir.exists():
        os.makedirs(str(output_dir))

    print(str(output_dir))

    prepare_dataset(str(data_dir), str(output_dir), species, test_size, file_ext)


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
