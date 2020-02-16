ikapati
==============================

Plant disease detection using TensorFlow 2.0.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── ikapati            <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


Setup with `tensorman`
------------------------

[`Tensorman`](https://support.system76.com/articles/use-tensorman/) is a tool created by the developers of `Pop!_OS` for managing TensorFlow toolchains. It runs `TensorFlow` inside docker containers.

Assuming the GPU requirements are satisfied, bring up a container as root:

```bash
tensorman run --gpu --python3 --root --name ikapati_build bash
```

Inside that container:

```bash
# apt-get install -y python3-venv
pip install poetry
```

Disable virtualenv creation by `poetry` since we're running it inside a docker container:

```bash
poetry config virtualenvs.create false
```

This creates a configuration file under `.config/pypoetry/`, which will be created in this directory since it's mounted in the container.

```ini
[virtualenvs]
create = false
```

Upgrade `pip`:

```bash
# Poetry runs a really older version of pip, which causes installation to fail
poetry run pip install pip==20.0.2
# Install the dependencies
poetry install
```

In another terminal window on the host, fix the permissions on `.config` and `.cache` and save the container as an image:

```bash
sudo chown -R $USER {.config,.cache}
sudo chgrp -R $USER {.config,.cache}
tensorman save ikapati_build ikapati_build
```

Exit the container, then run a new one as a regular user:

```bash
tensorman =ikapati run --gpu --name ikapati_dev bash
```

For `jupyter`:

```bash
tensorman =ikapati run -p 8888:8888 --gpu --python3 --jupyter bash
```

Then inside the container:

```bash
jupyter lab --ip=0.0.0.0 --no-browser
```


Setup with `poetry`
-----------------------

```bash
poetry shell
# necessary otherwise dependency resolution will fail
pip install -U pip
```

Update dependencies and generate new `poetry.lock`:

```bash
poetry update
```

Install:

```bash
poetry install
```


Setup with `conda`
-----------------

With conda, you can install from the `environment.yml` file:

```
conda env create --file environment.yml
```

Alternatively:

```
conda install pip pandas numpy
conda install -c conda-forge pillow tensorflow-gpu seaborn scikit-learn
```

Then install requirements:

```
conda install pip
pip install -r requirements.txt
```

These environment variables need to be set:

```
SM_HOSTS=["algo-1","algo-2"]
CUDA_VISIBLE_DEVICES=0
```

To set these inside the conda env:

```
# Make sure you're in the env
conda activate ikapati

# Set vars
conda env config vars set CUDA_VISIBLE_DEVICES=0
conda env config vars set SM_HOSTS='["algo-1","algo-2"]'

# Re-activate env to load new variables
conda activate ikapati

# Look at list of set env vars
conda env config vars list
```

Running
----------

```
python src/data/make_dataset.py data/fashion_test
python src/models/train_model.py --epochs 20 --batch_size 64 --model_dir models --train data/fashion_test
```

