ikapati
==============================

Plant disease detection

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

Setup
--------


With conda, you can install from the `environment.yml` file:

```
conda env create --file environment.yml
```

Alternatively:

```
conda install pip pandas numpy
conda install -c conda-forge pillow tensorflow seaborn scikit-learn
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
