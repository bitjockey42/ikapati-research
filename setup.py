from setuptools import find_packages, setup

setup(
    name='ikapati',
    packages=find_packages(),
    version='0.1.0',
    description='Plant disease detection',
    author='Allyson Julian',
    license='BSD-3',
    entry_points={
        'console_scripts': [
            'ikapati-data=ikapati.data.make_dataset:main',
            'ikapati-train=ikapati.models.train_model:main',
            'ikapati-eval=ikapati.models.evaluate_model:main',
        ]
    },
)
