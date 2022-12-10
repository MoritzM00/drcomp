# Old but gold? - Statistics vs. Machine Learning in Dimensionality Reduction

This repo contains the python code for my bachelor thesis on the topic of dimensionality reduction techniques.

## Installation
Install the package via pip
```bash
pip3 install git+https://github.com/MoritzM00/drcomp
```

## Usage

You can the cli tool to train and evaluate models. E.g. to train a PCA model on the MNIST Dataset, execute:

```bash
drcomp reducer=PCA dataset=MNIST
```
## Development

Create a virtual environment first, for example by executing:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

and then install the package `drcomp` locally with pip:

```bash
pip3 install -r requirements.txt
pip3 install -e .
```

## Usage

TODO

## Requirements

Python 3.9 or higher

The main dependencies are:

- numpy
- scikit-learn
- matplotlib
- pandas
- jupyter

### Development

To develop, you need to install the development dependencies:

```bash
pip3 install -r requirements-dev.txt
```

and install the pre-commit hooks by executing:

```bash
pre-commit install
```
