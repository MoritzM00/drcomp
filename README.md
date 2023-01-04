# Old but gold? - Statistics vs. Machine Learning in Dimensionality Reduction

This repository contains the python code for my bachelor thesis on the topic of dimensionality reduction techniques. It compares statistical dimensionality reduction techniques like PCA, Kernel PCA and LLE with newer machine learning methods for dimensionality reduction like Autoencoders. Specifically, fully connected as well as convolutional autoencoders and contractive autoencoders are studied. 

The Repository structure is as follows:
...

The configuration specifications can be found in the drcomp/conf directory. for the parameter settings for the dimensionality reduction techniques can be found 

## Installation

Install the package via pip

```bash
pip3 install git+https://github.com/MoritzM00/drcomp.git
```

## Usage

You can the cli tool to train and evaluate models. E.g. to train a PCA model on the MNIST Dataset, execute:

```bash
drcomp reducer=PCA dataset=MNIST
```

To train a model with different parameters, e.g. a PCA model on the mnist dataset with a intrinsic dimensionality of 10, execute:

```bash
drcomp reducer=PCA dataset=MNIST dataset.intrinsic_dim=10
```

### Sweeping over multiple datasets and reducers

To sweep over multiple arguments for `reducer` or `dataset`, use the `--multirun` (`-m`) flag, e.g.:

```bash
drcomp --multirun reducer=PCA,kPCA,AE dataset=MNIST,SwissRoll
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
pip3 install -r requirements-dev.txt
pip3 install -e .
```

and install the pre-commit hooks by executing:

```bash
pre-commit install
```

### Debugging the CLI

To enable debug level logging, execute the `drcomp` command with

```bash
drcomp hydra.verbose=drcomp.__main__
```

## Requirements

Python 3.9 or higher

The main dependencies are:

- numpy
- scikit-learn
- matplotlib
- pandas
- jupyter

### Development
