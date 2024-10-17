# WNE Masters Thesis

The repository contains the implementation of evaluation framework and strategies for master's thesis: "Informer In Algorithmic Investment
Strategies on High Frequency Bitcoin Data"

The implementation uses [PytorchForecasting](https://pytorch-forecasting.readthedocs.io/en/stable/) framework, and the Informer implementation is taken from the following [Github repository](https://github.com/martinwhl/Informer-PyTorch-Lightning). Experiment management is done using [Weights&Biases](https://wandb.ai/site).

## Repository structure
Thre repository is organized in the following way:
- `src/ml`: contains the code that implements utilities for training the machine learning models. 
- `src/informer`: contains the implementation of the Informer.
- `src/strategy`: contains the implementation of the strategies, as well as the methods for efficient evaluation and hyperparameter serach.
- `notebooks/`: contains the evaluations of various srategies and methods for generating the visualistions used in the publication.
- `scripts/`: contains the main training script.
- `configs/`: contains configurations for different experiments of ml models, that can be passed to the main training script.

## Installation

Install package with the following command:
```
pip install -e .
```

## Training 

Usage of the training script:
```
usage: train.py [-h] [-p PROJECT] [-l LOG_LEVEL] [-s SEED] [-n LOG_INTERVAL] [-v VAL_CHECK_INTERVAL] [-t PATIENCE] [--no-wandb] [--store-predictions] config

positional arguments:
  config                Experiment configuration file in yaml format.

optional arguments:
  -h, --help            show this help message and exit
  -p PROJECT, --project PROJECT
                        W&B project name. (default: wne-masters-thesis-testing)
  -l LOG_LEVEL, --log-level LOG_LEVEL
                        Sets the log level. (default: 20)
  -s SEED, --seed SEED  Random seed for the training. (default: 42)
  -n LOG_INTERVAL, --log-interval LOG_INTERVAL
                        Log every n steps. (default: 100)
  -v VAL_CHECK_INTERVAL, --val-check-interval VAL_CHECK_INTERVAL
                        Run validation every n batches. (default: 300)
  -t PATIENCE, --patience PATIENCE
                        Patience for early stopping. (default: 5)
  --no-wandb            Disables wandb, for testing. (default: False)
  --store-predictions   Whether to store predictions of the best run. (default: False)
```