Bayesian Optimization Framework (Optuna + CSV-Defined Search Spaces)

This repository provides a lightweight, generic framework for running Bayesian hyperparameter optimization using Optuna.
Its core idea is simple:

Define your hyperparameter search space in a CSV file

Implement any black-box function (run_model) that returns a scalar loss

Run optimization using a TPE sampler

Automatically log all trials to CSV for reproducibility

The system is model-agnostic and works with any Python code—machine learning or otherwise.

Features

CSV-based hyperparameter search space

Optuna TPE optimization (supports multivariate mode)

Automatic logging of all trials to history CSV files

Plug-and-play black-box objective functions

Example implementations included:

Deterministic toy objective

PyTorch regression model

Installation

Install required packages:

pip install optuna torch numpy pandas

If you’re not using the PyTorch example, you may omit torch.

Defining the Search Space

Hyperparameters are defined in a semicolon-separated CSV file.
Example (param_space.csv):

name;type;low;high;log;choices;default
lr;float;1e-5;1e-2;True;;
weight_decay;float;1e-6;1e-2;True;;
hidden_size;int;16;256;False;;
dropout;float;0.0;0.5;False;;
optimizer;categorical;;;;"adam,adamw,sgd";
gamma;float;0.90;0.999;False;;

Columns:
| Column         | Description                                    |
| -------------- | ---------------------------------------------- |
| `name`         | Hyperparameter name passed to your `run_model` |
| `type`         | `float`, `int`, `categorical`, or `bool`       |
| `low` / `high` | Numeric range (for float/int types)            |
| `log`          | Whether to sample on a log scale               |
| `choices`      | For categorical parameters                     |
| `default`      | Optional fallback when not suggested           |

The optimizer will automatically convert each row into the appropriate Optuna trial.suggest_* call.

Implementing the Black-Box Function

You provide any function that returns a scalar loss:

def run_model(**params):
    # Train a model, evaluate it, or run any computation
    loss = ...
    return loss

Two example implementations are included:

toy_black_box.py – deterministic objective with a known optimum

test_black_box.py – PyTorch neural network training example

Running Optimization

Example script:

from general_bayes_opt import bayesian_optimize
from toy_black_box import run_model

study, df = bayesian_optimize(
    run_model,
    param_csv_path="param_space.csv",
    n_trials=100,
)

This will:

Read the CSV search space

Create an Optuna TPE study

Suggest parameter sets

Evaluate run_model

Write all trials to bayes_trials.csv

Return:

study (Optuna object)

df (pandas DataFrame of trial history)

Inspect the best result:

print("Best params:", study.best_params)
print("Best loss:", study.best_value)

Example Use Cases
Toy Optimization (default in main.py)

Runs the optimizer on a handcrafted deterministic function to verify system behavior.
Fast and reproducible.

PyTorch Regression Example

Uncomment the corresponding code in main.py to run the neural network example.
This performs real training and tunes:

Architecture (hidden size, dropout)

Optimizer (Adam, AdamW, SGD)

Learning rate, weight decay

Scheduler gamma

Logs results to trial_history.csv.

Repository Structure

.
├── general_bayes_opt.py     # Core optimization engine
├── main.py                  # Example entry point
├── toy_black_box.py         # Deterministic toy objective
├── test_black_box.py        # PyTorch regression black-box
├── param_space.csv          # Example search space
├── bayes_trials.csv         # Trial history (toy example output)
├── trial_history.csv        # Trial history (PyTorch example output)

Extending the Framework

You can customize:

The search space (via CSV)

The objective function (run_model)

The TPE sampler settings

The logging process

Support for new parameter types

This framework is intentionally simple so you can modify or embed it easily.

Requirements

Python 3.8+

Optuna

Pandas

Numpy

PyTorch (only required for the PyTorch example)
