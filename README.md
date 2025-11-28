This repository provides a lightweight framework for running Bayesian hyperparameter optimization using Optuna. Hyperparameters are defined in a simple CSV file, and any Python function that returns a scalar loss can be optimized. The framework automatically constructs the search space, runs TPE-based optimization, and logs all trials to CSV for easy analysis.

Included examples:

A deterministic toy objective (toy_black_box.py)

A small PyTorch regression model (test_black_box.py)

An example script (main.py) showing how to run optimization

The core logic lives in general_bayes_opt.py, which handles reading the CSV, creating the Optuna study, sampling parameters, evaluating the objective, and saving trial history.
