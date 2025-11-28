import optuna
from optuna.samplers import TPESampler
import pandas as pd
import ast
import os
from typing import Callable, Dict, Any, Optional, Tuple

def read_param_csv(path: str) -> pd.DataFrame:
    """Reads parameter ranges CSV into DataFrame and fills missing columns."""
    df = pd.read_csv(path, sep=';', dtype=str).fillna("")
    # ensure expected columns exist
    for c in ["name","type","low","high","log","choices","default"]:
        if c not in df.columns:
            df[c] = ""
    return df

def suggest_from_spec(trial: optuna.trial.Trial, spec_row: pd.Series):
    """Given one row of param spec, call trial.suggest_... and return value."""
    name = spec_row["name"]
    ptype = spec_row["type"].strip().lower()
    log = spec_row["log"].strip().lower() in ("true","1","yes","y","t")
    if ptype == "float":
        low = float(spec_row["low"])
        high = float(spec_row["high"])
        # allow log suggestion
        return trial.suggest_float(name, low, high, log=log)
    elif ptype == "int":
        low = int(float(spec_row["low"]))
        high = int(float(spec_row["high"]))
        return trial.suggest_int(name, low, high, log=log)
    elif ptype == "categorical":
        # parse choices as comma separated; allow python-literal list too
        choices_raw = spec_row["choices"].strip()
        if choices_raw.startswith("[") or choices_raw.startswith("("):
            choices = ast.literal_eval(choices_raw)
        else:
            choices = [c.strip() for c in choices_raw.split(",") if c.strip() != ""]
        return trial.suggest_categorical(name, choices)
    elif ptype == "bool":
        # store as True/False
        return trial.suggest_categorical(name, [False, True])
    else:
        raise ValueError(f"Unknown param type '{ptype}' for {name}")

def _make_tpe_sampler(good_fraction: float = 0.2, n_startup: int = 10, multivariate: bool = False) -> TPESampler:
    """
    Create a TPESampler that uses the provided good_fraction (0 < good_fraction <= 1).
    We implement gamma as a function that returns number of 'good' trials to use
    (gamma should be integer count). See Optuna docs for TPESampler.gamma.
    """
    if not (0 < good_fraction <= 1.0):
        raise ValueError("good_fraction must be in (0, 1].")
    def gamma_fn(n_finished: int) -> int:
        # keep at least 1 trial; ensure we do not ask for more than n_finished
        k = max(1, int(round(good_fraction * max(1, n_finished))))
        return min(k, n_finished)
    return TPESampler(n_startup_trials=n_startup, gamma=gamma_fn, multivariate=multivariate)

def bayesian_optimize(
    run_model: Callable[..., float],
    param_csv_path: str,
    n_trials: int = 100,
    output_history_csv: str = "bayes_trials.csv",
    resume: bool = True,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    good_fraction: float = 0.2,
    n_startup: int = 10,
    multivariate_tpe: bool = False,
    verbose: bool = True,
    *args,
    **kwargs
) -> Tuple[optuna.Study, pd.DataFrame]:
    """
    Generic Bayesian optimization using Optuna TPE (configurable 'good' fraction).
    - run_model: function that accepts hyperparameters as kwargs (and other *args/**kwargs)
                 and returns a scalar loss (to minimize).
                 Example signature the user can implement:
                   def run_model(*args, opt=None, scheduler=None, loss_fn=None, **params) -> float
    - param_csv_path: CSV path describing the search space (see above).
    - n_trials: number of trials to run (including resumed).
    - output_history_csv: file to write/appand trial results.
    - resume: if True and output_history_csv exists, will load it as starting history (for logging).
    - storage & study_name: optional optuna storage string + name (for RDB persistence).
    - good_fraction: fraction of completed trials considered 'good' for TPE (0..1].
    - n_startup: number of random startup trials before using TPE logic.
    - multivariate_tpe: whether to enable multivariate TPE.
    - *args/**kwargs forwarded into run_model (so you can pass opt/scheduler/loss_fn etc).
    Returns: (study, dataframe-of-trials)
    """
    # Load search-space spec
    spec_df = read_param_csv(param_csv_path)
    # Setup sampler with user-controlled "good fraction"
    sampler = _make_tpe_sampler(good_fraction=good_fraction, n_startup=n_startup, multivariate=multivariate_tpe)

    # create study (optionally with DB storage if provided)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=study_name or None, storage=storage, load_if_exists=True)

    # in-memory history df for full trace
    if resume and os.path.exists(output_history_csv):
        history_df = pd.read_csv(output_history_csv)
        if verbose:
            print(f"[bayes] Loaded existing history with {len(history_df)} rows from {output_history_csv}")
    else:
        history_df = pd.DataFrame()

    # objective closure
    def objective(trial: optuna.trial.Trial) -> float:
        # for each param in spec, ask the trial to suggest it
        params = {}
        for _, row in spec_df.iterrows():
            val = suggest_from_spec(trial, row)
            params[row["name"]] = val

        # combine user provided kwargs and these params; let run_model accept arbitrary args and kwargs
        # Note: if user passes an explicit param in kwargs, we won't overwrite it unless they wanted to.
        merged_kwargs = dict(kwargs)  # copy
        # only add params that are not already explicitly provided in kwargs
        for k, v in params.items():
            if k in merged_kwargs:
                # do not override provided kwarg; treat CSV as search-space unless user pre-specified
                continue
            merged_kwargs[k] = v

        # Call the black box function. It must return a scalar loss (lower is better)
        loss = run_model(*args, **merged_kwargs)
        # Save trial to history
        row = {"trial_number": trial.number, "value": float(loss)}
        # include params in row
        for k, v in params.items():
            row[k] = v
        nonlocal history_df, output_history_csv
        history_df = pd.concat([history_df, pd.DataFrame([row])], ignore_index=True)
        # write CSV every trial for safety
        history_df.to_csv(output_history_csv, index=False)
        if verbose:
            print(f"[bayes] Trial {trial.number} -> loss: {loss:.6f}")
        return float(loss)

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # finalize: return study and dataframe
    # load CSV again for canonical ordering
    final_df = pd.read_csv(output_history_csv)
    return study, final_df

# Example run_model stub (user replaces with real model-training function)
if __name__ == "__main__":
    import time
    def run_model_stub(*args, opt=None, scheduler=None, loss_fn=None, **params):
        """
        Example black-box: pretend to train a model and return a synthetic loss.
        In real usage: build model, optimizer (opt), scheduler, call training loop, compute val loss, return it.
        run_model may accept opt/scheduler/loss_fn objects or parameters to create them.
        """
        # Example: use params to make synthetic deterministic loss
        # (replace with actual model training + return validation loss)
        lr = float(params.get("lr", 1e-3))
        bs = int(params.get("batch_size", 64))
        dropout = float(params.get("dropout", 0.2))
        # Make a fake loss (toy)
        loss = (lr - 0.001)**2 + (bs - 64)**2 / 1e6 + dropout**2
        time.sleep(0.1)  # pretend this took some time
        return loss

    # Example usage:
    study, df = bayesian_optimize(
        run_model=run_model_stub,
        param_csv_path="param_ranges.csv",
        n_trials=30,
        output_history_csv="bayes_trials.csv",
        resume=False,
        good_fraction=0.2,
        n_startup=5,
        multivariate_tpe=False,
        verbose=True,
    )
    print("Best trial:", study.best_params, study.best_value)


