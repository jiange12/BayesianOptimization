# from general_bayes_opt import bayesian_optimize
# from test_black_box import run_model

# if __name__ == "__main__":
#     study, df = bayesian_optimize(
#         run_model=run_model,                 # our black-box model
#         param_csv_path="param_ranges.csv",   # parameter space
#         n_trials=25,                         # number of trials
#         output_history_csv="trial_history.csv",
#         resume=False,                        # start fresh
#         good_fraction=0.2,                   # top 20% = "good" trials for TPE
#         n_startup=5,                         # number of random startup trials
#         multivariate_tpe=True,               # enable correlation-aware sampling
#         verbose=True
#     )

#     print("\n=== Optimization Finished ===")
#     print("Best trial parameters:", study.best_params)
#     print("Best trial loss:", study.best_value)

from general_bayes_opt import bayesian_optimize
from toy_black_box import run_model

if __name__ == "__main__":
    study, df = bayesian_optimize(
        run_model,
        param_csv_path="param_space.csv",
        n_trials=100 
    )

    print("Best params found:")
    print(study.best_params)
    print("Best loss:")
    print(study.best_value)
