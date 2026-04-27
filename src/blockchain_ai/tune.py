import time
import optuna
import plotext as plt
import pandas as pd
from dataclasses import replace
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
from blockchain_ai.config import TrainConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_hpo(input_path: str, config: TrainConfig, n_trials: int) -> TrainConfig:
    df = pd.read_csv(input_path)
    X = df.drop(columns=[config.target_col])
    y = df[config.target_col]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    trial_nums: list[int] = []
    rmses: list[float] = []
    times: list[float] = []
    best_rmses: list[float] = []

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
        }
        t0 = time.perf_counter()
        scores = cross_val_score(XGBRegressor(**params), X, y, cv=cv, scoring="neg_root_mean_squared_error")
        elapsed = time.perf_counter() - t0

        rmse = -scores.mean()
        trial_nums.append(trial.number + 1)
        rmses.append(rmse)
        times.append(elapsed)
        best_rmses.append(min(rmses))
        return -rmse

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    _print_charts(trial_nums, rmses, best_rmses, times)

    best = study.best_trial
    print(f"Best trial #{best.number + 1}  RMSE={-best.value:.6f}")
    print(f"Params: {study.best_params}")

    best_params = {**study.best_params, "random_state": 42}
    return replace(config, hyperparameters=best_params)


def _print_charts(
    trials: list[int],
    rmses: list[float],
    best_rmses: list[float],
    times: list[float],
) -> None:
    plt.clf()
    plt.plot_size(80, 20)
    plt.title("HPO — RMSE per trial")
    plt.scatter(trials, rmses, label="trial RMSE", marker="dot")
    plt.plot(trials, best_rmses, label="best so far")
    plt.xlabel("Trial")
    plt.ylabel("RMSE")
    plt.show()

    plt.clf()
    plt.plot_size(80, 15)
    plt.title("HPO — CV time per trial (seconds)")
    plt.bar(trials, times)
    plt.xlabel("Trial")
    plt.ylabel("Time (s)")
    plt.show()
