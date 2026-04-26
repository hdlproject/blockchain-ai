import optuna
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

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
        }
        model = XGBRegressor(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = {**study.best_params, "random_state": 42}
    return replace(config, hyperparameters=best_params)
