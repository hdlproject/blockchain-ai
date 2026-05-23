import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from blockchain_ai.config import PipelineConfig, TrainConfig

LABEL_TO_INT = {"sanctioned": 0, "phishing": 1, "scammer": 2}


def train_model(
        input_path: str,
        model_path: str,
        test_path: str,
        cfg: PipelineConfig,
) -> object:
    train_config = cfg.train

    if cfg.hpo is not None:
        from blockchain_ai.tune import run_hpo
        train_config = run_hpo(input_path, train_config, n_trials=cfg.hpo.n_trials)

    df = pd.read_csv(input_path)
    X = df.drop(columns=[train_config.target_col])
    y = df[train_config.target_col]
    stratify = None
    if cfg.task == "classification":
        y = y.map(LABEL_TO_INT)
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=train_config.test_size,
        random_state=train_config.hyperparameters.get("random_state", 42),
        stratify=stratify,
    )

    if cfg.task == "classification":
        if train_config.model_type == "xgboost":
            model = XGBClassifier(**train_config.hyperparameters)
        else:
            raise ValueError(f"Unknown model_type: {train_config.model_type!r}. Supported: 'xgboost'")
    elif cfg.task == "regression":
        if train_config.model_type == "xgboost":
            model = XGBRegressor(**train_config.hyperparameters)
        elif train_config.model_type == "lstm":
            from blockchain_ai.model.lstm import LSTMWrapper
            model = LSTMWrapper(**train_config.hyperparameters)
        else:
            raise ValueError(
                f"Unknown model_type: {train_config.model_type!r}. Supported: 'xgboost', 'lstm'"
            )
    else:
        raise ValueError(f"Unknown task: {cfg.task!r}")

    model.fit(X_train, y_train)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    test_df = X_test.copy()
    test_df[train_config.target_col] = y_test
    Path(test_path).parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(test_path, index=False)

    return model
