import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from blockchain_ai.config import TrainConfig

LABEL_TO_INT = {"sanctioned": 0, "scammer": 1, "phishing": 2}


def train_model(
        input_path: str,
        model_path: str,
        test_path: str,
        config: TrainConfig,
        task: str = "regression",
) -> object:
    df = pd.read_csv(input_path)
    X = df.drop(columns=[config.target_col])
    y = df[config.target_col]

    if task == "classification":
        y = y.map(LABEL_TO_INT)

    if task == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.test_size,
            random_state=config.hyperparameters.get("random_state", 42),
            stratify=y,
        )
        if config.model_type == "xgboost":
            hparams = {k: v for k, v in config.hyperparameters.items()}
            model = XGBClassifier(**hparams)
        else:
            raise ValueError(f"Unknown model_type: {config.model_type!r}. Supported: 'xgboost'")
    else:
        stratify = df[config.stratify_col] if config.stratify_col else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.test_size,
            random_state=config.hyperparameters.get("random_state", 42),
            stratify=None,
        )
        if config.model_type == "xgboost":
            model = XGBRegressor(**config.hyperparameters)
        else:
            raise ValueError(f"Unknown model_type: {config.model_type!r}. Supported: 'xgboost'")

    model.fit(X_train, y_train)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    test_df = X_test.copy()
    test_df[config.target_col] = y_test
    Path(test_path).parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(test_path, index=False)

    return model
