import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def train_model(
    input_path: str,
    target_col: str,
    model_path: str,
    test_path: str,
    model_type: str = "xgboost",
) -> object:
    df = pd.read_csv(input_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df["transaction_type"]
    )

    if model_type == "xgboost":
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Supported: 'xgboost'")

    model.fit(X_train, y_train)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    test_df = X_test.copy()
    test_df[target_col] = y_test
    Path(test_path).parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(test_path, index=False)

    return model
