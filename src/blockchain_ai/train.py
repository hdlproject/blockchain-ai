import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression


def train_model(input_path: str, target_col: str, model_path: str, model_type: str = "linear") -> object:
    df = pd.read_csv(input_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if model_type == "linear":
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Supported: 'linear'")

    model.fit(X, y)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model
