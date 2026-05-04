import joblib
import pandas as pd
import numpy as np


def run_inference(input_path: str, model_path: str) -> np.ndarray:
    df = pd.read_csv(input_path)
    model = joblib.load(model_path)
    return model.predict(df)


LABEL_ENCODER: dict[int, str] = {0: "sanctioned", 1: "scammer", 2: "phishing"}


def predict_address(
    features: dict[str, float],
    model,
    feature_cols: list[str],
    threshold: float,
) -> dict:
    df = pd.DataFrame([features])[feature_cols]
    proba = model.predict_proba(df)[0]
    label_idx = int(proba.argmax())
    max_prob = float(proba.max())
    label = LABEL_ENCODER[label_idx] if max_prob >= threshold else "unknown"
    return {
        "label": label,
        "probabilities": {
            "sanctioned": round(float(proba[0]), 4),
            "scammer": round(float(proba[1]), 4),
            "phishing": round(float(proba[2]), 4),
        },
    }
