import pandas as pd

LABEL_ENCODER: dict[int, str] = {0: "sanctioned", 1: "phishing", 2: "scammer"}


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
            name: round(float(proba[idx]), 4) if idx < len(proba) else 0.0
            for idx, name in LABEL_ENCODER.items()
        },
    }
