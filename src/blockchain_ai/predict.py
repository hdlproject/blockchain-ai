import joblib
import pandas as pd
import numpy as np


def run_inference(input_path: str, model_path: str) -> np.ndarray:
    df = pd.read_csv(input_path)
    model = joblib.load(model_path)
    return model.predict(df)
