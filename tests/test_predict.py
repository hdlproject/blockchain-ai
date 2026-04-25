import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from blockchain_ai.predict import run_inference


def test_run_inference_returns_predictions(tmp_path):
    csv_path = tmp_path / "input.csv"
    model_path = tmp_path / "model.joblib"

    train_df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0], "target": [2.0, 4.0, 6.0]})
    model = LinearRegression()
    model.fit(train_df[["feature1"]], train_df["target"])
    joblib.dump(model, model_path)

    input_df = pd.DataFrame({"feature1": [4.0, 5.0]})
    input_df.to_csv(csv_path, index=False)

    predictions = run_inference(str(csv_path), str(model_path))

    assert len(predictions) == 2
    assert np.isclose(predictions[0], 8.0, atol=0.1)
    assert np.isclose(predictions[1], 10.0, atol=0.1)
