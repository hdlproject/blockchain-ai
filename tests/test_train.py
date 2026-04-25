import pandas as pd
import pytest
from blockchain_ai.train import train_model


def test_train_model_saves_artifact(tmp_path):
    csv_path = tmp_path / "processed.csv"
    model_path = tmp_path / "model.joblib"
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [2, 4, 6, 8, 10],
        "target": [3, 6, 9, 12, 15],
    })
    df.to_csv(csv_path, index=False)

    train_model(str(csv_path), "target", str(model_path), model_type="linear")

    assert model_path.exists()


def test_train_model_raises_on_unknown_model_type(tmp_path):
    csv_path = tmp_path / "processed.csv"
    df = pd.DataFrame({"feature1": [1, 2], "target": [3, 4]})
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Unknown model_type"):
        train_model(str(csv_path), "target", str(tmp_path / "m.joblib"), model_type="unknown")
