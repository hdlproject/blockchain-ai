import numpy as np
import pandas as pd
import pytest
from blockchain_ai.config import IngestConfig, PipelineConfig, ServeConfig, TrainConfig
from blockchain_ai.ingest import load_and_clean


def _make_csv(tmp_path, df):
    csv_path = tmp_path / "ethereum-transactions.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def _minimal_df(**overrides):
    base = {
        "base_fee_gwei": [10.0],
        "gas_used_ratio": [0.5],
        "hour_of_day": [12],
        "day_of_week": [1],
        "base_fee_trend": [0.02],
    }
    base.update(overrides)
    return pd.DataFrame(base)


_FEATURE_COLS = ["base_fee_gwei", "gas_used_ratio", "hour_of_day", "day_of_week", "base_fee_trend"]


def _regression_cfg(target_col: str = "log_base_fee_gwei", fill_zero_cols=None) -> PipelineConfig:
    return PipelineConfig(
        ingest=IngestConfig(feature_cols=_FEATURE_COLS, fill_zero_cols=fill_zero_cols or [], target_col=target_col),
        train=TrainConfig(target_col=target_col, model_type="xgboost", test_size=0.2, hyperparameters={}),
        serve=ServeConfig(model_path="", log_transform=True),
    )


def _no_transform_cfg(target_col: str, fill_zero_cols=None, feature_cols=None) -> PipelineConfig:
    return PipelineConfig(
        ingest=IngestConfig(
            feature_cols=feature_cols or _FEATURE_COLS,
            fill_zero_cols=fill_zero_cols or [],
            target_col=target_col,
        ),
        train=TrainConfig(target_col=target_col, model_type="xgboost", test_size=0.2, hyperparameters={}),
    )


def test_load_and_clean_keeps_only_feature_cols(tmp_path):
    df = _minimal_df()
    csv_path = _make_csv(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    result = load_and_clean(csv_path, out_path, _regression_cfg())

    assert set(result.columns) == {
        "base_fee_gwei", "gas_used_ratio", "hour_of_day", "day_of_week", "base_fee_trend",
        "log_base_fee_gwei",
    }


def test_load_and_clean_adds_log_target(tmp_path):
    df = _minimal_df(base_fee_gwei=[10.0])
    csv_path = _make_csv(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    result = load_and_clean(csv_path, out_path, _regression_cfg())

    assert "log_base_fee_gwei" in result.columns
    expected = np.log1p(10.0)
    assert abs(result["log_base_fee_gwei"].iloc[0] - expected) < 1e-6


def test_load_and_clean_target_outside_feature_cols(tmp_path):
    """Target source column not in feature_cols — the production fix for data leakage."""
    cfg = _regression_cfg(target_col="log_next_base_fee_gwei")
    df = _minimal_df()
    df["next_base_fee_gwei"] = [12.0]
    csv_path = _make_csv(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    result = load_and_clean(csv_path, out_path, cfg)

    assert result["base_fee_gwei"].iloc[0] == pytest.approx(10.0)
    assert "log_next_base_fee_gwei" in result.columns
    assert abs(result["log_next_base_fee_gwei"].iloc[0] - np.log1p(12.0)) < 1e-6
    assert "next_base_fee_gwei" not in result.columns


def test_load_and_clean_fills_zero_cols(tmp_path):
    cfg = _no_transform_cfg(
        target_col="label",
        fill_zero_cols=["gas_used_ratio"],
        feature_cols=["base_fee_gwei", "gas_used_ratio"],
    )
    df = pd.DataFrame({"base_fee_gwei": [10.0], "gas_used_ratio": [None], "label": [1]})
    csv_path = _make_csv(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    result = load_and_clean(csv_path, out_path, cfg)

    assert result["gas_used_ratio"].iloc[0] == 0.0


def test_load_and_clean_saves_csv(tmp_path):
    df = _minimal_df()
    csv_path = _make_csv(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    load_and_clean(csv_path, out_path, _regression_cfg())

    assert (tmp_path / "out.csv").exists()


def test_load_and_clean_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_and_clean(str(tmp_path / "missing.csv"), str(tmp_path / "out.csv"), _regression_cfg())


def test_load_and_clean_no_log_transform_uses_target_directly(tmp_path):
    df = pd.DataFrame({"feat": [1.0, 2.0], "label": [0, 1]})
    csv_path = _make_csv(tmp_path, df)
    cfg = _no_transform_cfg(target_col="label", feature_cols=["feat"])

    result = load_and_clean(csv_path, str(tmp_path / "out.csv"), cfg)

    assert list(result["label"]) == [0, 1]
