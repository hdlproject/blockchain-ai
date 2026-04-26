from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class IngestConfig:
    drop_cols: list[str]
    fill_zero_cols: list[str]
    timestamp_col: str
    target_col: str


@dataclass
class TrainConfig:
    target_col: str
    model_type: str
    stratify_col: str
    test_size: float
    hyperparameters: dict


@dataclass
class HpoConfig:
    n_trials: int


@dataclass
class PipelineConfig:
    ingest: IngestConfig
    train: TrainConfig
    hpo: "HpoConfig | None" = None


def load_config(path: str) -> PipelineConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(p) as f:
        raw = yaml.safe_load(f)

    if "ingest" not in raw:
        raise ValueError("Config missing required key: 'ingest'")
    if "train" not in raw:
        raise ValueError("Config missing required key: 'train'")

    i = raw["ingest"]
    t = raw["train"]

    for key in ("drop_cols", "fill_zero_cols", "timestamp_col", "target_col"):
        if key not in i:
            raise ValueError(f"Config ingest section missing required key: '{key}'")

    for key in ("target_col", "model_type", "stratify_col", "test_size", "hyperparameters"):
        if key not in t:
            raise ValueError(f"Config train section missing required key: '{key}'")

    hpo_cfg = None
    if "hpo" in raw:
        h = raw["hpo"]
        if not h or "n_trials" not in h:
            raise ValueError("Config hpo section missing required key: 'n_trials'")
        hpo_cfg = HpoConfig(n_trials=h["n_trials"])

    return PipelineConfig(
        ingest=IngestConfig(
            drop_cols=i["drop_cols"],
            fill_zero_cols=i["fill_zero_cols"],
            timestamp_col=i["timestamp_col"],
            target_col=i["target_col"],
        ),
        train=TrainConfig(
            target_col=t["target_col"],
            model_type=t["model_type"],
            stratify_col=t["stratify_col"],
            test_size=t["test_size"],
            hyperparameters=t["hyperparameters"],
        ),
        hpo=hpo_cfg,
    )
