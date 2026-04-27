from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class IngestConfig:
    feature_cols: list[str]
    fill_zero_cols: list[str]
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
class FieldConfig:
    type: str                        # "float" or "int"
    description: str
    example: Any
    ge: float | None = None          # >=
    gt: float | None = None          # >
    le: float | None = None          # <=
    lt: float | None = None          # <


@dataclass
class ServeConfig:
    title: str
    description: str
    model_path: str
    target_description: str
    target_unit: str
    log_transform: bool
    fields: dict[str, FieldConfig]   # keyed by feature column name


@dataclass
class PipelineConfig:
    ingest: IngestConfig
    train: TrainConfig
    hpo: "HpoConfig | None" = None
    serve: "ServeConfig | None" = None


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

    for key in ("feature_cols", "fill_zero_cols", "target_col"):
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

    serve_cfg = None
    if "serve" in raw:
        s = raw["serve"]
        for key in ("title", "description", "model_path", "target_description", "target_unit", "fields"):
            if key not in s:
                raise ValueError(f"Config serve section missing required key: '{key}'")
        serve_cfg = ServeConfig(
            title=s["title"],
            description=s["description"].strip(),
            model_path=s["model_path"],
            target_description=s["target_description"],
            target_unit=s["target_unit"],
            log_transform=bool(s.get("log_transform", False)),
            fields={
                name: FieldConfig(
                    type=meta["type"],
                    description=meta["description"].strip(),
                    example=meta["example"],
                    ge=meta.get("ge"),
                    gt=meta.get("gt"),
                    le=meta.get("le"),
                    lt=meta.get("lt"),
                )
                for name, meta in s["fields"].items()
            },
        )

    return PipelineConfig(
        ingest=IngestConfig(
            feature_cols=i["feature_cols"],
            fill_zero_cols=i["fill_zero_cols"],
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
        serve=serve_cfg,
    )
