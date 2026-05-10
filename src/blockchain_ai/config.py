from dataclasses import dataclass
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
    test_size: float
    hyperparameters: dict


@dataclass
class HpoConfig:
    n_trials: int


@dataclass
class FieldConfig:
    type: str
    description: str
    example: Any
    ge: float | None = None
    gt: float | None = None
    le: float | None = None
    lt: float | None = None


@dataclass
class ServeConfig:
    model_path: str
    # regression-only
    title: str | None = None
    description: str | None = None
    target_description: str | None = None
    target_unit: str | None = None
    log_transform: bool = False
    fields: dict[str, FieldConfig] | None = None
    # classification-only
    confidence_threshold: float | None = None
    db_path: str | None = None


@dataclass
class EtherscanConfig:
    base_url: str
    chain_id: int
    rate_limit_per_sec: int
    timeout_sec: int


@dataclass
class CollectConfig:
    n_blocks: int
    output_path: str
    checkpoint_every: int = 100
    max_history_blocks: int = 50_000  # ~7 days of Ethereum blocks


@dataclass
class GoPlusConfig:
    base_url: str
    chain_id: int
    rate_limit_per_sec: int
    timeout_sec: int


@dataclass
class OFACConfig:
    alt_url: str
    timeout_sec: int


@dataclass
class FortaConfig:
    graphql_url: str
    timeout_sec: int
    max_alerts: int
    scam_bot_ids: list[str]


@dataclass
class PipelineConfig:
    ingest: IngestConfig
    train: TrainConfig
    task: str = "regression"
    hpo: "HpoConfig | None" = None
    serve: "ServeConfig | None" = None
    etherscan: "EtherscanConfig | None" = None
    collect: "CollectConfig | None" = None
    goplus: "GoPlusConfig | None" = None
    ofac: "OFACConfig | None" = None
    forta: "FortaConfig | None" = None


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

    task = raw.get("task", "regression")
    if task not in ("regression", "classification"):
        raise ValueError(f"Config 'task' must be 'regression' or 'classification', got: {task!r}")

    i = raw["ingest"]
    t = raw["train"]

    for key in ("feature_cols", "fill_zero_cols", "target_col"):
        if key not in i:
            raise ValueError(f"Config ingest section missing required key: '{key}'")

    for key in ("target_col", "model_type", "test_size", "hyperparameters"):
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
        if task == "regression":
            for key in ("title", "description", "model_path", "target_description", "target_unit", "fields"):
                if key not in s:
                    raise ValueError(f"Config serve section missing required key: '{key}'")
            serve_cfg = ServeConfig(
                model_path=s["model_path"],
                title=s["title"],
                description=s["description"].strip(),
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
        else:
            for key in ("model_path", "confidence_threshold", "db_path"):
                if key not in s:
                    raise ValueError(f"Config serve section missing required key: '{key}'")
            serve_cfg = ServeConfig(
                model_path=s["model_path"],
                title=s.get("title"),
                description=s["description"].strip() if s.get("description") else None,
                confidence_threshold=float(s["confidence_threshold"]),
                db_path=s["db_path"],
            )

    etherscan_cfg = None
    if "etherscan" in raw:
        e = raw["etherscan"]
        for key in ("base_url", "chain_id", "rate_limit_per_sec", "timeout_sec"):
            if key not in e:
                raise ValueError(f"Config etherscan section missing required key: '{key}'")
        etherscan_cfg = EtherscanConfig(
            base_url=e["base_url"],
            chain_id=int(e["chain_id"]),
            rate_limit_per_sec=int(e["rate_limit_per_sec"]),
            timeout_sec=int(e["timeout_sec"]),
        )

    collect_cfg = None
    if "collect" in raw:
        c = raw["collect"]
        for key in ("n_blocks", "output_path"):
            if key not in c:
                raise ValueError(f"Config collect section missing required key: '{key}'")
        collect_cfg = CollectConfig(
            n_blocks=int(c["n_blocks"]),
            output_path=c["output_path"],
            checkpoint_every=int(c.get("checkpoint_every", 100)),
            max_history_blocks=int(c.get("max_history_blocks", 50_000)),
        )

    if task == "classification":
        for section in ("goplus", "ofac", "forta"):
            if section not in raw:
                raise ValueError(f"Config task=classification requires section: '{section}'")

    goplus_cfg = None
    if "goplus" in raw:
        g = raw["goplus"]
        for key in ("base_url", "chain_id", "rate_limit_per_sec", "timeout_sec"):
            if key not in g:
                raise ValueError(f"Config goplus section missing required key: '{key}'")
        goplus_cfg = GoPlusConfig(
            base_url=g["base_url"],
            chain_id=int(g["chain_id"]),
            rate_limit_per_sec=int(g["rate_limit_per_sec"]),
            timeout_sec=int(g["timeout_sec"]),
        )

    ofac_cfg = None
    if "ofac" in raw:
        o = raw["ofac"]
        for key in ("alt_url", "timeout_sec"):
            if key not in o:
                raise ValueError(f"Config ofac section missing required key: '{key}'")
        ofac_cfg = OFACConfig(
            alt_url=o["alt_url"],
            timeout_sec=int(o["timeout_sec"]),
        )

    forta_cfg = None
    if "forta" in raw:
        fo = raw["forta"]
        for key in ("graphql_url", "timeout_sec", "max_alerts", "scam_bot_ids"):
            if key not in fo:
                raise ValueError(f"Config forta section missing required key: '{key}'")
        forta_cfg = FortaConfig(
            graphql_url=fo["graphql_url"],
            timeout_sec=int(fo["timeout_sec"]),
            max_alerts=int(fo["max_alerts"]),
            scam_bot_ids=fo["scam_bot_ids"],
        )

    return PipelineConfig(
        task=task,
        ingest=IngestConfig(
            feature_cols=i["feature_cols"],
            fill_zero_cols=i["fill_zero_cols"],
            target_col=i["target_col"],
        ),
        train=TrainConfig(
            target_col=t["target_col"],
            model_type=t["model_type"],

            test_size=t["test_size"],
            hyperparameters=t["hyperparameters"],
        ),
        hpo=hpo_cfg,
        serve=serve_cfg,
        etherscan=etherscan_cfg,
        collect=collect_cfg,
        goplus=goplus_cfg,
        ofac=ofac_cfg,
        forta=forta_cfg,
    )
