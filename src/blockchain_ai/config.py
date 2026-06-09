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
class PathsConfig:
    processed_path: str
    report_path: str
    test_path: str = ""


@dataclass
class CollectConfig:
    output_path: str
    days: int = 7
    txs_per_hour: int = 50
    n_blocks: int = 0  # kept for gas-price pipeline scripts
    checkpoint_every: int = 100
    max_history_blocks: int = 50_000


@dataclass
class GoPlusConfig:
    base_url: str
    chain_id: int
    rate_limit_per_sec: int
    timeout_sec: int


@dataclass
class OFACConfig:
    sdn_url: str
    timeout_sec: int


@dataclass
class ScamSnifferConfig:
    url: str
    timeout_sec: int


@dataclass
class MEWConfig:
    url: str
    timeout_sec: int


@dataclass
class AirdropConfig:
    contract_address: str
    date: str  # ISO format e.g. "2024-01-15"


@dataclass
class PipelineConfig:
    ingest: IngestConfig
    train: TrainConfig
    task: str = "regression"
    hpo: "HpoConfig | None" = None
    serve: "ServeConfig | None" = None
    paths: "PathsConfig | None" = None
    etherscan: "EtherscanConfig | None" = None
    collect: "CollectConfig | None" = None
    goplus: "GoPlusConfig | None" = None
    ofac: "OFACConfig | None" = None
    scamsniffer: "ScamSnifferConfig | None" = None
    mew: "MEWConfig | None" = None
    airdrop: "AirdropConfig | None" = None


def _parse_fields(fields_raw: dict) -> dict[str, FieldConfig]:
    return {
        name: FieldConfig(
            type=meta["type"],
            description=meta["description"].strip(),
            example=meta["example"],
            ge=meta.get("ge"),
            gt=meta.get("gt"),
            le=meta.get("le"),
            lt=meta.get("lt"),
        )
        for name, meta in fields_raw.items()
    }


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
    if task not in ("regression", "classification", "clustering", "airdrop_farmer"):
        raise ValueError(f"Config 'task' must be 'regression', 'classification', 'clustering', or 'airdrop_farmer', got: {task!r}")

    i = raw["ingest"]
    t = raw["train"]

    required_ingest = ["feature_cols", "fill_zero_cols"] if task in ("clustering", "airdrop_farmer") else ["feature_cols", "fill_zero_cols", "target_col"]
    for key in required_ingest:
        if key not in i:
            raise ValueError(f"Config ingest section missing required key: '{key}'")

    required_train = ["model_type", "hyperparameters"] if task in ("clustering", "airdrop_farmer") else ["target_col", "model_type", "test_size", "hyperparameters"]
    for key in required_train:
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
                fields=_parse_fields(s["fields"]),
            )
        elif task == "classification":
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
        elif task == "clustering":
            for key in ("model_path", "title", "description", "fields"):
                if key not in s:
                    raise ValueError(f"Config serve section missing required key: '{key}'")
            serve_cfg = ServeConfig(
                model_path=s["model_path"],
                title=s["title"],
                description=s["description"].strip(),
                fields=_parse_fields(s["fields"]),
            )
        elif task == "airdrop_farmer":
            for key in ("model_path", "db_path"):
                if key not in s:
                    raise ValueError(f"Config serve section missing required key: '{key}'")
            serve_cfg = ServeConfig(
                model_path=s["model_path"],
                db_path=s["db_path"],
            )
        else:
            raise ValueError(f"Unknown task: {task!r}")

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

    paths_cfg = None
    if "paths" in raw:
        p = raw["paths"]
        required_paths = ["processed_path", "report_path"]
        if task != "clustering":
            required_paths.append("test_path")
        for key in required_paths:
            if key not in p:
                raise ValueError(f"Config paths section missing required key: '{key}'")
        paths_cfg = PathsConfig(
            processed_path=p["processed_path"],
            test_path=p.get("test_path", ""),
            report_path=p["report_path"],
        )

    collect_cfg = None
    if "collect" in raw:
        c = raw["collect"]
        if "output_path" not in c:
            raise ValueError("Config collect section missing required key: 'output_path'")
        collect_cfg = CollectConfig(
            output_path=c["output_path"],
            days=int(c.get("days", 7)),
            txs_per_hour=int(c.get("txs_per_hour", 50)),
            n_blocks=int(c.get("n_blocks", 0)),
            checkpoint_every=int(c.get("checkpoint_every", 100)),
            max_history_blocks=int(c.get("max_history_blocks", 50_000)),
        )

    if task == "classification":
        for section in ("goplus", "ofac"):
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
        for key in ("sdn_url", "timeout_sec"):
            if key not in o:
                raise ValueError(f"Config ofac section missing required key: '{key}'")
        ofac_cfg = OFACConfig(
            sdn_url=o["sdn_url"],
            timeout_sec=int(o["timeout_sec"]),
        )

    scamsniffer_cfg = None
    if "scamsniffer" in raw:
        s = raw["scamsniffer"]
        for key in ("url", "timeout_sec"):
            if key not in s:
                raise ValueError(f"Config scamsniffer section missing required key: '{key}'")
        scamsniffer_cfg = ScamSnifferConfig(url=s["url"], timeout_sec=int(s["timeout_sec"]))

    mew_cfg = None
    if "mew" in raw:
        m = raw["mew"]
        for key in ("url", "timeout_sec"):
            if key not in m:
                raise ValueError(f"Config mew section missing required key: '{key}'")
        mew_cfg = MEWConfig(url=m["url"], timeout_sec=int(m["timeout_sec"]))

    airdrop_cfg = None
    if "airdrop" in raw:
        a = raw["airdrop"]
        for key in ("contract_address", "date"):
            if key not in a:
                raise ValueError(f"Config airdrop section missing required key: '{key}'")
        airdrop_cfg = AirdropConfig(
            contract_address=a["contract_address"],
            date=a["date"],
        )

    if task == "airdrop_farmer" and airdrop_cfg is None:
        raise ValueError("Config task=airdrop_farmer requires an 'airdrop' section")

    return PipelineConfig(
        task=task,
        ingest=IngestConfig(
            feature_cols=i["feature_cols"],
            fill_zero_cols=i["fill_zero_cols"],
            target_col=i.get("target_col", ""),
        ),
        train=TrainConfig(
            target_col=t.get("target_col", ""),
            model_type=t["model_type"],
            test_size=float(t.get("test_size", 0.0)),
            hyperparameters=t["hyperparameters"],
        ),
        hpo=hpo_cfg,
        serve=serve_cfg,
        paths=paths_cfg,
        etherscan=etherscan_cfg,
        collect=collect_cfg,
        goplus=goplus_cfg,
        ofac=ofac_cfg,
        scamsniffer=scamsniffer_cfg,
        mew=mew_cfg,
        airdrop=airdrop_cfg,
    )
