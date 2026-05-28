from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import Field, create_model

from blockchain_ai.config import FieldConfig, ServeConfig

_TYPE_MAP = {"float": float, "int": int}


def _pydantic_field(fc: FieldConfig) -> Any:
    constraints: dict[str, Any] = {"description": fc.description, "examples": [fc.example]}
    if fc.ge is not None:
        constraints["ge"] = fc.ge
    if fc.gt is not None:
        constraints["gt"] = fc.gt
    if fc.le is not None:
        constraints["le"] = fc.le
    if fc.lt is not None:
        constraints["lt"] = fc.lt
    return (_TYPE_MAP[fc.type], Field(**constraints))


def create_router(serve: ServeConfig, feature_cols: list[str], model) -> APIRouter:
    router = APIRouter()

    TxModel = create_model(
        "TransactionInput",
        **{name: _pydantic_field(fc) for name, fc in (serve.fields or {}).items()},
    )

    @router.post("/detect/transaction", summary="Detect anomalous transaction")
    def detect_transaction(tx: TxModel):  # type: ignore[valid-type]
        if model is None:
            raise HTTPException(status_code=503, detail="Model not available.")
        df = pd.DataFrame([tx.model_dump()])[feature_cols]
        score = float(model.anomaly_score(df.values)[0])
        anomaly = score > model.eps
        return {
            "anomaly": anomaly,
            "score": round(score, 4),
            "label": "anomaly" if anomaly else "normal",
        }

    return router
