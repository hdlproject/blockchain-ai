import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from blockchain_ai.feature.block_features import BlockFeatureExtractor
from blockchain_ai.config import ServeConfig


def create_router(
    serve: ServeConfig,
    feature_cols: list[str],
    model,
    etherscan_client,
) -> APIRouter:
    router = APIRouter()

    def _fetch_blocks() -> pd.DataFrame:
        if etherscan_client is None:
            raise HTTPException(
                status_code=503,
                detail="Etherscan client not available. Check ETHERSCAN_API_KEY.",
            )
        n_fetch = model.sequence_length + BlockFeatureExtractor.TREND_LOOKBACK
        latest = etherscan_client.get_latest_block_number()
        raw_rows = [
            etherscan_client.get_block(n)
            for n in range(latest - n_fetch + 1, latest + 1)
        ]
        raw_rows = [r for r in raw_rows if r]
        try:
            return BlockFeatureExtractor().extract(raw_rows)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    def _predict_one(rolling: pd.DataFrame) -> float:
        raw = float(model.predict(rolling)[-1])
        return float(np.expm1(raw) if serve.log_transform else raw)

    @router.get(
        "/predict/gas-price/v2/latest",
        summary="Predict next-block base fee using LSTM (v2)",
        description=(
            "Fetches recent blocks from Etherscan, feeds the last sequence_length rows "
            "into the LSTM, and returns up to n_blocks auto-regressive predictions."
        ),
    )
    def predict_latest_v2(n_blocks: int = Query(default=1, ge=1, le=50)):
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not available yet. Run training pipeline first.",
            )
        df = _fetch_blocks()
        seq_len = model.sequence_length
        rolling_df = df[feature_cols].tail(seq_len).reset_index(drop=True)
        last = df.iloc[-1]
        last_block = int(last["block_number"])
        last_timestamp = float(last["timestamp"])
        gas_used_ratio = float(last["gas_used_ratio"])

        predictions = []
        for step in range(1, n_blocks + 1):
            pred_gwei = _predict_one(rolling_df)
            timestamp = last_timestamp + step * 12
            predictions.append({
                "step": step,
                "block_number": last_block + step,
                "base_fee_gwei": pred_gwei,
                "base_fee_wei": pred_gwei * 1e9,
                "method": "lstm",
            })
            if step < n_blocks:
                dt = pd.Timestamp(timestamp, unit="s", tz="UTC")
                lookback = min(BlockFeatureExtractor.TREND_LOOKBACK, len(rolling_df))
                ref_fee = float(rolling_df["base_fee_gwei"].iloc[-lookback])
                trend = (pred_gwei - ref_fee) / ref_fee if ref_fee > 0 else 0.0
                new_row = pd.DataFrame([{
                    "base_fee_gwei": pred_gwei,
                    "gas_used_ratio": gas_used_ratio,
                    "hour_of_day": dt.hour,
                    "day_of_week": dt.dayofweek,
                    "base_fee_trend": trend,
                }])[feature_cols]
                rolling_df = pd.concat(
                    [rolling_df.iloc[1:], new_row], ignore_index=True
                )

        return {
            "block_number": last_block,
            "block_history": (
                df[["block_number", "base_fee_gwei"]]
                .rename(columns={"block_number": "block"})
                .to_dict(orient="records")
            ),
            "predictions": predictions,
        }

    return router
