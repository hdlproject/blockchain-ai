#!/usr/bin/env python3
import io
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

MODEL_PATH = "models/model.joblib"

app = FastAPI(title="blockchain-ai", version="0.1.0")
model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse CSV: {e}")
    try:
        preds: np.ndarray = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")
    return JSONResponse({"predictions": preds.tolist(), "count": len(preds)})
