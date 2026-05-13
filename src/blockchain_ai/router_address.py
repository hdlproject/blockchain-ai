import json
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from blockchain_ai.database.job_store import JobStore
from blockchain_ai.predict import predict_address


def create_router(
    job_store: JobStore,
    model,
    feature_extractor,
    feature_cols: list[str],
    threshold: float,
) -> APIRouter:
    router = APIRouter()

    @router.get("/predict/address/{address}")
    def predict_address_endpoint(address: str, background_tasks: BackgroundTasks):
        address = address.lower()
        job = job_store.get(address)
        if job is None:
            job_store.create_pending(address)
            background_tasks.add_task(_run_job, address, job_store, model, feature_extractor, feature_cols, threshold)
            return JSONResponse({"address": address, "status": "pending"}, status_code=202)
        if job["status"] == "pending":
            return JSONResponse({"address": address, "status": "pending"}, status_code=202)
        if job["status"] == "done":
            result = json.loads(job["result"])
            return {"address": address, "status": "done", **result}
        return {"address": address, "status": "failed", "error": job.get("error")}

    return router


def _run_job(
    address: str,
    job_store: JobStore,
    model,
    feature_extractor,
    feature_cols: list[str],
    threshold: float,
) -> None:
    try:
        if feature_extractor is None:
            raise RuntimeError("Etherscan client not available")
        if model is None:
            raise RuntimeError("Model not loaded")
        features = feature_extractor.extract(address)
        result = predict_address(features, model, feature_cols, threshold)
        job_store.mark_done(address, result)
    except Exception as exc:
        job_store.mark_failed(address, str(exc))
