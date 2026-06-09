import json
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from blockchain_ai.database.job_store import JobStore


def create_router(
    job_store: JobStore,
    model,
    feature_extractor,
    feature_cols: list[str],
) -> APIRouter:
    router = APIRouter()

    @router.get("/airdrop-farmer/analyze/{address}")
    def analyze_address(address: str, background_tasks: BackgroundTasks):
        address = address.lower()
        job = job_store.get(address)
        if job is None:
            job_store.create_pending(address)
            background_tasks.add_task(
                _run_job, address, job_store, model, feature_extractor, feature_cols
            )
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
) -> None:
    try:
        if model is None:
            raise RuntimeError("Model not available — run the seed step first")
        if feature_extractor is None:
            raise RuntimeError("Etherscan client not available")
        features = feature_extractor.extract(address)
        result = model.score_wallet(features, feature_cols)
        result.update({col: features[col] for col in feature_cols})
        job_store.mark_done(address, result)
    except Exception as exc:
        job_store.mark_failed(address, str(exc))
