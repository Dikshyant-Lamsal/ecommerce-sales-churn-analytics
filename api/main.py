"""FastAPI inference service for the registered churn classifier."""
from __future__ import annotations
import os, time, logging
from pathlib import Path
from typing import Literal
import joblib
import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from ml.features import FEATURE_COLUMNS

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("churn-api")
app = FastAPI(title="E-commerce Customer Churn API", version="1.0.0",
              description="Predicts 90-day customer churn from a 180-day historical feature window.")
MODEL_URI = os.getenv("MODEL_URI", "models:/EcommerceChurnClassifier@champion")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "EcommerceChurnClassifier")
model = None
model_source = None
request_count = 0
failure_count = 0
latencies_ms: list[float] = []

class CustomerFeatures(BaseModel):
    customer_id: str | None = None
    recency_days: int = Field(ge=0, le=10000)
    orders_180d: int = Field(ge=0, le=1000000)
    monetary_180d: float = Field(ge=0, le=1e12)
    avg_order_value_180d: float = Field(ge=0, le=1e9)
    items_180d: int = Field(ge=0, le=10000000)
    tenure_days: int = Field(ge=0, le=100000)
    unique_products_180d: int = Field(ge=0, le=1000000)

class Prediction(BaseModel):
    customer_id: str | None
    churn_prediction: Literal[0, 1]
    churn_probability: float
    model_name: str
    model_source: str
    definition: str = "No qualifying purchase in the 90 days after a snapshot"

@app.on_event("startup")
def load_model():
    global model, model_source
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    try:
        model = mlflow.sklearn.load_model(MODEL_URI)
        model_source = MODEL_URI
        logger.info("Loaded registered model from %s", MODEL_URI)
    except Exception as exc:
        fallback = Path(os.getenv("LOCAL_MODEL_PATH", "artifacts/churn_model.joblib"))
        if fallback.exists():
            model = joblib.load(fallback)
            model_source = str(fallback)
            logger.warning("Registry model unavailable; loaded local model: %s", exc)
        else:
            logger.error("Model unavailable: %s", exc)

@app.middleware("http")
async def record_request(request: Request, call_next):
    global request_count, failure_count
    start = time.perf_counter()
    try:
        response = await call_next(request)
        if response.status_code >= 500: failure_count += 1
        return response
    except Exception:
        failure_count += 1
        raise
    finally:
        request_count += 1
        latencies_ms.append((time.perf_counter() - start) * 1000)
        if len(latencies_ms) > 1000: del latencies_ms[:500]

@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "model_unavailable", "model_loaded": model is not None}

@app.get("/model-info")
def model_info():
    return {"model_name": MODEL_NAME, "model_uri": model_source or MODEL_URI,
            "features": FEATURE_COLUMNS, "churn_window_days": 90, "feature_window_days": 180}

@app.get("/metrics")
def monitoring_metrics():
    return {"requests_total": request_count, "server_errors_total": failure_count,
            "mean_latency_ms": round(float(np.mean(latencies_ms)), 3) if latencies_ms else 0.0,
            "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 3) if latencies_ms else 0.0}

@app.post("/predict", response_model=Prediction)
def predict(features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training and register a model before serving predictions.")
    values = features.model_dump()
    customer_id = values.pop("customer_id")
    X = np.array([[values[name] for name in FEATURE_COLUMNS]], dtype=float)
    try:
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=422, detail=f"Could not produce prediction: {exc}") from exc
    return Prediction(customer_id=customer_id, churn_prediction=prediction,
                      churn_probability=round(probability, 6), model_name=MODEL_NAME,
                      model_source=model_source or MODEL_URI)

@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled API error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
