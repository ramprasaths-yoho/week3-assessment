"""FastAPI service. Run:  uvicorn app:app --reload   then open http://127.0.0.1:8000/docs"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from transformers import pipeline

from common import MODEL_DIR

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # model is loaded ONCE at start-up, not on every request
    models["clf"] = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR)
    yield
    models.clear()


app = FastAPI(title="Customer Feedback Classifier", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")      # FastAPI turns this into HTTP 422
        return v.strip()


class PredictResponse(BaseModel):
    label: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    out = models["clf"](req.text, truncation=True, max_length=128)[0]
    return PredictResponse(label=out["label"], confidence=round(float(out["score"]), 4))
