"""Quick check of /predict (run AFTER task4_transformers.py has saved the model):  python test_api.py"""
from fastapi.testclient import TestClient
from app import app

with TestClient(app) as client:          # 'with' runs the lifespan -> model loads once
    r = client.post("/predict", json={"text": "The delivery was late and the phone arrived broken"})
    print(r.status_code, r.json())
    r = client.post("/predict", json={"text": "I love this laptop, great value"})
    print(r.status_code, r.json())
    r = client.post("/predict", json={"text": "   "})
    print("empty text ->", r.status_code)        # expect 422
