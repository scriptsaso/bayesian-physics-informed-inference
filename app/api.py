from fastapi import FastAPI
from pydantic import BaseModel
from app.inference import predict

app = FastAPI()

# ---- Request model ----
class InputData(BaseModel):
    features: list[float]

# ---- Health check ----
@app.get("/")
def health():
    return {"status": "running"}

# ---- Prediction endpoint ----
@app.post("/predict")
def predict_endpoint(data: InputData):
    y = predict(data.features)
    return {"prediction": y}