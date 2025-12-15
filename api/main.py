from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI()

# Load model (ensure correct path)
model_path = os.path.join(os.path.dirname(__file__), "../models/model.pkl")
model = joblib.load(model_path)

# Pydantic model for input validation
class PredictRequest(BaseModel):
    features: list

@app.get("/health")
def health():
    return {"status": "API is running"}

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        # Convert to numeric numpy array
        features = np.array(request.features, dtype=float).reshape(1, -1)
        prediction = model.predict(features)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
