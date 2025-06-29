import numpy as np
import pickle
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global model variable
model = None

# Load model from pickle
def load_model(model_name):
    path = os.path.join(os.getcwd(), 'model_dump', model_name)
    try:
        with open(path, 'rb') as f:
            loaded_model = pickle.load(f)
        logging.info(f"✅ Model loaded successfully from {path}")
        return loaded_model
    except FileNotFoundError:
        logging.error(f"❌ Model file '{model_name}' not found at {path}")
    except Exception as e:
        logging.error(f"❌ Error loading model: {e}")
    return None

# FastAPI lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_filename = os.getenv("MODEL_FILENAME", "housing_regression.pkl")
    model = load_model(model_filename)
    if model is None:
        logging.error("❌ FastAPI startup failed: Model not loaded.")
    yield

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Request model
class Features(BaseModel):
    features: list

# Routes
@app.post("/predict")
async def predict(features_data: Features):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        features = np.array(features_data.features).reshape(1, -1)
        logging.info(f"Received features: {features_data.features}")
        prediction = model.predict(features)
        return {"prediction": prediction[0]}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/test")
async def test_route():
    return {"message": "test route working"}


"""
curl -X POST "http://127.0.0.1:5000/predict" \
-H "Content-Type: application/json" \
-d '{"features": [-122.23, 37.88, 41.0, 880.0, 129.0, 322.0, 126.0, 8.3252]}'
"""