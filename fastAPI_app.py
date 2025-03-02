import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pickle
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

def load_model(model_name):
    """Loads a model from a pickle file."""
    pth = os.path.join(os.getcwd(), 'model_dump', model_name)
    try:
        model = pickle.load(open(pth, 'rb'))
        return model
    except FileNotFoundError:
        logging.error(f"Error: Model file '{model_name}' in {pth} not found.")
        return None
    except Exception as e:
        logging.error(f"An error occurred loading the model: {e}")
        return None

class Features(BaseModel):
    features: list

@app.post("/predict")
async def predict(features_data: Features):
    """Predicts a value based on the input data."""
    try:
        features = np.array(features_data.features).reshape(1, -1)
        logging.info(f"Received prediction request: {features_data.features}")
        prediction = model.predict(features)
        output = prediction[0]
        return output
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/test")
async def test():
    return "test route working"

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description='FastAPI for model prediction.')
    parser.add_argument('--model', type=str, required=True, help='Path to the model pickle file.')
    parser.add_argument('--app_type', type=str, default='uvicorn', required=True, help='What type of app to run.')
    args = parser.parse_args()

    model = load_model(args.model)

    if model is not None:
        if args.app_type == 'uvicorn':
            uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
    else:
        print("Failed to start the API due to model loading issues.")



"""
How to run:
$body = @{ features = @(-122.23, 37.88, 41.0, 880.0, 129.0, 322.0, 126.0, 8.3252); } | ConvertTo-Json  
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -Body $body -ContentType "application/json"
"""