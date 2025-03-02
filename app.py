import numpy as np
from flask import Flask, request, jsonify
import pickle
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


worker_class = 'sync'

app = Flask(__name__)

def load_model(model_name):
    """Loads a model from a pickle file."""
    pth = os.path.join(os.getcwd(), 'model_dump', model_name)

    try:
        model = pickle.load(open(pth, 'rb'))
        return model
    except FileNotFoundError:
        print(f"Error: Model file '{model_name}' in {pth} not found.")
        return None
    except Exception as e:
        print(f"An error occurred loading the model: {e}")
        return None


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts a value based on the input data.

    Receives JSON data containing an 'features' key, converts it to a NumPy array,
    uses the loaded model to make a prediction, and returns the prediction as JSON.

    Returns:
        JSON: The prediction from the model.
    """
    try:
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)
        print(features)
        print(features.shape)
        logging.info(f"Received prediction request: {data}")
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" key in input'}), 400

        prediction = model.predict(features)
        output = prediction[0]
        return jsonify(output)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/test', methods=['POST'])
def test():
    return "test route working"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask API for model prediction.')
    parser.add_argument('--model', type=str, required=True, help='Path to the model pickle file.')
    parser.add_argument('--app_type', type=str, default='trial', required=True, help='What type of app to run.')
    args = parser.parse_args()

    model = load_model(args.model)

    if model is not None:
        if args.app_type == 'trial':
            app.run(debug=True, port=5000)

        elif args.app_type == 'production':
            # Production deployment using Gunicorn
            from gunicorn.app.base import BaseApplication

            class StandaloneApplication(BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super().__init__()

                def load_config(self):
                    config = {key: value for key, value in self.options.items()
                            if key in self.cfg.settings and value is not None}
                    for key, value in config.items():
                        self.cfg.set(key.lower(), value)

                def load(self):
                    return self.application
                
            options = {
            'bind': '0.0.0.0:5000',  # Bind to all interfaces
            'workers': 4,  # Adjust number of workers as needed
            'worker-class': 'sync', # or 'gevent', 'uvicorn.workers.UvicornWorker'
            'timeout': 30, # Adjust timeout
            'log-level': 'info',
            'accesslog': '-', # log to stdout
            'errorlog': '-', # log to stderr
        }

        StandaloneApplication(app, options).run()
    else:
        print("Failed to start the API due to model loading issues.")


"""
How to run:

$body = @{ features = @(-122.23, 37.88, 41.0, 880.0, 129.0, 322.0, 126.0, 8.3252)} | ConvertTo-Json
$headers = @{ "Content-Type" = "application/json" }
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -Body $body

"""