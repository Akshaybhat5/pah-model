import os
import json
import pandas as pd
from azureml.core.model import Model
import mlflow
from io import StringIO
import logging
import joblib

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def init():
    global model
    try:
        model_path = Model.get_model_path('model') 
        logging.info(f"Model path: {model_path}")
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")

    except Exception as e:
        print(f"Error loading model: {e}")
        logging.error(f"Error loading model: {e}")
        model = None



def run(raw_data):
	
    if model is None:
      return json.dumps({"errors": "Model is not initialized. Check init() function for issues."})
    try:

        input_df = pd.read_json(StringIO(raw_data), orient='records')

        # Make predictions
        predictions = model.predict(input_df)
        logging.info(f"Predictions: {predictions}")
        predictions_probabilities = model.predict_proba(input_df)

        # print("Predictions:", predictions)
        # print("Probabilities:", predictions_probabilities)
        # Return predictions as a JSON string
        return json.dumps({
            'predictions': predictions.tolist(),
            'probabilities': predictions_probabilities.tolist()  
        })
    
    except Exception as e:
        return json.dumps({'errors': str(e)})





