from fastapi import FastAPI
import joblib
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

app = FastAPI()
model: RandomForestClassifier = joblib.load("models/random_forest_model.pkl")
scaler: StandardScaler = joblib.load("models/scaler.pkl")

@app.post("/predict")
def predict(features: dict):
    """
    Predict if a transaction is fraudulent based on input features.

    Parameters:
    features (dict): Dictionary of features for prediction.
    """
    # Convert features to a dataframe
    features = pd.DataFrame({k: [float(v)] for k, v in features.items()}).drop(columns=['Class'])

    # Scale both Time and Amount at the same time
    features[['Time', 'Amount']] = scaler.transform(features[['Time', 'Amount']])
    
    # Make prediction 
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}