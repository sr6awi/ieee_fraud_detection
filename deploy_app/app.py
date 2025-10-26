from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import joblib
import json
import numpy as np
import pandas as pd

app = FastAPI(title="Dual-Path Fraud Detection API")

# -------------------------------------------------------------------
# 🔹 Load model, scalers, features, and threshold
# -------------------------------------------------------------------
model = tf.keras.models.load_model("model_fraud_detection.h5")
scaler_gnn = joblib.load("scaler_gnn.pkl")
scaler_tabnet = joblib.load("scaler_tabnet.pkl")

with open("gnn_features.json") as f:
    GNN_FEATURES = json.load(f)
with open("tabnet_features.json") as f:
    TABNET_FEATURES = json.load(f)
with open("threshold.txt") as f:
    THRESHOLD = float(f.read().strip())

# -------------------------------------------------------------------
# 🔹 Input schema
# -------------------------------------------------------------------
class FraudInput(BaseModel):
    data: dict   # one transaction’s raw feature values

# -------------------------------------------------------------------
# 🔹 Prediction route
# -------------------------------------------------------------------
@app.post("/predict")
def predict(payload: FraudInput):
    row = payload.data
    df = pd.DataFrame([row])

    missing_gnn = [c for c in GNN_FEATURES if c not in df.columns]
    missing_tab = [c for c in TABNET_FEATURES if c not in df.columns]
    if missing_gnn or missing_tab:
        return {
            "error": "Missing required features",
            "missing_gnn": missing_gnn,
            "missing_tabnet": missing_tab,
        }

    X_gnn = df[GNN_FEATURES].fillna(0).to_numpy()
    X_tabnet = df[TABNET_FEATURES].fillna(0).to_numpy()

    X_gnn_scaled = scaler_gnn.transform(X_gnn)
    X_tabnet_scaled = scaler_tabnet.transform(X_tabnet)

    proba = float(model.predict([X_gnn_scaled, X_tabnet_scaled])[0][0])
    label = int(proba >= THRESHOLD)

    return {
        "fraud_probability": proba,
        "predicted_label": label
    }

# Run with: uvicorn app:app --reload
