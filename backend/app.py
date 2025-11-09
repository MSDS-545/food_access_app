from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib  # using joblib instead of pickle for the 3-feature model

app = FastAPI(title="Food Access Prediction API")

# Load model & preprocessing pipeline (3-feature version)
imputer = joblib.load("imputer_3features.pkl")
scaler  = joblib.load("scaler_3features.pkl")
model   = joblib.load("model_3features.pkl")

print("Loaded preprocessing & model. Imputer, scaler, model ready.")

class SimpleFeatures(BaseModel):
    HUNVFlag:    float
    PovertyRate: float
    LA1and10:    float

class PredictionResponse(BaseModel):
    predicted_class: int
    probability:     float

@app.post("/predict/simple", response_model=PredictionResponse)
def predict_simple(features: SimpleFeatures):
    try:
        X_raw = np.array([[features.HUNVFlag,
                           features.PovertyRate,
                           features.LA1and10]])
        # Impute, scale, predict
        X_imp    = imputer.transform(X_raw)
        X_scaled = scaler.transform(X_imp)
        prob     = model.predict_proba(X_scaled)[0][1]
        pred     = int(model.predict(X_scaled)[0])
        return PredictionResponse(predicted_class=pred, probability=prob)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# (Optional) If you still want the full-feature endpoint, adapt below
class FullFeatures(BaseModel):
    # List your full features here if needed
    HUNVFlag:    float
    PovertyRate: float
    LA1and10:    float
    # feature4: float
    # …
    # feature143: float

@app.post("/predict/full", response_model=PredictionResponse)
def predict_full(features: FullFeatures):
    try:
        # Example of building full vector – you’ll need all feature names + defaults
        features_order = [
            "HUNVFlag", "PovertyRate", "LA1and10",
            # "feature4", … "feature143"
        ]
        defaults = {
            # "feature4": 0.0,
            # …
        }
        input_vals = [
            getattr(features, feat) if hasattr(features, feat) else defaults.get(feat, 0.0)
            for feat in features_order
        ]
        X_new = np.array([input_vals])
        X_imp    = imputer.transform(X_new)
        X_scaled = scaler.transform(X_imp)
        prob     = model.predict_proba(X_scaled)[0][1]
        pred     = int(model.predict(X_scaled)[0])
        return PredictionResponse(predicted_class=pred, probability=prob)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
