from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(
    title="Heartbeat Stability Prediction API",
    description="Predict heartbeat stability using Random Forest ML",
    version="1.0"
)

# -------------------------------
# Load trained model
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "heartbeat_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
features = saved_data["features"]


# -------------------------------
# Home endpoint
# -------------------------------
@app.get("/")
def home():
    return {
        "message": "Welcome to Heartbeat Stability Prediction API",
        "status": "Running"
    }


# -------------------------------
# Prediction input schema
# -------------------------------
class HeartbeatData(BaseModel):
    features: dict[str, float]


# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
def predict(data: HeartbeatData):

    # Check for missing features
    missing_features = [
        feature for feature in features
        if feature not in data.features
    ]

    if missing_features:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Missing required features",
                "missing_features": missing_features
            }
        )

    try:
        # Arrange features in correct model order
        input_data = pd.DataFrame(
            [[data.features[feature] for feature in features]],
            columns=features
        )

        # Generate prediction
        prediction = model.predict(input_data)

        return {
            "predicted_result": str(prediction[0]),
            "status": "Prediction successful"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )