import requests

url = "http://127.0.0.1:8000/predict"

heartbeat_data = {
    "features": {
        "heart_rate_bpm": 75.0,
        "rr_interval_ms": 800.0,
        "heart_rate_variability_hrv_ms": 45.0
    }
}

response = requests.post(url, json=heartbeat_data)

print("Status Code:", response.status_code)
print("Heartbeat Stability Prediction:", response.json())