import numpy as np
import joblib
from keras.models import load_model

model = load_model("models/my_model.keras")
scaler = joblib.load("models/scaler.pkl")

print("Enter 22 values separated by commas (exclude 'name' and 'status'):")
features = list(map(float, input().split(",")))

if len(features) == 22:
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    result = "Parkinson's" if prediction > 0.5 else "Healthy"
    print(f"Diagnosis: {result}")
else:
    print("Invalid input. Please enter exactly 22 numeric values.")
