import joblib
from keras.models import load_model

def load_scaler_model():
    scaler = joblib.load("models/scaler.pkl")
    model = load_model("models/my_model.keras")
    return scaler, model
