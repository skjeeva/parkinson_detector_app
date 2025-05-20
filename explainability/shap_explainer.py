import shap
import pandas as pd
import joblib
from keras.models import load_model

# Load data, scaler, and model
df = pd.read_csv("data/parkinsons.data").drop(['name'], axis=1)
X = df.drop('status', axis=1)
scaler = joblib.load("models/scaler.pkl")
X_scaled = scaler.transform(X)

model = load_model("models/my_model.keras")

# Explain
explainer = shap.Explainer(model, X_scaled)
shap_values = explainer(X_scaled[:100])

shap.summary_plot(shap_values, features=X_scaled[:100], feature_names=X.columns.tolist(), show=False)
import matplotlib.pyplot as plt
plt.savefig("explainability/explanation_results.png")
