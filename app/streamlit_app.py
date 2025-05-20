import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_scaler_model

st.set_page_config(page_title="Parkinson's Disease Detector", layout="centered")

# Title and instructions
st.title("🧠 Parkinson's Disease Detector")
st.markdown("Upload a CSV with voice features (excluding `name` and `status`). The model will predict Parkinson's disease presence.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your CSV file", type="csv")

# Load model and scaler
scaler, model = load_scaler_model()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(df)

    # Prediction Button
    if st.button("🔍 Predict Parkinson's"):
        try:
            # Drop non-feature columns if present
            X = df.drop(columns=["name", "status"], errors="ignore")

            # Scale input features
            scaled = scaler.transform(X)

            # Model Prediction
            predictions = (model.predict(scaled) > 0.5).astype(int)
            df["Prediction"] = predictions

            st.subheader("🧠 Prediction Results")
            st.dataframe(df)

            # Summary
            positive_cases = int(predictions.sum())
            total_cases = len(predictions)

            st.success(f"✅ {positive_cases} out of {total_cases} samples predicted as Parkinson's Positive.")

            # Pie Chart
            st.subheader("📊 Prediction Summary")
            fig, ax = plt.subplots()
            ax.pie(
                [positive_cases, total_cases - positive_cases],
                labels=["Parkinson's", "Healthy"],
                autopct="%1.1f%%",
                startangle=90,
                colors=["#ff6361", "#58508d"]
            )
            ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
            st.pyplot(fig)

            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv,
                file_name="parkinsons_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
