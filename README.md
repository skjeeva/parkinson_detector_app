# 🧠 Parkinson's Disease Detection using Deep Learning

This project detects Parkinson's Disease using a neural network trained on vocal measurements. It integrates explainability through SHAP, a modern and user-friendly web interface via Streamlit, and also supports real-time predictions from the command line.

---

## 🔍 Features

* ✅ Deep Learning with TensorFlow/Keras
* ✅ SHAP for Explainable AI (model interpretation)
* ✅ Intuitive Streamlit Web App
* ✅ Real-time predictions via Command Line Interface (CLI)
* ✅ Hyperparameter tuning support
* ✅ Clean, scalable codebase

---

## 📁 Dataset

The dataset is taken from the [UCI Machine Learning Repository - Parkinson's Disease Data Set](https://archive.ics.uci.edu/ml/datasets/parkinsons), and includes biomedical voice measurements from people with and without Parkinson’s disease.

* `.data` file contains the feature data
* `.names` file explains each feature

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/parkinsons-detector.git
cd parkinsons-detector
```

### 2. Create Virtual Environment and Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python models/model_builder.py
```

This will:

* Train the model
* Save it as `models/model.h5`
* Save the scaler as `models/scaler.pkl`

### 4. Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

A browser window will open with the interactive Parkinson's Disease prediction interface.

### 5. Make CLI Predictions (Optional)

You can test the model via the terminal as well:

```bash
python cli/predict.py --input "samples/sample_input.csv"
```

---

## 🧠 Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) is used to interpret the output of the neural network. It helps you understand **which features influenced the model's prediction**, adding transparency to the decision-making process.

---

## 📌 Dependencies

Main dependencies include:

* `tensorflow`
* `scikit-learn`
* `pandas`
* `numpy`
* `matplotlib`
* `shap`
* `streamlit`
* `joblib`

Install them all using:

```bash
pip install -r requirements.txt
```

---

## ✅ To-Do (Optional Enhancements)

* [ ] Add voice recording input support
* [ ] Deploy app on cloud (e.g., Streamlit Cloud / Hugging Face Spaces)
* [ ] Add support for mobile devices

---

## 📃 License

MIT License.
Feel free to use, fork, and improve the project.

