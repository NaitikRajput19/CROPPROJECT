from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
import os  # <- Add this line
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load models
cnn_model = load_model("yield_model_cnn.h5")
lstm_model = load_model("yield_model_lstm.h5")

# Load scaler and label encoders
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Feature order
features = ['State', 'Crop', 'Season', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

@app.route("/", methods=["GET"])
def home():
    return "Crop Yield Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    model_type = data.get("model", "cnn")  # Default to CNN

    # Convert input to DataFrame
    input_df = pd.DataFrame([{k: v for k, v in data.items() if k != "model"}])

    # Encode categorical variables
    for col in ['State', 'Crop', 'Season']:
        input_df[col] = label_encoders[col].transform([input_df[col].iloc[0]])[0]

    # Ensure correct feature order
    input_array = input_df[features].values

    # Scale numerical features
    input_array[:, 3:] = scaler.transform(input_array[:, 3:])

    # Reshape for model input
    input_reshaped = input_array.reshape((1, input_array.shape[1], 1))

    # Select model and predict
    model = cnn_model if model_type == "cnn" else lstm_model
    prediction = model.predict(input_reshaped)[0][0]
    return jsonify({'predicted_yield': round(float(prediction), 2), 'model_used': model_type})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
