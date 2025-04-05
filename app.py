from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)
CORS(app)  # ✅ Enables cross-origin requests from anywhere

# ✅ Load models
cnn_model = load_model("yield_model_cnn.h5")
lstm_model = load_model("yield_model_lstm.h5")

# ✅ Load scaler and label encoders
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# ✅ Required feature order
features = ['State', 'Crop', 'Season', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

@app.route("/", methods=["GET"])
def home():
    return "🌾 Crop Yield Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("✅ Received data:", data)

        model_type = data.get("model", "cnn")  # 'cnn' or 'lstm'

        # ✅ Convert input to DataFrame
        input_df = pd.DataFrame([{k: v for k, v in data.items() if k != "model"}])

        # ✅ Encode categorical variables
    
for col in ['State', 'Crop', 'Season']:
    val = input_df[col].iloc[0]
    try:
        input_df[col] = label_encoders[col].transform([val])[0]
    except ValueError:
        return jsonify({
            "error": f"❌ Error: '{val}' is not a known label for '{col}'. Please use a valid value from the training dataset."
        }), 400


                

        # ✅ Ensure correct feature order
        input_array = input_df[features].values

        # ✅ Scale numerical features
        input_array[:, 3:] = scaler.transform(input_array[:, 3:])

        # ✅ Reshape input for model
        input_reshaped = input_array.reshape((1, input_array.shape[1], 1))

        # ✅ Select and run model
        model = cnn_model if model_type == "cnn" else lstm_model
        prediction = model.predict(input_reshaped)[0][0]

        return jsonify({
            'predicted_yield': round(float(prediction), 2),
            'model_used': model_type
        })

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({'error': 'Unable to get prediction', 'details': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
