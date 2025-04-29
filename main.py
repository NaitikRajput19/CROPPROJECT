from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler, LabelEncoder

app = FastAPI()

# Load models
cnn_model = load_model("yield_model_cnn.keras")
lstm_model = load_model("yield_model_lstm.keras")

# Load scaler and label encoders
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Define feature order
features = ['State', 'Crop', 'Crop_Year', 'Season', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
numeric_cols = ['Crop_Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']  # only these are scaled

class PredictionRequest(BaseModel):
    State: str
    Crop: str
    Crop_Year: int
    Season: str
    Area: float
    Annual_Rainfall: float
    Fertilizer: float
    Pesticide: float
    model: str = "cnn"  # default to CNN


def preprocess_input(data_dict):
    for key in ['State', 'Crop', 'Season']:
        data_dict[key] = data_dict[key].strip()

    # Create input DataFrame
    input_df = pd.DataFrame([{
        'State': data_dict['State'],
        'Crop': data_dict['Crop'],
        'Crop_Year': data_dict['Crop_Year'],
        'Season': data_dict['Season'],
        'Area': data_dict['Area'],
        'Annual_Rainfall': data_dict['Annual_Rainfall'],
        'Fertilizer': data_dict['Fertilizer'],
        'Pesticide': data_dict['Pesticide'],
    }])[features]

    # Encode categorical features
    for col in ['State', 'Crop', 'Season']:
        if data_dict[col] not in label_encoders[col].classes_:
            raise ValueError(f"❌ Error: Unknown label in '{col}': '{data_dict[col]}'")
        input_df[col] = label_encoders[col].transform([input_df[col].iloc[0]])[0]

    # Scale only numeric columns
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Reshape for CNN/LSTM input
    return input_df.values.reshape((1, len(features), 1))


@app.get("/")
def read_root():
    return {"message": "🌾 Crop Yield Prediction API is running!"}


@app.post("/predict")
def predict_yield(request_data: PredictionRequest):
    try:
        data_dict = request_data.dict()
        input_data = preprocess_input(data_dict)

        # Predict from both models
        cnn_prediction = cnn_model.predict(input_data)[0][0]
        lstm_prediction = lstm_model.predict(input_data)[0][0]

        # Get the selected one
        model_type = data_dict["model"].strip().lower()
        if model_type == "cnn":
            selected_prediction = cnn_prediction
        elif model_type == "lstm":
            selected_prediction = lstm_prediction
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return {
            "predicted_yield": round(float(selected_prediction), 2),
            "model_used": model_type,
            "cnn_yield": round(float(cnn_prediction), 2),
            "lstm_yield": round(float(lstm_prediction), 2)
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"❌ Error: {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"🔥 Server Error: {e}")

