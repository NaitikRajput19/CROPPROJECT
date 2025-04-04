import os
import zipfile

project_dir = "crop-yield-prediction"
os.makedirs(project_dir, exist_ok=True)
os.makedirs(f"{project_dir}/api", exist_ok=True)
os.makedirs(f"{project_dir}/web", exist_ok=True)

# Write code files
with open(f"{project_dir}/train_model.py", "w") as f:
    f.write("""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, LSTM
import pickle

# Load and preprocess the dataset
df = pd.read_csv("Crop_Yield.csv")
df = df.dropna()

# Define features and target
features = ['State', 'Crop', 'Season', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
target = 'Yield'

# Encode categorical columns
label_encoders = {}
for col in ['State', 'Crop', 'Season']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split features and target
X = df[features].values
y = df[target].values

# Normalize numerical features (Area, Annual_Rainfall, Fertilizer, Pesticide)
scaler = StandardScaler()
X[:, 3:] = scaler.fit_transform(X[:, 3:])  # Columns 3-6 are numerical

# Save scaler and label encoders
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for CNN and LSTM input
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# MLP + CNN Model
cnn_model = Sequential([
    Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)
])
cnn_model.compile(optimizer='adam', loss='mse')
cnn_model.fit(X_train_cnn, y_train, epochs=20, batch_size=32, verbose=1)
cnn_model.save("yield_model_cnn.h5")

# LSTM Model
lstm_model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=1)
lstm_model.save("yield_model_lstm.h5")

# Evaluate both models
cnn_pred = cnn_model.predict(X_test_cnn).flatten()
lstm_pred = lstm_model.predict(X_test_lstm).flatten()

cnn_rmse = np.sqrt(mean_squared_error(y_test, cnn_pred))
cnn_r2 = r2_score(y_test, cnn_pred)
lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred))
lstm_r2 = r2_score(y_test, lstm_pred)

print("\\nModel Comparison:")
print(f"MLP + CNN - RMSE: {cnn_rmse:.2f}, R²: {cnn_r2:.2f}")
print(f"LSTM      - RMSE: {lstm_rmse:.2f}, R²: {lstm_r2:.2f}")
""")

with open(f"{project_dir}/generate_dropdowns.py", "w") as f:
    f.write("""
import pandas as pd
import json

df = pd.read_csv("Crop_Yield.csv")
df = df.dropna()

dropdown_data = {
    "states": sorted(df['State'].unique().tolist()),
    "crops": sorted(df['Crop'].unique().tolist()),
    "seasons": sorted(df['Season'].unique().tolist())
}

with open("dropdown_data.json", "w") as f:
    json.dump(dropdown_data, f, indent=2)
""")

with open(f"{project_dir}/api/app.py", "w") as f:
    f.write("""
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
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
    app.run(debug=True)
""")

with open(f"{project_dir}/api/requirements.txt", "w") as f:
    f.write("Flask\nnumpy\npandas\nscikit-learn\ntensorflow")

with open(f"{project_dir}/web/index.html", "w") as f:
    f.write("""
<h2>Crop Yield Prediction</h2>
<form id="yieldForm">
  <label for="state">State:</label>
  <select id="state" name="State" required></select><br><br>

  <label for="crop">Crop:</label>
  <select id="crop" name="Crop" required></select><br><br>

  <label for="season">Season:</label>
  <select id="season" name="Season" required></select><br><br>

  <input type="number" step="any" name="Area" placeholder="Area (hectares)" required><br><br>
  <input type="number" step="any" name="Annual_Rainfall" placeholder="Annual Rainfall (mm)" required><br><br>
  <input type="number" step="any" name="Fertilizer" placeholder="Fertilizer (kg)" required><br><br>
  <input type="number" step="any" name="Pesticide" placeholder="Pesticide (kg)" required><br><br>

  <label for="model">Model:</label>
  <select id="model" name="model">
    <option value="cnn">MLP + CNN</option>
    <option value="lstm">LSTM</option>
  </select><br><br>

  <button type="submit">Predict Yield</button>
</form>

<p id="result"></p>

<script>
document.addEventListener("DOMContentLoaded", function() {
  fetch('/wp-content/uploads/dropdown_data.json')  // Update after uploading to WordPress
    .then(response => response.json())
    .then(data => {
      const stateSelect = document.getElementById('state');
      const cropSelect = document.getElementById('crop');
      const seasonSelect = document.getElementById('season');

      data.states.forEach(state => stateSelect.add(new Option(state, state)));
      data.crops.forEach(crop => cropSelect.add(new Option(crop, crop)));
      data.seasons.forEach(season => seasonSelect.add(new Option(season, season)));
    })
    .catch(error => console.error('Error loading dropdown data:', error));
});

document.getElementById("yieldForm").addEventListener("submit", async function(e) {
  e.preventDefault();
  const formData = new FormData(this);
  const data = {
    State: formData.get("State"),
    Crop: formData.get("Crop"),
    Season: formData.get("Season"),
    Area: parseFloat(formData.get("Area")),
    Annual_Rainfall: parseFloat(formData.get("Annual_Rainfall")),
    Fertilizer: parseFloat(formData.get("Fertilizer")),
    Pesticide: parseFloat(formData.get("Pesticide")),
    model: formData.get("model")
  };

  const response = await fetch("http://localhost:5000/predict", {  // Update with Render URL
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });

  const result = await response.json();
  document.getElementById("result").innerText = `Predicted Yield: ${result.predicted_yield} kg/ha (Model: ${result.model_used})`;
});
</script>
""")


with open(f"{project_dir}/README.md", "w") as f:
    f.write("""r# Crop Yield Prediction (India)

This project predicts crop yield using MLP+CNN and LSTM models, integrated with a WordPress site via a Flask API.

## File Structure
- `api/`: Flask API, models, and preprocessing files
- `web/`: HTML form and dropdown data
- `train_model.py`: Trains both models
- `generate_dropdowns.py`: Creates dropdown_data.json

## Setup
1. **Install dependencies**:
   ```bash
   pip install -r api/requirements.txt """)  # Paste the README.md content

# Add generated files (after running scripts)
for file in ["yield_model_cnn.h5", "yield_model_lstm.h5", "scaler.pkl", "label_encoders.pkl"]:
    if os.path.exists(file):
        os.rename(file, f"{project_dir}/api/{file}")
if os.path.exists("dropdown_data.json"):
    os.rename("dropdown_data.json", f"{project_dir}/web/dropdown_data.json")

# Create zip
with zipfile.ZipFile("crop_yield_prediction.zip", "w") as zipf:
    for root, _, files in os.walk(project_dir):
        for file in files:
            zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), project_dir))

print("Zip created: crop_yield_prediction.zip")
