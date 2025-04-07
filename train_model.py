import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, LSTM
import pickle

# Load dataset
df = pd.read_csv("crop_yield.csv").dropna()

# Drop unrealistic values
df = df[df['Yield'] >= 0]
df = df[df['Yield'] <= 20]  # optional cap on outliers

# Features and target
features = ['State', 'Crop', 'Crop_Year', 'Season', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
target = 'Yield'

# Label Encoding
label_encoders = {}
for col in ['State', 'Crop', 'Season']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Scale only numeric features
numeric_cols = ['Crop_Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Feature matrix and target
X = df[features].values
y = df[target].values

# Save encoders and scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for CNN/LSTM
X_train_seq = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_seq = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# CNN model
cnn_model = Sequential([
    Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='relu')  # prevents negative outputs
])
cnn_model.compile(optimizer='adam', loss='mean_squared_error')
cnn_model.fit(X_train_seq, y_train, epochs=30, batch_size=32, verbose=1)
cnn_model.save("yield_model_cnn.keras")

# LSTM model
lstm_model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], 1)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='relu')  # prevents negative outputs
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_seq, y_train, epochs=30, batch_size=32, verbose=1)
lstm_model.save("yield_model_lstm.keras")

# Evaluation
cnn_pred = cnn_model.predict(X_test_seq).flatten()
lstm_pred = lstm_model.predict(X_test_seq).flatten()

cnn_rmse = np.sqrt(mean_squared_error(y_test, cnn_pred))
cnn_r2 = r2_score(y_test, cnn_pred)
lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred))
lstm_r2 = r2_score(y_test, lstm_pred)

print("\n✅ Model Evaluation:")
print(f"CNN  - RMSE: {cnn_rmse:.2f}, R²: {cnn_r2:.2f}")
print(f"LSTM - RMSE: {lstm_rmse:.2f}, R²: {lstm_r2:.2f}")
