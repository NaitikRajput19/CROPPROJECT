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
cnn_model.compile(optimizer='adam', loss='mean_squared_error')
cnn_model.fit(X_train_cnn, y_train, epochs=20, batch_size=32, verbose=1)
cnn_model.save("yield_model_cnn.h5")

# LSTM Model
lstm_model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=1)
lstm_model.save("yield_model_lstm.h5")

# Evaluate both models
cnn_pred = cnn_model.predict(X_test_cnn).flatten()
lstm_pred = lstm_model.predict(X_test_lstm).flatten()

cnn_rmse = np.sqrt(mean_squared_error(y_test, cnn_pred))
cnn_r2 = r2_score(y_test, cnn_pred)
lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred))
lstm_r2 = r2_score(y_test, lstm_pred)

print("\nModel Comparison:")
print(f"MLP + CNN - RMSE: {cnn_rmse:.2f}, R²: {cnn_r2:.2f}")
print(f"LSTM      - RMSE: {lstm_rmse:.2f}, R²: {lstm_r2:.2f}")