import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Example feature names (replace these with the actual feature names you used during training)
expected_columns = ['State', 'Crop', 'Season', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Model']

# Example input data (replace with your actual input)
input_data = {
    'State': 'Assam',
    'Crop': 'Coconut',
    'Season': 'Whole Year',
    'Area': 19656,
    'Annual_Rainfall': 2051.4,
    'Fertilizer': 1870661.52,
    'Pesticide': 6093.36,
    'Model': 'cnn'
}

# Convert input data to a pandas DataFrame
input_df = pd.DataFrame([input_data])

# Check if all columns in the input are the same as the model training data
missing_columns = [col for col in expected_columns if col not in input_df.columns]
if missing_columns:
    print(f"Missing columns: {missing_columns}")
    # Handle missing columns appropriately
else:
    print("All required columns are present")

# Example of categorical feature handling (you may need to one-hot encode or label encode categorical columns)
# This is just an example and may not be necessary if your model uses one-hot encoding or label encoding
# For example, if 'State' and 'Crop' are categorical, you may need to encode them

# Example of handling categorical columns (update as per your training preprocessing)
# Convert categorical columns to dummy variables (one-hot encoding) or label encoding
# input_df = pd.get_dummies(input_df, columns=['State', 'Crop', 'Season'])

# Example of input array (ensure correct order and scaling)
input_array = input_df[['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']].values

# Assuming you have a scaler that was fit on the training data (loaded from a file or created during training)
scaler = StandardScaler()

# Apply the same scaler transformation used during model training
# Note: Ensure you fit the scaler on your training data, not the input data
# scaler.fit(training_data)  # This should be your training dataset used to train the model

# Transform the input data (match the shape with the model's input)
try:
    input_array_scaled = scaler.transform(input_array)  # Apply the same scaling to input data
    print(f"Scaled input data: {input_array_scaled}")
except ValueError as e:
    print(f"Error in scaling: {e}")

# Now you can pass the scaled input_array to your model for prediction
