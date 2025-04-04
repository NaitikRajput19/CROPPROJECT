import pandas as pd
import json

df = pd.read_csv("crop_yield.csv")
df = df.dropna()

dropdown_data = {
    "states": sorted(df['State'].unique().tolist()),
    "crops": sorted(df['Crop'].unique().tolist()),
    "seasons": sorted(df['Season'].unique().tolist())
}

with open("dropdown_data.json", "w") as f:
    json.dump(dropdown_data, f, indent=2)