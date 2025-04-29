import streamlit as st
import requests
import matplotlib.pyplot as plt

# Backend FastAPI URL
API_URL = "https://crop-prediction-ykux.onrender.com/predict"

st.title("🌾 Crop Yield Prediction App")
st.markdown("Enter the details below to predict crop yield.")

# Form inputs
state = st.text_input("State", "Assam")
crop = st.text_input("Crop", "Arecanut")
crop_year = st.number_input("Crop Year", min_value=1950, max_value=2050, value=2020)
season = st.text_input("Season", "Whole Year")
area = st.number_input("Area (hectares)", value=1000.0)
rainfall = st.number_input("Annual Rainfall (mm)", value=2000.0)
fertilizer = st.number_input("Fertilizer (kg/ha)", value=100000.0)
pesticide = st.number_input("Pesticide (kg/ha)", value=1000.0)
model = st.selectbox("Choose Model", ["cnn", "lstm"])

# Prediction button
if st.button("Predict"):
    with st.spinner("Sending data to the backend..."):
        payload = {
            "State": state,
            "Crop": crop,
            "Crop_Year": crop_year,
            "Season": season,
            "Area": area,
            "Annual_Rainfall": rainfall,
            "Fertilizer": fertilizer,
            "Pesticide": pesticide,
            "model": model
        }

        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()

            if "predicted_yield" in result:
                # Display selected model prediction
                st.success(f"✅ Predicted Yield: **{result['predicted_yield']} tons/ha** using **{result['model_used'].upper()}** model")

                # Optional: show both model predictions if available
                cnn_yield = result.get("cnn_yield")
                lstm_yield = result.get("lstm_yield")

                if cnn_yield and lstm_yield:
                    st.subheader("📊 CNN vs LSTM Yield Prediction")
                    models = ["CNN", "LSTM"]
                    yields = [cnn_yield, lstm_yield]

                    fig, ax = plt.subplots()
                    bars = ax.bar(models, yields, color=["#4CAF50", "#2196F3"])
                    ax.set_ylabel("Yield (tons/ha)")
                    ax.set_title("Model Output Comparison")

                    for bar in bars:
                        yval = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f"{yval:.2f}", ha='center')

                    st.pyplot(fig)

            else:
                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"❌ Could not connect to backend: {e}")
