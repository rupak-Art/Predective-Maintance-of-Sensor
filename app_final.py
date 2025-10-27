import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Load trained model + scaler
# ------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("F:\Project\model.pkl")      # trained XGBoost or ML model
    scaler = joblib.load("F:\Project\scaler.pkl")    # StandardScaler
    return model, scaler

model, scaler = load_artifacts()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🔧 Predictive Maintenance Demo")
st.write("Enter machine sensor values to predict maintenance needs.")

# Collect inputs with SAME feature names as training
air_temp = st.number_input("Air temperature [K]", min_value=250.0, max_value=400.0, value=300.0)
process_temp = st.number_input("Process temperature [K]", min_value=250.0, max_value=400.0, value=310.0)
rot_speed = st.number_input("Rotational speed [rpm]", min_value=500.0, max_value=3000.0, value=1500.0)
torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0)
tool_wear = st.number_input("Tool wear [min]", min_value=0.0, max_value=500.0, value=10.0)

# Put into DataFrame with EXACT training column names
input_data = pd.DataFrame([{
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rot_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear
}])

# ------------------------------
# Scale input and predict
# ------------------------------
if st.button("🔍 Predict"):
    input_scaled = scaler.transform(input_data)   # apply same scaling
    prediction = model.predict(input_scaled)      # model prediction
    
    st.subheader("📊 Prediction Result")
    if prediction[0] == 0:
        st.success("✅ Machine is Healthy. No immediate maintenance needed.")
    else:
        st.error("⚠️ Maintenance Required! Possible machine failure detected.")
