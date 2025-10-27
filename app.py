import streamlit as st
import pandas as pd
import joblib

# ============================
# Load Model & Scaler
# ============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("F:\Project\model.pkl")
    scaler = joblib.load("F:\Project\scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ============================
# App Title
# ============================
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="centered")
st.title("🔧 Predictive Maintenance Dashboard")
st.write("Upload sensor data or enter values manually to predict machine failure.")

# ============================
# File Upload Section
# ============================
uploaded_file = st.file_uploader("📂 Upload a CSV file with sensor data", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data")
    st.dataframe(data.head())

    # Scale features
    data_scaled = scaler.transform(data)

    # Predictions
    predictions = model.predict(data_scaled)
    data["Prediction"] = predictions

    st.subheader("✅ Predictions")
    st.dataframe(data)

    # Download option
    csv_output = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Predictions as CSV",
        data=csv_output,
        file_name="predictions.csv",
        mime="text/csv",
    )

# ============================
# Manual Input Section
# ============================
st.subheader("📝 Try Manual Input")

with st.form("manual_input_form"):
    st.write("Enter sensor values to predict machine failure:")

    # Collect inputs with SAME feature names as training
    air_temp = st.number_input("Air temperature [K]", value=250.0)
    process_temp = st.number_input("Process temperature [K]", value=310.0)
    rot_speed = st.number_input("Rotational speed [rpm]", value=1500.0)
    torque = st.number_input("Torque [Nm]", min_value=0.0,value=40.0)
    tool_wear = st.number_input("Tool wear [min]", value=10.0)


    submitted = st.form_submit_button("Predict")

    if submitted:
        # Build a DataFrame for input
        input_data = pd.DataFrame([{
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rot_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
    }])
        

        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.success(f"🔮 Prediction: {'⚠️ Failure' if prediction == 1 else '✅ No Failure'}")
