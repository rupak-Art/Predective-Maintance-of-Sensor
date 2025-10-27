import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ------------------------------
# Load artifacts
# ------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("F:\\Project\\model.pkl")
    scaler = joblib.load("F:\\Project\\scaler.pkl")
    features = joblib.load("F:\\Project\\features.pkl")
    # These must be saved during training
    y_test = joblib.load("F:\\Project\\y_test.pkl")
    y_pred_test = joblib.load("F:\\Project\\y_pred_test.pkl")
    return model, scaler, features, y_test, y_pred_test

model, scaler, features, y_test, y_pred_test = load_artifacts()

# ------------------------------
# UI
# ------------------------------
st.title("🔧 Predictive Maintenance Dashboard")
st.write("Enter values manually or upload a CSV file to predict machine failures.")

option = st.radio("Choose Prediction Mode:", ["Manual Input", "Batch (CSV Upload)"])

# ------------------------------
# Manual Input Mode
# ------------------------------y_
if option == "Manual Input":
    user_input = {}
    for col in features:
        user_input[col] = st.number_input(col, value=0.0)

    data = pd.DataFrame([user_input])

    if st.button("🔍 Predict"):
        data_scaled = scaler.transform(data)
        pred = model.predict(data_scaled)[0]
        prob = model.predict_proba(data_scaled)[0][1]

        st.subheader("📊 Prediction Result")
        if pred == 0:
            st.success(f"✅ Machine is Healthy (Failure probability: {prob:.2f})")
        else:
            st.error(f"⚠️ Maintenance Required! Failure probability: {prob:.2f}")

# ------------------------------
# Batch Mode
# ------------------------------
else:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Ensure correct columns
        missing_cols = set(features) - set(df.columns)
        if missing_cols:
            st.error(f"❌ Missing columns in uploaded file: {missing_cols}")
        else:
            df = df[features]  # Keep only needed columns
            df_scaled = scaler.transform(df)
            preds = model.predict(df_scaled)

            df["Prediction"] = preds
            df["Prediction_Label"] = df["Prediction"].map({0: "Healthy", 1: "Failure"})

            st.subheader("📑 Predictions")
            st.dataframe(df.head(20))

            # ------------------------------
            # Class Distribution Bar Chart
            # ------------------------------
            st.subheader("📊 Prediction Distribution")
            class_counts = df["Prediction_Label"].value_counts()

            fig, ax = plt.subplots()
            class_counts.plot(kind="bar", ax=ax)
            ax.set_title("Prediction Distribution")
            ax.set_ylabel("Count")
            st.pyplot(fig)

# ------------------------------
# Model Reliability Section
# ------------------------------
st.markdown("---")
st.subheader("🧪 Model Reliability (on Test Set)")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Healthy", "Failure"],
            yticklabels=["Healthy", "Failure"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)

# Classification Report
st.text("Classification Report (on Test Set):")
report = classification_report(y_test, y_pred_test, target_names=["Healthy", "Failure"])
st.text(report)
