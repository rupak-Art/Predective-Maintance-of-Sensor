# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import shap

st.set_page_config(page_title="Predictive Maintenance (Pro)", layout="wide")

# =======================================
# Utilities
# =======================================
def safe_load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

@st.cache_resource
def load_artifacts():
    #scaler = safe_load("models/scaler.pkl") or safe_load("scaler.pkl")
    #features = safe_load("models/features.pkl") or safe_load("features.pkl")
    # Optional reliability artifacts
    #y_test = safe_load("y_test.pkl") or safe_load("models/y_test.pkl")
    #y_pred_test = safe_load("y_pred_test.pkl") or safe_load("models/y_pred_test.pkl")
    # Load scaler
    scaler = safe_load("scaler.pkl")
    if scaler is None:
        scaler = safe_load("F:\\Project\\new\\scaler.pkl")

    # Load features
    features = safe_load("features.pkl")
    if features is None:
        features = safe_load("F:\\Project\\new\\features.pkl")

    # Load models
    models = safe_load("models.pkl")
    if models is None:
        models = safe_load("F:\\Project\\model.pkl")

    # Load y_test
    y_test = safe_load("y_test.pkl")
    if y_test is None:
        y_test = safe_load("F:\\Project\\y_test.pkl")

    # Load y_pred_test
    y_pred_test = safe_load("y_pred_test.pkl")
    if y_pred_test is None:
        y_pred_test = safe_load("F:\\Project\\y_pred_test.pkl")


    # Load any available models
    available_models = {}
    for name, fname in [
        ("XGBoost", "F:\\Project\\models\\XGBoost.pkl"),
        ("RandomForest", "F:\\Project\\models\\RandomForest.pkl"),
        ("LightGBM", "F:\\Project\\models\\LightGBM.pkl"),
    ]:
        m = safe_load(fname)
        if m is not None:
            available_models[name] = m

    return scaler, features, available_models, y_test, y_pred_test

scaler, FEATURES, MODELS, Y_TEST, Y_PRED_TEST = load_artifacts()

if FEATURES is None or scaler is None or len(MODELS) == 0:
    st.error(
        "Required artifacts not found. Ensure you have trained and saved:\n"
        "- models/scaler.pkl\n- features.pkl (list of feature names)\n"
        "- At least one model file: models/XGBoost.pkl or RandomForest.pkl or LightGBM.pkl"
    )
    st.stop()

# Reasonable default ranges for UI / simulation (edit to match your data)
RANGES = {
    "Air temperature [K]": (295.0, 321.0, 300.0),
    "Process temperature [K]": (305.0, 341.0, 320.0),
    "Rotational speed [rpm]": (1400.0, 1700.0, 1500.0),
    "Torque [Nm]": (30.0, 45.0, 38.0),
    "Tool wear [min]": (0.0, 250.0, 50.0),
}

# Filter ranges to only those present in FEATURES
RANGES = {f: RANGES.get(f, (0.0, 100.0, 0.0)) for f in FEATURES}

# =======================================
# Sidebar: model selection
# =======================================
st.sidebar.title("⚙️ Settings")
model_name = st.sidebar.selectbox("Select Model", list(MODELS.keys()))
MODEL = MODELS[model_name]

st.sidebar.markdown("**Loaded Artifacts:**")
st.sidebar.write(f"- Scaler: ✅")
st.sidebar.write(f"- Features: {len(FEATURES)}")
st.sidebar.write(f"- Model: {model_name}")

# Prepare SHAP explainer (TreeExplainer works for tree models)
@st.cache_resource(show_spinner=False)
def make_explainer(_model):
    try:
        explainer = shap.TreeExplainer(_model)
        return explainer
    except Exception:
        return None

EXPLAINER = make_explainer(MODEL)

# Synthetic background for SHAP (if needed)
def sample_background(n=200):
    rows = []
    for _ in range(n):
        row = []
        for f in FEATURES:
            lo, hi, default = RANGES[f]
            if "rpm" in f or "min" in f or "K" in f:
                row.append(np.random.uniform(lo, hi))
            else:
                row.append(np.random.uniform(lo, hi))
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURES)

BACKGROUND = sample_background(400)

# =======================================
# Header
# =======================================
st.title("🔧 Predictive Maintenance")
st.caption("Manual & batch predictions • Real-time simulation • Model reliability • SHAP explainability")

tabs = st.tabs([
    "🧪 Manual Prediction",
    "📂 Batch (CSV) Upload",
    "📡 Real-time Simulation",
    "🧾 Model Reliability",
])

# =======================================
# Tab 1: Manual Prediction
# =======================================
with tabs[0]:
    st.subheader("Enter sensor values")
    cols = st.columns(min(4, len(FEATURES)))
    user_vals = {}
    for i, f in enumerate(FEATURES):
        lo, hi, default = RANGES[f]
        with cols[i % len(cols)]:
            user_vals[f] = st.number_input(f, value=float(default), min_value=float(lo), max_value=float(hi))

    if st.button("🔍 Predict", type="primary"):
        input_df = pd.DataFrame([user_vals], columns=FEATURES)
        x_scaled = scaler.transform(input_df)
        pred = int(MODEL.predict(x_scaled)[0])
        if hasattr(MODEL, "predict_proba"):
            prob = float(MODEL.predict_proba(x_scaled)[0][1])
        else:
            # Fallback probability-like score
            prob = float(getattr(MODEL, "predict", lambda z: [0])(x_scaled)[0])
        st.success(f"Prediction: {'⚠️ Failure' if pred==1 else '✅ Healthy'}  |  Failure Probability: {prob:.2f}")

        # Local SHAP explanation for this instance
        #if EXPLAINER is not None:
            #try:
                #shap_values = EXPLAINER(input_df, check_additivity=False)
                #st.markdown("**Top feature contributions (local explanation):**")
                #fig = shap.plots.bar(shap_values[0], show=False)
                #st.pyplot(fig, clear_figure=True)
            #except Exception as e:
                #st.info("SHAP visualization not available for this model/instance.")

# =======================================
# Tab 2: Batch (CSV) Upload
# =======================================
with tabs[1]:
    st.subheader("Upload CSV with the same feature columns used during training")
    file = st.file_uploader("Choose CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        missing = list(set(FEATURES) - set(df.columns))
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            X = df[FEATURES].copy()
            Xs = scaler.transform(X)
            preds = MODEL.predict(Xs)
            probs = MODEL.predict_proba(Xs)[:, 1] if hasattr(MODEL, "predict_proba") else np.zeros(len(preds))
            out = df.copy()
            out["Prediction"] = preds
            out["Failure_Prob"] = probs
            out["Label"] = out["Prediction"].map({0: "Healthy", 1: "Failure"})
            st.dataframe(out.head(50), use_container_width=True)

            # Download
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

            # Distribution chart
            st.markdown("**Prediction distribution**")
            counts = out["Label"].value_counts()
            fig, ax = plt.subplots()
            counts.plot(kind="bar", ax=ax)
            ax.set_ylabel("Count")
            ax.set_xlabel("Class")
            st.pyplot(fig, clear_figure=True)

# =======================================
# Tab 3: Real-time Simulation
# =======================================
with tabs[2]:
    st.subheader("Simulate live sensor data & predictions")
    n_steps = st.number_input("Number of timesteps", min_value=5, max_value=300, value=50, step=5)
    delay = st.slider("Delay per step (seconds)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    run = st.button("▶️ Start Simulation")

    if run:
        holder = st.empty()
        chart_holder = st.empty()
        records = []
        for t in range(int(n_steps)):
            row = {}
            for f in FEATURES:
                lo, hi, _ = RANGES[f]
                row[f] = np.random.uniform(lo, hi)
            X1 = pd.DataFrame([row], columns=FEATURES)
            X1s = scaler.transform(X1)
            pred = int(MODEL.predict(X1s)[0])
            prob = float(MODEL.predict_proba(X1s)[0][1]) if hasattr(MODEL, "predict_proba") else float(pred)
            row.update({"timestep": t+1, "Prediction": pred, "Failure_Prob": prob})
            records.append(row)

            df_live = pd.DataFrame(records)
            holder.dataframe(df_live.tail(10), use_container_width=True)

            # Plot probability over time
            fig, ax = plt.subplots()
            ax.plot(df_live["timestep"], df_live["Failure_Prob"])
            ax.set_ylim(0, 1)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Failure Probability")
            ax.set_title("Live Failure Probability")
            chart_holder.pyplot(fig, clear_figure=True)

            time.sleep(delay)

# =======================================
# Tab 4: Model Reliability
# =======================================
with tabs[3]:
    st.subheader("Test-set performance (saved from training)")
    if Y_TEST is None or Y_PRED_TEST is None:
        st.info("No saved test-set artifacts found (y_test.pkl / y_pred_test.pkl). "
                "Save them in training to show real reliability here.")
    else:
        # Confusion Matrix
        cm = confusion_matrix(Y_TEST, Y_PRED_TEST)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Healthy","Failure"]); ax.set_yticklabels(["Healthy","Failure"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for (i,j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        st.pyplot(fig, clear_figure=True)

        # Classification report
        st.markdown("**Classification Report**")
        try:
            report = classification_report(Y_TEST, Y_PRED_TEST, target_names=["Healthy","Failure"])
        except Exception:
            report = classification_report(Y_TEST, Y_PRED_TEST)
        st.text(report)

# =======================================
# Tab 5: Explainability (Global SHAP)
# =======================================
#with tabs[4]:
    #st.subheader(f"Global feature importance for: {model_name}")
    #if EXPLAINER is None:
        #st.info("SHAP not available for this model.")
    #else:
        #try:
            # Use background sample for a global view
            # Compute SHAP values (may be slow for large background; adjust size above)
            #shap_vals = EXPLAINER(BACKGROUND, check_additivity=False)
            #st.markdown("**Global feature importance (mean |SHAP|)**")
            #fig = shap.plots.bar(shap_vals, show=False)
            #st.pyplot(fig, clear_figure=True)

            #st.markdown("**Beeswarm (distribution of impacts)**")
            #fig2 = shap.plots.beeswarm(shap_vals, show=False, max_display=min(10, len(FEATURES)))
            #st.pyplot(fig2, clear_figure=True)
        #except Exception:
            #st.info("Could not render SHAP plots for this model/type.")
