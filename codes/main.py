"""
Predictive Maintenance - Training Script
----------------------------------------
This script:
1. Loads the AI4I dataset (ai4i2020.csv)
2. Prepares data (features + target)
3. Scales numerical values
4. Trains an XGBoost classifier
5. Saves the trained model & scaler for later use in Streamlit app
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# -------------------- STEP 1: Load Dataset --------------------
df = pd.read_csv("ai4i2020_100k.csv")  # download dataset & place in same folder
print("Dataset Loaded. Shape:", df.shape)
print(df.head())

# -------------------- STEP 2: Select Features & Target --------------------
#X = df[["Air temperature [K]", 
        #"Process temperature [K]", 
        #"Rotational speed [rpm]", 
        #"Torque [Nm]", 
        #"Tool wear [min]"]]  # input features
X  = df.drop(columns=["UDI", "Product ID", "Type", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"])    

y = df["Machine failure"]  # target variable (0 = No Failure, 1 = Failure)

# -------------------- STEP 3: Train-Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------- STEP 4: Scaling --------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------- STEP 5: Train Model --------------------
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train_scaled, y_train)

# -------------------- STEP 6: Evaluate --------------------
y_pred_test = model.predict(X_test_scaled)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test, target_names=["Healthy", "Failure"]))

# -------------------- STEP 7: Save Model & Scaler --------------------
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "features.pkl")
joblib.dump(y_test, "y_test.pkl")
joblib.dump(y_pred_test, "y_pred_test.pkl")

print("\n✅ Training complete. Model and scaler saved!")
