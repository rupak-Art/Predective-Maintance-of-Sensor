# ModelTraining.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# 1. Load dataset
#data = pd.read_csv("F:\\Project\\ai4i2020_100k.csv")
#data.columns = data.columns.str.strip()

# 2. Select features & target

#features = ["Air temperature[K]", "Process temperature[K]",
#            "Rotational speed[rpm]", "Torque[Nm]", "Tool wear[min]"]

#X = data[features]
# 1. Load dataset
data = pd.read_csv(r"F:\\Project\\ai4i2020_100k.csv")

# 2. Clean column names (remove units like [K], [rpm], [Nm], [min])
data.columns = [re.sub(r"\[.*?\]", "", col).strip() for col in data.columns]

# 3. Select features & target
features = ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]

X = data[features]
y = data["Machine failure"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)


# Save scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# 5. Train multiple models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}


results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    
    # Save model
    joblib.dump(model, f"models/{name}.pkl")
    
    # Save classification report
    with open(f"reports/{name}_report.txt", "w") as f:
        f.write(classification_report(y_test, preds))

# 6. Plot comparison
plt.figure(figsize=(6,4))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.savefig("reports/model_comparison.png")
plt.show()
