
# STEP 1: Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================
# STEP 2: Load the dataset
# ============================
print("\nLoading dataset...")
df = pd.read_csv("ai4i2020_100k.csv")   # <-- put your CSV file name here
print("Dataset loaded successfully!\n")

print("First 5 rows of dataset:")
print(df.head())

# ============================
# STEP 3: Data Preprocessing
# ============================
print("\nPreprocessing data...")

# Drop useless ID columns
df = df.drop(["UDI", "Product ID"], axis=1)

# Encode categorical variable 'Type' into numeric (one-hot encoding)
df = pd.get_dummies(df, columns=["Type"], drop_first=True)

# Features (X) and Target (y)
X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

# Check datatypes
print("\nColumn datatypes after preprocessing:")
print(X.dtypes)

# Train-Test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nData preprocessing complete!")

# ============================
# STEP 4: Model Training
# ============================
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model training complete!")

# ============================
# STEP 5: Model Evaluation
# ============================
print("\nEvaluating model...")

y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score: ", accuracy_score(y_test, y_pred))

# ============================
# STEP 6: Predict on New Data
# ============================
print("\nTesting prediction on a new machine sensor input...")