"""
Fabric Fault Detection using Machine Learning (SVM & Random Forest)
--------------------------------------------------------------
This script is designed to classify fabric faults using:
 Support Vector Machine (SVM) with Hyperparameter Tuning
 Random Forest (RF) Classifier

**Workflow:**
    1. Load dataset containing extracted fabric fault features.
    2. Normalize features for better model performance (especially for SVM).
    3. Split data into training (80%) and testing (20%).
    4. Train SVM with GridSearchCV for hyperparameter tuning.
    5. Train a Random Forest classifier for comparison.
    6. Evaluate both models and compare accuracy.
    7. Save the best-performing model for future predictions.
    8. Save the StandardScaler to maintain data consistency.

 **Output:**
    - Accuracy and classification report for both models.
    - Saves the best model (`fabric_fault_detector_svm.pkl` or `fabric_fault_detector_rf.pkl`).
    - Saves the scaler (`scaler.pkl`) for consistent feature scaling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
file_path = "fabric_fault_features.csv"  # Update with your actual file path
df = pd.read_csv(file_path)

# Separate features & labels
X = df.drop(columns=["Label"])  # Features
y = df["Label"]  # Target labels

# Normalize features (important for SVM)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training (80%) & testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM with Hyperparameter Tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1]
}

svm_grid = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"), param_grid, cv=5)
svm_grid.fit(X_train, y_train)

# Best SVM Model
svm_model = svm_grid.best_estimator_
svm_pred = svm_model.predict(X_test)

# Random Forest Classifier (for comparison)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Model Evaluation
print(f"Best SVM Parameters: {svm_grid.best_params_}")
print("\nSVM Model Performance:")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

print("\nRandom Forest Model Performance:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# Save the Best Model (SVM or RF, whichever is better)
if accuracy_score(y_test, svm_pred) > accuracy_score(y_test, rf_pred):
    joblib.dump(svm_model, "fabric_fault_detector_svm.pkl")
    print("\nSaved Best Model: SVM")
else:
    joblib.dump(rf_model, "fabric_fault_detector_rf.pkl")
    print("\nSaved Best Model: Random Forest")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")
