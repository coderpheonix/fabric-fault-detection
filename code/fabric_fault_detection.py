"""
Fabric Fault Detection using Machine Learning
=============================================
This script trains two machine learning models—Support Vector Machine (SVM) and
Random Forest Classifier—to classify fabric defects based on extracted features.
 Steps in this script:
 Load the dataset (`fabric_defect_features.csv`)
 Preprocess the data (feature normalization)
 Split the dataset into training and testing sets
Train two models:
   - Support Vector Machine (SVM)
   - Random Forest Classifier
 Evaluate both models on test data
 Save the best-performing model (`fabric_fault_detector.pkl`) and the scaler (`scaler.pkl`)

Author: Shahidul Islam (Sawon)
Date: March 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

#  Step 1: Load the dataset
df = pd.read_csv("fabric_defect_features.csv")  # Replace with your actual CSV file

#  Step 2: Preprocess the data
# Assuming the last column is the label (Hole, Stain, Vertical_Defect, Defect_Free)
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values   # Labels

# Normalize the features (optional but improves performance)
scaler = StandardScaler()
X = scaler.fit_transform(X)

#  Step 3: Split the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Step 4: Train models

# Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#  Step 5: Model evaluation
svm_pred = svm_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

print(" SVM Model Performance:")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, svm_pred))

print("\n Random Forest Model Performance:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# 🔹 Step 6: Save the best model
joblib.dump(rf_model, "fabric_fault_detector.pkl")  # Saving the trained Random Forest model
joblib.dump(scaler, "scaler.pkl")  # Saving the scaler for normalization

print("\n Model training and evaluation completed! Best model saved.")
