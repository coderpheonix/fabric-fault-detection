"""
Fabric Fault Detection - Feature Extraction Script

This script processes fabric images to extract key features for fault detection.
It applies:
1. Edge Detection (for holes)
2. Color Segmentation (for stains using HSV thresholding)
3. Texture Analysis (for vertical defects using Gabor Filters)

The extracted features are stored in a CSV file (`fabric_fault_features.csv`) for further analysis and model training.

Author: Shahidul Islam (Sawon)
GitHub: https://github.com/coderpheonix/fabric_fault_detection
"""

import cv2
import os
import numpy as np
import pandas as pd

# Define dataset folders
dataset_path = r"C:\study\Research\Image for research"
categories = ["Holes", "Stains", "Vertical_Defects", "Defect_Free"]
output_size = (256, 256)  # Ensure images are resized to 256x256


def extract_features(image, category):
    """Extracts edge, color, and texture features based on defect type."""
    features = []

    #  Edge Detection (for holes)
    edges = cv2.Canny(image, 100, 200)  # Canny edge detection
    edge_mean = np.mean(edges)
    features.append(edge_mean)

    #  Color Segmentation (for stains - HSV thresholding)
    if category == "Stains":
        hsv = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR first
        hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)  # Convert to HSV
        h_mean = np.mean(hsv[:, :, 0])  # Hue
        s_mean = np.mean(hsv[:, :, 1])  # Saturation
        v_mean = np.mean(hsv[:, :, 2])  # Value
        features.extend([h_mean, s_mean, v_mean])
    else:
        features.extend([0, 0, 0])  # Placeholder for non-stain images

    # Texture Analysis (for vertical defects - Gabor Filters)
    gabor_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
    texture_mean = np.mean(filtered_img)
    features.append(texture_mean)

    return features


# Initialize dataset
columns = ["Edge_Mean", "H_Mean", "S_Mean", "V_Mean", "Texture_Mean", "Label"]
data = []

# Process images for feature extraction
for category in categories:
    folder = os.path.join(dataset_path, category)
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, output_size)
        features = extract_features(img, category)
        features.append(category)  # Add label
        data.append(features)
        print(f"Processed: {filename} - {category}")

# Convert to DataFrame and save
feature_df = pd.DataFrame(data, columns=columns)
feature_df.to_csv("fabric_fault_features.csv", index=False)
print("Feature extraction completed and saved to fabric_fault_features.csv!")
