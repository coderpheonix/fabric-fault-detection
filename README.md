# Automated Fabric Fault Detection Using Python and OpenCV

## Overview
This project presents a **Machine Vision-Based Approach** for **Automated Fabric Fault Detection** using **Python and OpenCV**. The goal is to detect various fabric defects using image processing and machine learning techniques.

## Research Objectives
- Develop an automated system to detect common fabric defects.
- Use image processing techniques to identify different types of defects.
- Train machine learning models for classification and accuracy comparison.

## Methodology
### 1. **Data Collection**
- Fabric images were collected, including both **defective and defect-free** samples.
- Categories of fabric defects analyzed:
  - **Holes** – Detected using Edge & Contour Detection.
  - **Stains** – Identified using Color Segmentation (HSV thresholding).
  - **Vertical Defects** – Recognized using Texture Analysis (Gabor Filters or FFT).
  - **Defect-Free Images** – Used for model training and comparison.

### 2. **Preprocessing & Feature Extraction**
- Image enhancement techniques applied.
- Texture and color-based feature extraction.

### 3. **Machine Learning Models**
- **Support Vector Machine (SVM)**
  - Accuracy: **85.29%**
- **Random Forest**
  - Accuracy: **87.06%**

## Results
- The **Random Forest model outperformed SVM** in classification accuracy.
- The proposed approach successfully identified fabric defects with high precision.

## Installation & Usage
### 1. **Clone the Repository**
```bash
git clone https://github.com/coderpheonix/fabric-fault-detection.git
cd fabric-fault-detection
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Run the Detection Script**
```bash
python detect_faults.py
```

### 4. **Train the Model (Optional)**
```bash
python train_model.py
```


```

## Future Work
- Improve detection accuracy by integrating **deep learning models (CNNs)**.
- Optimize real-time detection using high-performance computing techniques.

## Citation
If you use this research in your work, please cite it as:
```
Shahidul Islam, "Automated Fabric Fault Detection Using Python and OpenCV: A Machine Vision-Based Approach," 2025.
```

## License
This project is open-source and available under the **MIT License**.

---
🚀 **Developed by**: Shahidul Islam (@coderpheonix)

