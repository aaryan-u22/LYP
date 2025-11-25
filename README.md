# Chaos-Enhanced ECG Arrhythmia Detection Framework

## 1. Project Overview
This research project investigates the efficacy of integrating **Nonlinear Dynamics (Chaos Theory)** with traditional statistical signal processing to develop a computationally efficient and interpretable alternative to Deep Learning for automated cardiac arrhythmia detection.

The study implements a rigorous comparative analysis across three distinct methodological tracks. For each track, multiple candidate algorithms were evaluated, and the optimal model was selected based on a balance of classification performance (F1-score) and computational efficiency.

---

## 2. Methodology

### **Track 1: Traditional Baseline**
**Objective:** Establish a performance baseline using established statistical feature engineering.  
**Features:** Temporal and morphological statistics, including Pre-RR interval, Post-RR interval, Local Average RR, and R-peak Amplitude.  
**Selected Model:** **Random Forest Classifier.** This model was selected for its robustness to noise and ability to handle tabular data effectively compared to other traditional classifiers.

---

### **Track 2: Modern Benchmark**
**Objective:** Determine the state-of-the-art performance ceiling using Deep Learning.  
**Features:** Raw, denoised ECG voltage signals (Time-domain representation).  
**Selected Model:** **1D-Convolutional Neural Network (1D-CNN).** The 1D-CNN architecture was chosen for its superior ability to extract hierarchical spatial features from time-series data compared to standard RNNs.

---

### **Track 3: Proposed Hybrid Framework**
**Objective:** Validate the hypothesis that dynamical stability metrics can enhance predictive performance while maintaining low computational cost.  

**Features:**  
A hybrid feature vector combining:
- **Chaos Invariants:** Largest Lyapunov Exponent (LLE), Higuchi Fractal Dimension (FD), Sample Entropy (SampEn).  
- **Recurrence Quantification Analysis (RQA):** Recurrence Rate (RR), Determinism (DET), Laminarity (LAM).  
- **Statistical Features:** R-R Intervals and Amplitude.

**Selected Model:** **XGBoost Classifier.** Selected for its state-of-the-art performance on structured data, handling of feature interactions, and native support for interpretability via SHAP values.

---

## 3. Results
The experimental results demonstrate that the Proposed Hybrid Method achieves superior performance metrics compared to both the Traditional Baseline and the Deep Learning Benchmark.

**Conclusion:**  
The Hybrid XGBoost model outperforms the 1D-CNN by **2.59% in Accuracy** and **5.8% in Sensitivity**, verifying that the integration of nonlinear dynamical features provides a significant diagnostic advantage over raw signal processing alone.

---

## 4. Explainability Analysis (SHAP)
To address the "black box" limitation of Deep Learning models, the proposed XGBoost model was analyzed using **SHAP (SHapley Additive exPlanations).**

The analysis of feature importance reveals the following physiological insights:

- **Dominance of Timing:**  
  Pre_RR (the interval preceding the beat) is the primary predictor, confirming that rhythm irregularity is the strongest indicator of arrhythmia.

- **Contribution of Chaos:**  
  FD (Fractal Dimension) and SampEn (Sample Entropy) appear as high-ranking features, validating that variations in signal complexity and regularity are critical for correctly classifying edge cases that timing features alone may miss.

- **Morphological Relevance:**  
  Amplitude remains a top feature, indicating that voltage magnitude changes are essential for distinguishing ventricular anomalies.

---

## 5. Project Structure
```plaintext
ECG_Research_Project/
├── data/                   
│   ├── raw/                # MIT-BIH Arrhythmia Database source files
│   └── processed/          # Generated feature sets for all tracks
│       ├── 1_traditional/  # Statistical features (CSV)
│       ├── 2_modern/       # Raw signal arrays (NPY)
│       └── 3_proposed/     # Hybrid Chaos+Stats features (CSV)
│
├── models/                 # Serialized models and evaluation artifacts
│   ├── 1_traditional/      # Random Forest (.pkl)
│   ├── 2_modern/           # CNN (.keras)
│   ├── 3_proposed_xgb/     # XGBoost (.pkl) and SHAP plots
│   └── final_charts/       # Comparative visualization
│
├── src/                    # Shared Logic Library
│   ├── features/           # Feature Extraction Modules
│   │   ├── physics.py      # Chaos Theory Engine (LLE, FD, RQA)
│   │   └── statistical.py  # Statistical Engine (Timing/Morphology)
│   └── data_loader.py      # Signal Processing (Denoising, Segmentation)
│
└── scripts/                # Execution Pipelines
    ├── 1_track_traditional/
    ├── 2_track_modern/
    └── 3_track_proposed/   # Proposed Methodology
        ├── 1_etl_combined.py  # Feature Extraction Pipeline
        └── 2_train_xgb.py     # Model Training & Evaluation
```

---

## 6. Replication Instructions

### Prerequisites
Install the required Python packages:
```
pip install -r requirements.txt
```

### Execution

#### 1. Feature Extraction
Run the ETL pipeline to generate the hybrid dataset:
```
python scripts/3_track_proposed/1_etl_combined.py
```

#### 2. Model Training & Evaluation
Train the XGBoost model and generate SHAP analysis:
```
python scripts/3_track_proposed/2_train_xgb.py
```

#### 3. Visualization
Generate the final comparative charts:
```
python scripts/generate_final_charts.py
```
