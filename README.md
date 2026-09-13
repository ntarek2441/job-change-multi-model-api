# Career Transition Predictor

> **AI-Powered Employee Job Change Prediction using Multiple Machine Learning Models**

A professional machine learning web application built with Streamlit that estimates whether an employee is actively seeking a job change or likely to stay.

Rather than relying on a single "black-box" model, this system evaluates candidate profiles across **four distinct ML algorithms** trained and evaluated in the project notebook `ITI_Project(V2) (1).ipynb`:

1. **K-Nearest Neighbors (KNN)** (Distance-weighted, tuned threshold, SMOTE-balanced)
2. **Logistic Regression + SMOTE** (With specialized HR domain feature engineering)
3. **Random Forest** (Tuned ensemble with balanced class weights)
4. **XGBoost** (Gradient boosted decision trees with early stopping & SMOTE)

---

## 🌟 Key Features

- **Multi-Model Consensus Engine**: Simultaneously evaluates an employee profile across all 4 models and calculates ensemble agreement and average risk.
- **Dynamic Probability Progress**: Live probability indicators for *Job Change* vs. *Stay* probabilities calculated dynamically using `predict_proba()`.
- **Domain Feature Engineering**: Reuses exact notebook features:
  - `experience_to_training_ratio = experience_numeric / (training_hours + 1)`
  - `has_relevant_degree = (major_discipline == "STEM")`
- **Explainability & Insights**:
  - Highlights influential features via Random Forest feature importances.
  - Per-model technical insights and behavioral interpretations.
- **Ground-Truth Benchmarks**: Directly displays the cross-validation and test set performance metrics from the original notebook evaluation.

---

## 📂 Project Structure

```
career-prediction/
│
├── app.py                     # Streamlit application UI & prediction engine
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── model/                     # Exported model binaries & preprocessors
│   ├── knn.pkl                # Trained KNN pipeline
│   ├── logistic_regression.pkl# Trained Logistic Regression + SMOTE
│   ├── random_forest.pkl      # Tuned Random Forest Classifier
│   ├── xgboost.pkl            # Tuned XGBoost Classifier
│   ├── encoder.pkl            # Fitted OneHotEncoder
│   ├── preprocessor_meta.pkl  # Imputation modes, column maps & categories
│   └── notebook_metrics.pkl   # Ground truth performance metrics & importances
│
└── scripts/
    └── train_and_export.py    # Standalone script to reproduce training & export models
```

---

## 🚀 Quickstart

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Application
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`.
