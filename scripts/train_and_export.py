import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def main():
    print("=== Training & Exporting Models from Notebook Pipeline ===", flush=True)
    
    csv_path = r"C:\Users\nadia\Downloads\aug_train.csv"
    if not os.path.exists(csv_path):
        csv_path = "aug_train.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find aug_train.csv at {csv_path}")

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print("Data shape:", df.shape)

    # Step 3: Handle Missing Values exactly as notebook
    df_clean = df.copy()
    mode_columns = ["enrolled_university", "education_level", "experience", "last_new_job"]
    mode_values = {}
    for col in mode_columns:
        m = df_clean[col].mode()[0]
        mode_values[col] = m
        df_clean[col] = df_clean[col].fillna(m)

    unknown_columns = ["gender", "major_discipline", "company_size", "company_type"]
    for col in unknown_columns:
        df_clean[col] = df_clean[col].fillna("Unknown")

    # Step 4: Define X / y and Split into Train/Test
    X = df_clean.drop(columns=["target", "enrollee_id"])
    y = df_clean["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Step 6: Encode Categorical Columns
    encoder = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    encoder.fit(X_train[categorical_columns])

    X_train_cat = pd.DataFrame(
        encoder.transform(X_train[categorical_columns]),
        index=X_train.index,
        columns=encoder.get_feature_names_out(categorical_columns)
    )
    X_train_encoded = pd.concat([X_train_cat, X_train[numerical_columns]], axis=1)
    X_train_encoded.columns = X_train_encoded.columns.astype(str)

    # Step 8: Feature Engineering
    experience_num_train = X_train["experience"].replace({"<1": 0, ">20": 21, "Unknown": 0}).astype(float)
    X_train_fe_num = X_train[numerical_columns].copy()
    X_train_fe_num["experience_to_training_ratio"] = experience_num_train / (X_train["training_hours"] + 1)
    X_train_fe_num["has_relevant_degree"] = (X_train["major_discipline"] == "STEM").astype(int)

    # Step 9: Build Feature-Engineered Matrix (181 columns)
    X_train_encoded_fe = pd.concat([X_train_cat, X_train_fe_num], axis=1)
    X_train_encoded_fe.columns = X_train_encoded_fe.columns.astype(str)

    # Output directory
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
    os.makedirs(output_dir, exist_ok=True)

    print("\n1. Training Model 1: KNN Pipeline (StandardScaler + SMOTE + KNN)...", flush=True)
    best_knn = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=33, weights="distance", p=1, n_jobs=1)),
    ])
    best_knn.fit(X_train_encoded, y_train)
    knn_threshold = 0.581
    joblib.dump(best_knn, os.path.join(output_dir, "knn.pkl"))
    print("-> Saved knn.pkl", flush=True)

    print("\n2. Training Model 2: Logistic Regression + SMOTE (Feature Engineered)...", flush=True)
    smote = SMOTE(random_state=42)
    X_train_fe_bal, y_train_fe_bal = smote.fit_resample(X_train_encoded_fe, y_train)
    model_fe_bal = LogisticRegression(max_iter=3000, random_state=42)
    model_fe_bal.fit(X_train_fe_bal, y_train_fe_bal)
    joblib.dump(model_fe_bal, os.path.join(output_dir, "logistic_regression.pkl"))
    print("-> Saved logistic_regression.pkl", flush=True)

    print("\n3. Training Model 3: Tuned Random Forest...", flush=True)
    best_rf = RandomForestClassifier(
        n_estimators=400,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features=0.5,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1
    )
    best_rf.fit(X_train_encoded_fe, y_train)
    joblib.dump(best_rf, os.path.join(output_dir, "random_forest.pkl"))
    print("-> Saved random_forest.pkl", flush=True)

    print("\n4. Training Model 4: Tuned XGBoost (Approach B + SMOTE)...", flush=True)
    def sanitize_columns(cols):
        clean = []
        for c in cols:
            c = str(c).replace("<", "lt").replace(">", "gt")
            c = re.sub(r"[\[\]{}():,]", "_", c)
            clean.append(c)
        return clean

    X_train_xgb = X_train_encoded.copy()
    X_train_xgb.columns = sanitize_columns(X_train_xgb.columns)

    best_params_smote = {
        'subsample': 0.7,
        'reg_lambda': 2,
        'reg_alpha': 0,
        'n_estimators': 150,
        'min_child_weight': 1,
        'max_depth': 5,
        'learning_rate': 0.08,
        'gamma': 0,
        'colsample_bytree': 0.6
    }
    X_tr_b, X_val_b, y_tr_b, y_val_b = train_test_split(
        X_train_xgb, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    X_tr_b_bal, y_tr_b_bal = SMOTE(random_state=42).fit_resample(X_tr_b, y_tr_b)
    
    xgb_model_smote = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        early_stopping_rounds=30,
        **best_params_smote,
    )
    xgb_model_smote.fit(X_tr_b_bal, y_tr_b_bal, eval_set=[(X_val_b, y_val_b)], verbose=False)
    joblib.dump(xgb_model_smote, os.path.join(output_dir, "xgboost.pkl"))
    print("-> Saved xgboost.pkl")

    # Save encoder
    joblib.dump(encoder, os.path.join(output_dir, "encoder.pkl"))
    print("-> Saved encoder.pkl")

    # Collect UI unique categories for selectboxes
    categorical_options = {}
    for col in categorical_columns:
        # unique values from clean training data, sorted
        unique_vals = sorted([str(x) for x in X[col].dropna().unique()])
        if "Unknown" not in unique_vals and col in unknown_columns:
            unique_vals.append("Unknown")
        categorical_options[col] = unique_vals

    # Preprocessor Metadata
    meta = {
        "mode_values": mode_values,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "categorical_options": categorical_options,
        "encoded_179_columns": X_train_encoded.columns.tolist(),
        "encoded_181_columns": X_train_encoded_fe.columns.tolist(),
        "xgb_columns": X_train_xgb.columns.tolist(),
        "knn_threshold": knn_threshold,
    }
    joblib.dump(meta, os.path.join(output_dir, "preprocessor_meta.pkl"))
    print("-> Saved preprocessor_meta.pkl")

    # Feature importances for RF
    rf_feature_importance = pd.DataFrame({
        "Feature": X_train_encoded_fe.columns,
        "Importance": best_rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    # Notebook Benchmarks (Cell 112)
    notebook_benchmarks = {
        "KNN": {
            "Accuracy": 0.762787,
            "Precision": 0.517451,
            "Recall": 0.714136,
            "F1-score": 0.600088,
            "Threshold": 0.581,
            "Balancing": "SMOTE inside pipeline",
        },
        "Logistic Regression + SMOTE": {
            "Accuracy": 0.780271,
            "Precision": 0.542196,
            "Recall": 0.760209,
            "F1-score": 0.632956,
            "Threshold": 0.500,
            "Balancing": "SMOTE on training data",
        },
        "Random Forest": {
            "Accuracy": 0.797756,
            "Precision": 0.571770,
            "Recall": 0.750785,
            "F1-score": 0.649163,
            "Threshold": 0.500,
            "Balancing": "Balanced Subsampling",
        },
        "XGBoost": {
            "Accuracy": 0.788622,
            "Precision": 0.558704,
            "Recall": 0.722513,
            "F1-score": 0.630137,
            "Threshold": 0.500,
            "Balancing": "SMOTE + Early Stopping",
        }
    }

    metrics_pack = {
        "benchmarks": notebook_benchmarks,
        "rf_feature_importance": rf_feature_importance.head(15).to_dict(orient="records"),
    }
    joblib.dump(metrics_pack, os.path.join(output_dir, "notebook_metrics.pkl"))
    print("-> Saved notebook_metrics.pkl")
    print("\nAll training and serialization completed successfully!")

if __name__ == "__main__":
    main()
