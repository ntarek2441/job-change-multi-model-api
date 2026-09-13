import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


# ============================================================
# CONFIG
# ============================================================

EXPORT_DIR = "model"
DATA_PATH = "aug_train.csv"

os.makedirs(EXPORT_DIR, exist_ok=True)

print("=" * 70)
print("JOB CHANGE - 7 MODEL EXPORT")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)
df_clean = df.copy()

print(f"Dataset shape: {df_clean.shape}")


# ============================================================
# CLEANING
# ============================================================

mode_columns = [
    "enrolled_university",
    "education_level",
    "experience",
    "last_new_job",
]

for col in mode_columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])


unknown_columns = [
    "gender",
    "major_discipline",
    "company_size",
    "company_type",
]

for col in unknown_columns:
    df_clean[col] = df_clean[col].fillna("Unknown")


# ============================================================
# X / y
# ============================================================

X = df_clean.drop(columns=["target", "enrollee_id"])
y = df_clean["target"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# ============================================================
# COLUMN TYPES
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


# ============================================================
# ENCODER
# ============================================================

encoder = OneHotEncoder(
    handle_unknown="ignore",
    drop="first",
    sparse_output=False,
)

encoder.fit(X_train[categorical_columns])


# ============================================================
# BASIC ENCODING
# ============================================================

X_train_cat = pd.DataFrame(
    encoder.transform(X_train[categorical_columns]),
    index=X_train.index,
    columns=encoder.get_feature_names_out(categorical_columns),
)

X_test_cat = pd.DataFrame(
    encoder.transform(X_test[categorical_columns]),
    index=X_test.index,
    columns=encoder.get_feature_names_out(categorical_columns),
)


X_train_encoded = pd.concat(
    [
        X_train_cat,
        X_train[numerical_columns],
    ],
    axis=1,
)

X_test_encoded = pd.concat(
    [
        X_test_cat,
        X_test[numerical_columns],
    ],
    axis=1,
)

X_train_encoded.columns = X_train_encoded.columns.astype(str)
X_test_encoded.columns = X_test_encoded.columns.astype(str)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df_in):

    df_fe = df_in.copy()

    exp_mapping = {
        "<1": 0,
        ">20": 21,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "11": 11,
        "12": 12,
        "13": 13,
        "14": 14,
        "15": 15,
        "16": 16,
        "17": 17,
        "18": 18,
        "19": 19,
        "20": 20,
    }

    num_exp = (
        df_fe["experience"]
        .map(exp_mapping)
        .fillna(0)
    )

    df_fe["experience_to_training_ratio"] = (
        num_exp / (df_fe["training_hours"] + 1e-5)
    )

    # IMPORTANT:
    # This matches the notebook definition.
    df_fe["has_relevant_degree"] = (
        df_fe["major_discipline"] == "STEM"
    ).astype(int)

    return df_fe[
        [
            "experience_to_training_ratio",
            "has_relevant_degree",
        ]
    ]


X_train_fe_extra = create_features(X_train)
X_test_fe_extra = create_features(X_test)


X_train_encoded_fe = pd.concat(
    [
        X_train_encoded,
        X_train_fe_extra,
    ],
    axis=1,
)

X_test_encoded_fe = pd.concat(
    [
        X_test_encoded,
        X_test_fe_extra,
    ],
    axis=1,
)

X_train_encoded_fe.columns = X_train_encoded_fe.columns.astype(str)
X_test_encoded_fe.columns = X_test_encoded_fe.columns.astype(str)


# ============================================================
# ALIGN TEST FEATURES EXACTLY WITH TRAIN FEATURES
# ============================================================

X_test_encoded = X_test_encoded[
    X_train_encoded.columns
]

X_test_encoded_fe = X_test_encoded_fe[
    X_train_encoded_fe.columns
]


# ============================================================
# SMOTE
# ============================================================

smote = SMOTE(random_state=42)

X_train_bal, y_train_bal = smote.fit_resample(
    X_train_encoded,
    y_train,
)


smote_fe = SMOTE(random_state=42)

X_train_bal_fe, y_train_bal_fe = smote_fe.fit_resample(
    X_train_encoded_fe,
    y_train,
)


# ============================================================
# 1. KNN
# ============================================================

print("\nTraining KNN...")

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(
    X_train_encoded,
    y_train,
)

joblib.dump(
    knn,
    os.path.join(EXPORT_DIR, "knn.pkl"),
)


# ============================================================
# 2. LOGISTIC REGRESSION + SMOTE
# ============================================================

print("Training LR + SMOTE...")

lr_smote = LogisticRegression(
    max_iter=3000,
    random_state=42,
)

lr_smote.fit(
    X_train_bal,
    y_train_bal,
)

joblib.dump(
    lr_smote,
    os.path.join(
        EXPORT_DIR,
        "logistic_regression_smote.pkl",
    ),
)


# ============================================================
# 3. FE LR + SMOTE
# ============================================================

print("Training FE LR + SMOTE...")

fe_lr_smote = LogisticRegression(
    max_iter=3000,
    random_state=42,
)

fe_lr_smote.fit(
    X_train_bal_fe,
    y_train_bal_fe,
)

joblib.dump(
    fe_lr_smote,
    os.path.join(
        EXPORT_DIR,
        "fe_logistic_regression_smote.pkl",
    ),
)


# ============================================================
# 4. BASELINE RANDOM FOREST
# ============================================================

print("Training Baseline Random Forest...")

baseline_rf = RandomForestClassifier(
    random_state=42,
)

baseline_rf.fit(
    X_train_encoded_fe,
    y_train,
)

joblib.dump(
    baseline_rf,
    os.path.join(
        EXPORT_DIR,
        "random_forest.pkl",
    ),
)


# ============================================================
# 5. TUNED RANDOM FOREST
# ============================================================

print("Training Tuned Random Forest...")

tuned_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
)

tuned_rf.fit(
    X_train_encoded_fe,
    y_train,
)

joblib.dump(
    tuned_rf,
    os.path.join(
        EXPORT_DIR,
        "tuned_random_forest.pkl",
    ),
)


# ============================================================
# 6. XGBOOST + scale_pos_weight
# ============================================================

print("Training XGBoost + scale_pos_weight...")

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos = negative / positive

xgb_spw = XGBClassifier(
    scale_pos_weight=scale_pos,
    random_state=42,
    eval_metric="logloss",
)

xgb_spw.fit(
    X_train_encoded,
    y_train,
)

joblib.dump(
    xgb_spw,
    os.path.join(
        EXPORT_DIR,
        "xgboost_scale_pos_weight.pkl",
    ),
)


# ============================================================
# 7. XGBOOST + SMOTE
# ============================================================

print("Training XGBoost + SMOTE...")

xgb_smote = XGBClassifier(
    random_state=42,
    eval_metric="logloss",
)

xgb_smote.fit(
    X_train_bal,
    y_train_bal,
)

joblib.dump(
    xgb_smote,
    os.path.join(
        EXPORT_DIR,
        "xgboost_smote.pkl",
    ),
)


# ============================================================
# MODELS
# ============================================================

models = {
    "KNN": (
        knn,
        X_test_encoded,
    ),
    "LR + SMOTE": (
        lr_smote,
        X_test_encoded,
    ),
    "FE LR + SMOTE": (
        fe_lr_smote,
        X_test_encoded_fe,
    ),
    "Baseline Random Forest": (
        baseline_rf,
        X_test_encoded_fe,
    ),
    "Tuned Random Forest": (
        tuned_rf,
        X_test_encoded_fe,
    ),
    "XGBoost + scale_pos_weight": (
        xgb_spw,
        X_test_encoded,
    ),
    "XGBoost + SMOTE": (
        xgb_smote,
        X_test_encoded,
    ),
}


# ============================================================
# METRICS
# ============================================================

metrics = {}

print("\n" + "=" * 70)
print("MODEL METRICS")
print("=" * 70)

for name, (model, X_eval) in models.items():

    predictions = model.predict(X_eval)
    probabilities = model.predict_proba(X_eval)[:, 1]

    result = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities,
            )
        ),
    }

    metrics[name] = result

    print(f"\n{name}")
    print(f"Accuracy : {result['accuracy']:.6f}")
    print(f"Precision: {result['precision']:.6f}")
    print(f"Recall   : {result['recall']:.6f}")
    print(f"F1       : {result['f1']:.6f}")
    print(f"ROC-AUC  : {result['roc_auc']:.6f}")


# ============================================================
# BEST MODEL
# ============================================================

best_model_name = max(
    metrics,
    key=lambda name: metrics[name]["accuracy"],
)

# Primary model is intentionally Baseline RF
primary_model_name = "Baseline Random Forest"


# ============================================================
# METRICS EXPORT
# ============================================================

metrics_payload = {
    "primary_model": primary_model_name,
    "best_model_by_accuracy": best_model_name,
    "models": metrics,
}

with open(
    os.path.join(EXPORT_DIR, "metrics.json"),
    "w",
) as f:

    json.dump(
        metrics_payload,
        f,
        indent=2,
    )


# ============================================================
# METADATA
# ============================================================

metadata = {
    "primary_model": primary_model_name,

    "categorical_columns": categorical_columns,

    "numerical_columns": numerical_columns,

    "feature_order_original": [
        str(x)
        for x in X_train_encoded.columns
    ],

    "feature_order_fe": [
        str(x)
        for x in X_train_encoded_fe.columns
    ],

    "models": {
        "KNN": {
            "file": "knn.pkl",
            "feature_type": "original",
        },

        "LR + SMOTE": {
            "file": "logistic_regression_smote.pkl",
            "feature_type": "original",
        },

        "FE LR + SMOTE": {
            "file": "fe_logistic_regression_smote.pkl",
            "feature_type": "feature_engineered",
        },

        "Baseline Random Forest": {
            "file": "random_forest.pkl",
            "feature_type": "feature_engineered",
        },

        "Tuned Random Forest": {
            "file": "tuned_random_forest.pkl",
            "feature_type": "feature_engineered",
        },

        "XGBoost + scale_pos_weight": {
            "file": "xgboost_scale_pos_weight.pkl",
            "feature_type": "original",
        },

        "XGBoost + SMOTE": {
            "file": "xgboost_smote.pkl",
            "feature_type": "original",
        },
    },

    "feature_engineering": {
        "experience_to_training_ratio": (
            "numeric_experience / (training_hours + 1e-5)"
        ),
        "has_relevant_degree": (
            "major_discipline == STEM"
        ),
    },

    "threshold": 0.5,
}

with open(
    os.path.join(EXPORT_DIR, "metadata.json"),
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


# ============================================================
# INSIGHTS
# ============================================================

total_candidates = len(df_clean)

looking_for_change = int(
    df_clean["target"].sum()
)

not_looking_for_change = (
    total_candidates - looking_for_change
)


def grouped_rates(column):

    result = (
        df_clean
        .groupby(column)["target"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    return {
        str(k): float(v)
        for k, v in result.items()
    }


factors = [
    "education_level",
    "experience",
    "company_size",
    "company_type",
    "last_new_job",
]

factor_rates = {}

for col in factors:

    grouped = (
        df_clean
        .groupby(col)["target"]
        .mean()
    )

    factor_rates[col] = float(
        grouped.max()
    )


# ============================================================
# TOP 10 RANKING
# ============================================================

X_all = df_clean.drop(
    columns=["target", "enrollee_id"]
).copy()


X_all_cat = pd.DataFrame(
    encoder.transform(
        X_all[categorical_columns]
    ),
    index=X_all.index,
    columns=encoder.get_feature_names_out(
        categorical_columns
    ),
)


X_all_encoded = pd.concat(
    [
        X_all_cat,
        X_all[numerical_columns],
    ],
    axis=1,
)

X_all_encoded.columns = (
    X_all_encoded.columns.astype(str)
)


X_all_fe_extra = create_features(
    X_all
)


X_all_encoded_fe = pd.concat(
    [
        X_all_encoded,
        X_all_fe_extra,
    ],
    axis=1,
)

X_all_encoded_fe.columns = (
    X_all_encoded_fe.columns.astype(str)
)

X_all_encoded_fe = (
    X_all_encoded_fe[
        X_train_encoded_fe.columns
    ]
)


probabilities = baseline_rf.predict_proba(
    X_all_encoded_fe
)[:, 1]


candidate_ranking = pd.DataFrame(
    {
        "enrollee_id": df_clean[
            "enrollee_id"
        ].values,

        "Job_Change_Probability": (
            probabilities * 100
        ).round(2),
    }
)


candidate_ranking = (
    candidate_ranking
    .sort_values(
        by="Job_Change_Probability",
        ascending=False,
    )
    .reset_index(drop=True)
)


candidate_ranking.insert(
    0,
    "Rank",
    range(
        1,
        len(candidate_ranking) + 1,
    ),
)


top_10 = candidate_ranking.head(10)


# ============================================================
# INSIGHTS EXPORT
# ============================================================

insights = {
    "dataset": {
        "total_candidates": total_candidates,
        "looking_for_change": looking_for_change,
        "not_looking_for_change": not_looking_for_change,
        "job_change_rate": float(
            looking_for_change / total_candidates
        ),
    },

    "factor_rates": factor_rates,

    "education_trends": grouped_rates(
        "education_level"
    ),

    "experience_trends": grouped_rates(
        "experience"
    ),

    "company_size_trends": grouped_rates(
        "company_size"
    ),

    "company_type_trends": grouped_rates(
        "company_type"
    ),

    "last_new_job_trends": grouped_rates(
        "last_new_job"
    ),

    "top_10_candidates": top_10.to_dict(
        orient="records"
    ),
}


with open(
    os.path.join(EXPORT_DIR, "insights.json"),
    "w",
) as f:

    json.dump(
        insights,
        f,
        indent=2,
    )


# ============================================================
# PREPROCESSOR EXPORT
# ============================================================

preprocessor = {
    "encoder": encoder,
    "categorical_columns": categorical_columns,
    "numerical_columns": numerical_columns,
}

joblib.dump(
    preprocessor,
    os.path.join(
        EXPORT_DIR,
        "preprocessor_meta.pkl",
    ),
)


print("\n" + "=" * 70)
print("EXPORT COMPLETE")
print("=" * 70)

print("\nExported models:")
for filename in os.listdir(EXPORT_DIR):
    print(" -", filename)

print("\nPrimary model:")
print(primary_model_name)

print("\nBest model by accuracy:")
print(best_model_name)
