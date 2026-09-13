import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Career Transition Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global style adjustments */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0EA5E9;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 1.8rem;
        margin-bottom: 1rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #E2E8F0;
    }
    /* Model Cards */
    .model-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
        border: 1px solid #E2E8F0;
        margin-bottom: 10px;
    }
    .model-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .pred-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .pred-change {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }
    .pred-stay {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }
    .risk-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .risk-high {
        background-color: #EF4444;
        color: white;
    }
    .risk-moderate {
        background-color: #F59E0B;
        color: white;
    }
    .risk-low {
        background-color: #10B981;
        color: white;
    }
    .metric-val {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-lbl {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .insight-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #334155;
        margin-top: 12px;
        line-height: 1.4;
    }
    /* Summary Hero Banner */
    .summary-hero {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: white;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 25px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .summary-hero h2 {
        color: white !important;
        margin: 0 0 8px 0;
        font-size: 1.5rem;
    }
    .summary-hero p {
        color: #CBD5E1;
        margin: 0;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")
    
    knn_model = joblib.load(os.path.join(model_dir, "knn.pkl"))
    lr_model = joblib.load(os.path.join(model_dir, "logistic_regression.pkl"))
    rf_model = joblib.load(os.path.join(model_dir, "random_forest.pkl"))
    xgb_model = joblib.load(os.path.join(model_dir, "xgboost.pkl"))
    encoder = joblib.load(os.path.join(model_dir, "encoder.pkl"))
    meta = joblib.load(os.path.join(model_dir, "preprocessor_meta.pkl"))
    metrics_pack = joblib.load(os.path.join(model_dir, "notebook_metrics.pkl"))
    
    return knn_model, lr_model, rf_model, xgb_model, encoder, meta, metrics_pack

try:
    knn_model, lr_model, rf_model, xgb_model, encoder, meta, metrics_pack = load_artifacts()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ---------------------------------------------------------
# Preprocessing & Prediction Functions
# ---------------------------------------------------------
def sanitize_columns(cols):
    clean = []
    for c in cols:
        c = str(c).replace("<", "lt").replace(">", "gt")
        c = re.sub(r"[\[\]{}():,]", "_", c)
        clean.append(c)
    return clean

def transform_input(raw_data: dict, meta: dict, encoder):
    df = pd.DataFrame([raw_data])
    
    # Impute missing values exactly as training
    for col, m in meta["mode_values"].items():
        if col in df.columns and (pd.isna(df[col].iloc[0]) or df[col].iloc[0] is None):
            df[col] = m
            
    unknown_cols = ["gender", "major_discipline", "company_size", "company_type"]
    for col in unknown_cols:
        if col in df.columns and (pd.isna(df[col].iloc[0]) or df[col].iloc[0] is None):
            df[col] = "Unknown"

    cat_cols = meta["categorical_columns"]
    num_cols = meta["numerical_columns"]

    # Reusing fitted OneHotEncoder
    encoded_cat = encoder.transform(df[cat_cols])
    df_cat = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_cols))
    
    # 179-feature matrix (for KNN)
    df_179 = pd.concat([df_cat, df[num_cols]], axis=1)
    df_179.columns = df_179.columns.astype(str)
    
    # Feature Engineering
    exp_val = df["experience"].iloc[0]
    if exp_val == "<1" or exp_val == "Unknown":
        exp_num = 0.0
    elif exp_val == ">20":
        exp_num = 21.0
    else:
        try:
            exp_num = float(exp_val)
        except (ValueError, TypeError):
            exp_num = 0.0

    df_fe_num = df[num_cols].copy()
    training_hrs = float(df["training_hours"].iloc[0])
    df_fe_num["experience_to_training_ratio"] = exp_num / (training_hrs + 1.0)
    df_fe_num["has_relevant_degree"] = int(df["major_discipline"].iloc[0] == "STEM")

    # 181-feature matrix (for LR & RF)
    df_181 = pd.concat([df_cat, df_fe_num], axis=1)
    df_181.columns = df_181.columns.astype(str)

    # Sanitized 179-feature matrix (for XGBoost)
    df_xgb = df_179.copy()
    df_xgb.columns = sanitize_columns(df_xgb.columns)

    return df_179, df_181, df_xgb, exp_num, df_fe_num["experience_to_training_ratio"].iloc[0], df_fe_num["has_relevant_degree"].iloc[0]

def get_risk_meta(p_change: float):
    if p_change < 0.35:
        return "Low Risk", "risk-low", "#10B981"
    elif p_change <= 0.65:
        return "Moderate Risk", "risk-moderate", "#F59E0B"
    else:
        return "High Risk", "risk-high", "#EF4444"

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">💼 Career Transition Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered employee job change prediction using multiple machine learning models.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Static default input values
# ---------------------------------------------------------
default_city = "city_103"
default_cdi = 0.920
default_gender = "Male"
default_relevent_exp = "Has relevent experience"
default_enrolled_uni = "no_enrollment"
default_edu_level = "Graduate"
default_major = "STEM"
default_exp = "15"
default_company_size = "50-99"
default_company_type = "Pvt Ltd"
default_last_job = "1"
default_training_hrs = 45

# ---------------------------------------------------------
# Candidate Input Form
# ---------------------------------------------------------
st.markdown('<div class="section-title">📋 Candidate Information Form</div>', unsafe_allow_html=True)

with st.form("employee_input_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📍 Location & Demographics**")
        city_list = meta["categorical_options"]["city"]
        city_idx = city_list.index(default_city) if default_city in city_list else 0
        city = st.selectbox("City", options=city_list, index=city_idx, help="City of candidate residence")
        
        cdi = st.slider(
            "City Development Index (CDI)",
            min_value=0.400, max_value=1.000,
            value=float(default_cdi), step=0.001,
            help="Scaled development index of the city"
        )
        
        gender_list = meta["categorical_options"]["gender"]
        gender_idx = gender_list.index(default_gender) if default_gender in gender_list else 0
        gender = st.selectbox("Gender", options=gender_list, index=gender_idx)

        training_hours = st.slider(
            "Training Hours Completed",
            min_value=1, max_value=350,
            value=int(default_training_hrs), step=1,
            help="Total completed training hours"
        )

    with col2:
        st.markdown("**🎓 Education & Background**")
        edu_list = meta["categorical_options"]["education_level"]
        edu_idx = edu_list.index(default_edu_level) if default_edu_level in edu_list else 0
        education_level = st.selectbox("Education Level", options=edu_list, index=edu_idx)

        major_list = meta["categorical_options"]["major_discipline"]
        major_idx = major_list.index(default_major) if default_major in major_list else 0
        major_discipline = st.selectbox("Major Discipline", options=major_list, index=major_idx)

        uni_list = meta["categorical_options"]["enrolled_university"]
        uni_idx = uni_list.index(default_enrolled_uni) if default_enrolled_uni in uni_list else 0
        enrolled_university = st.selectbox("Enrolled University", options=uni_list, index=uni_idx)

        relevent_exp_list = meta["categorical_options"]["relevent_experience"]
        relevent_exp_idx = relevent_exp_list.index(default_relevent_exp) if default_relevent_exp in relevent_exp_list else 0
        relevent_experience = st.selectbox("Relevant Experience", options=relevent_exp_list, index=relevent_exp_idx)

    with col3:
        st.markdown("**💼 Professional Profile**")
        exp_list = meta["categorical_options"]["experience"]
        exp_idx = exp_list.index(default_exp) if default_exp in exp_list else 0
        experience = st.selectbox("Total Experience (Years)", options=exp_list, index=exp_idx)

        comp_size_list = meta["categorical_options"]["company_size"]
        comp_size_idx = comp_size_list.index(default_company_size) if default_company_size in comp_size_list else 0
        company_size = st.selectbox("Company Size (Employees)", options=comp_size_list, index=comp_size_idx)

        comp_type_list = meta["categorical_options"]["company_type"]
        comp_type_idx = comp_type_list.index(default_company_type) if default_company_type in comp_type_list else 0
        company_type = st.selectbox("Company Type", options=comp_type_list, index=comp_type_idx)

        last_job_list = meta["categorical_options"]["last_new_job"]
        last_job_idx = last_job_list.index(default_last_job) if default_last_job in last_job_list else 0
        last_new_job = st.selectbox("Years Since Last Job Change", options=last_job_list, index=last_job_idx)

    predict_btn = st.form_submit_button("🚀 Predict Career Transition", use_container_width=True, type="primary")

# ---------------------------------------------------------
# Transformation & Multi-Model Inference
# ---------------------------------------------------------
raw_input = {
    "city": city,
    "city_development_index": cdi,
    "gender": gender,
    "relevent_experience": relevent_experience,
    "enrolled_university": enrolled_university,
    "education_level": education_level,
    "major_discipline": major_discipline,
    "experience": experience,
    "company_size": company_size,
    "company_type": company_type,
    "last_new_job": last_new_job,
    "training_hours": training_hours,
}

df_179, df_181, df_xgb, exp_num, exp_ratio, has_stem = transform_input(raw_input, meta, encoder)

# 1. KNN (179 features, threshold 0.581)
knn_proba = knn_model.predict_proba(df_179)[0]
p_stay_knn = float(knn_proba[0])
p_change_knn = float(knn_proba[1])
knn_thresh = meta.get("knn_threshold", 0.581)
knn_pred = 1 if p_change_knn >= knn_thresh else 0

# 2. Logistic Regression + SMOTE (181 features)
lr_proba = lr_model.predict_proba(df_181)[0]
p_stay_lr = float(lr_proba[0])
p_change_lr = float(lr_proba[1])
lr_pred = 1 if p_change_lr >= 0.50 else 0

# 3. Random Forest (181 features)
rf_proba = rf_model.predict_proba(df_181)[0]
p_stay_rf = float(rf_proba[0])
p_change_rf = float(rf_proba[1])
rf_pred = 1 if p_change_rf >= 0.50 else 0

# 4. XGBoost (179 features with sanitized columns)
xgb_proba = xgb_model.predict_proba(df_xgb)[0]
p_stay_xgb = float(xgb_proba[0])
p_change_xgb = float(xgb_proba[1])
xgb_pred = 1 if p_change_xgb >= 0.50 else 0

models_data = [
    {
        "name": "KNN",
        "icon": "📍",
        "pred": knn_pred,
        "p_change": p_change_knn,
        "p_stay": p_stay_knn,
        "thresh": knn_thresh,
        "insight": f"Distance-weighted neighborhood prediction using tuned {knn_thresh:.3f} decision threshold. Relies on local proximity in 179-D space."
    },
    {
        "name": "Logistic Regression + SMOTE",
        "icon": "📈",
        "pred": lr_pred,
        "p_change": p_change_lr,
        "p_stay": p_stay_lr,
        "thresh": 0.50,
        "insight": f"Linear log-odds response incorporates engineered interaction terms (ratio: {exp_ratio:.3f}, STEM: {has_stem}) on balanced data."
    },
    {
        "name": "Random Forest",
        "icon": "🌲",
        "pred": rf_pred,
        "p_change": p_change_rf,
        "p_stay": p_stay_rf,
        "thresh": 0.50,
        "insight": f"400-tree balanced ensemble. Strongly weighted by City Development Index ({cdi:.3f}), company size, and experience ratio."
    },
    {
        "name": "XGBoost",
        "icon": "⚡",
        "pred": xgb_pred,
        "p_change": p_change_xgb,
        "p_stay": p_stay_xgb,
        "thresh": 0.50,
        "insight": f"Sequential gradient-boosted trees with SMOTE and early stopping. Highly sensitive to non-linear CDI thresholds."
    }
]

# ---------------------------------------------------------
# Overall Model Agreement & Consensus
# ---------------------------------------------------------
total_change_votes = sum(m["pred"] for m in models_data)
avg_p_change = float(np.mean([m["p_change"] for m in models_data]))
overall_risk_label, overall_risk_class, overall_risk_color = get_risk_meta(avg_p_change)

st.markdown('<div class="section-title">📊 Overall Model Agreement & Risk Synthesis</div>', unsafe_allow_html=True)

hero_html = f"""
<div class="summary-hero">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
        <div style="flex: 1; min-width: 300px;">
            <h2>Ensemble Consensus: {total_change_votes} of 4 Models Predict Job Change</h2>
            <p>
                {"A majority of models indicate this employee is actively open to or seeking a career transition." if total_change_votes >= 2 else "A majority of models predict this employee is likely to remain in their current position."}
            </p>
        </div>
        <div style="display: flex; gap: 24px; align-items: center; background: rgba(255,255,255,0.08); padding: 14px 28px; border-radius: 12px;">
            <div style="text-align: center;">
                <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Average Job Change Probability</div>
                <div style="font-size: 2rem; font-weight: 800; color: #38BDF8;">{avg_p_change * 100:.1f}%</div>
            </div>
            <div style="width: 1px; height: 44px; background-color: rgba(255,255,255,0.2);"></div>
            <div style="text-align: center;">
                <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Overall Risk Level</div>
                <div style="margin-top: 5px;"><span class="risk-pill {overall_risk_class}" style="font-size: 0.95rem; padding: 4px 14px;">{overall_risk_label}</span></div>
            </div>
        </div>
    </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# Individual Model Cards
# ---------------------------------------------------------
st.markdown('<div class="section-title">🎯 Individual Model Predictions</div>', unsafe_allow_html=True)

cols = st.columns(4)

for i, m in enumerate(models_data):
    with cols[i]:
        pred_text = "Looking for Job Change" if m["pred"] == 1 else "Likely to Stay"
        pred_badge_class = "pred-change" if m["pred"] == 1 else "pred-stay"
        risk_label, risk_class, risk_color = get_risk_meta(m["p_change"])
        
        st.markdown(f"""
        <div class="model-card">
            <div class="model-title">{m['icon']} {m['name']}</div>
            <div style="margin-bottom: 12px;">
                <span class="pred-badge {pred_badge_class}">{pred_text}</span>
                <span class="risk-pill {risk_class}">{risk_label}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <div>
                    <div class="metric-lbl">Job Change</div>
                    <div class="metric-val" style="color: #DC2626;">{m['p_change']*100:.1f}%</div>
                </div>
                <div style="text-align: right;">
                    <div class="metric-lbl">Stay</div>
                    <div class="metric-val" style="color: #16A34A;">{m['p_stay']*100:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(m['p_change'], text=f"Job Change Probability: {m['p_change']*100:.1f}%")
        
        st.markdown(f"""
        <div class="insight-box">
            <b>Insight:</b> {m['insight']}
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Cross-Model Probability Visualization (Altair)
# ---------------------------------------------------------
st.markdown('<div class="section-title">📊 Cross-Model Probability Comparison</div>', unsafe_allow_html=True)

col_chart1, col_chart2 = st.columns([3, 2])

with col_chart1:
    chart_df = pd.DataFrame([
        {
            "Model": m["name"],
            "Job Change (%)": round(m["p_change"] * 100, 1),
            "Stay (%)": round(m["p_stay"] * 100, 1),
            "Risk Level": get_risk_meta(m["p_change"])[0],
            "Color": get_risk_meta(m["p_change"])[2]
        }
        for m in models_data
    ])

    bar_chart = alt.Chart(chart_df).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        x=alt.X('Job Change (%):Q', scale=alt.Scale(domain=[0, 100]), title="Job Change Probability (%)"),
        y=alt.Y('Model:N', sort=None, title=None),
        color=alt.Color('Color:N', scale=None, legend=None),
        tooltip=['Model:N', 'Job Change (%):Q', 'Stay (%):Q', 'Risk Level:N']
    ).properties(height=240)

    text_labels = bar_chart.mark_text(
        align='left',
        baseline='middle',
        dx=5,
        fontWeight='bold'
    ).encode(
        text=alt.Text('Job Change (%):Q', format='.1f')
    )

    rule = alt.Chart(pd.DataFrame({'x': [50]})).mark_rule(
        color='#94A3B8',
        strokeDash=[5, 5],
        size=1.5
    ).encode(x='x:Q')

    st.altair_chart(bar_chart + text_labels + rule, use_container_width=True)

with col_chart2:
    st.markdown("#### Key Engineered Values for This Candidate")
    st.markdown(f"""
    - **Calculated Experience (Numeric):** `{exp_num:.1f} years`
    - **Experience-to-Training Ratio:** `{exp_ratio:.4f}`
      *(Formula: `experience_numeric / (training_hours + 1)`)*
    - **Has STEM Degree:** `{'Yes (1)' if has_stem == 1 else 'No (0)'}`
    - **City Development Index:** `{cdi:.3f}`
    """)
    st.info(
        "💡 **Data Observation:** In the notebook exploration, candidates in low-development cities (CDI < 0.75) "
        "and early-career candidates showed significantly higher propensities to change jobs."
    )

# ---------------------------------------------------------
# Explainability & Insights
# ---------------------------------------------------------
st.markdown('<div class="section-title">🔍 Explainability & Feature Insights</div>', unsafe_allow_html=True)

with st.expander("🌲 Random Forest Feature Importance Analysis", expanded=True):
    rf_feat_list = metrics_pack.get("rf_feature_importance", [])
    if rf_feat_list:
        df_rf_imp = pd.DataFrame(rf_feat_list).sort_values(by="Importance", ascending=False).head(10)
        df_rf_imp["Importance (%)"] = (df_rf_imp["Importance"] * 100).round(2)
        
        rf_chart = alt.Chart(df_rf_imp).mark_bar(color="#3B82F6", cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
            x=alt.X('Importance (%):Q', title="Relative Importance (%)"),
            y=alt.Y('Feature:N', sort='-x', title=None),
            tooltip=['Feature:N', 'Importance (%):Q']
        ).properties(height=280)
        
        rf_text = rf_chart.mark_text(
            align='left',
            baseline='middle',
            dx=4,
            fontSize=11,
            color='#1E293B'
        ).encode(
            text=alt.Text('Importance (%):Q', format='.2f')
        )
        
        st.altair_chart(rf_chart + rf_text, use_container_width=True)
    
    st.caption(
        "Feature importances extracted directly from the tuned 400-tree Random Forest classifier. "
        "`city_development_index` contributes over 17% of decision splits, followed by `city_city_21` and `experience_to_training_ratio`."
    )

with st.expander("📚 Model-Appropriate Behavioral Insights"):
    st.markdown("""
    #### Why do the models produce different predictions?
    1. **KNN (K-Nearest Neighbors):**
       - Evaluates Euclidean distance in the scaled 179-dimensional feature space among its 33 nearest neighbors.
       - If a candidate shares characteristics with a dense pocket of historical job switchers, KNN flags high probability even if global features are moderate.
       - Employs an optimized decision threshold of **0.581** calibrated on cross-validation F1.
       
    2. **Logistic Regression (+ SMOTE):**
       - Fits a global linear boundary in log-odds space with balanced sample weights.
       - Directly weights monotonic increases in `city_development_index`, `training_hours`, and the interaction term `experience_to_training_ratio`.
       
    3. **Random Forest:**
       - Constructs 400 deep decision trees using bootstrap aggregation and balanced subsampling.
       - Captures complex non-linear combinations between education level, company size, and years of experience.
       
    4. **XGBoost:**
       - Sequentially builds shallow trees (depth 5) optimizing the logistic loss gradient.
       - Early stopping at iteration 18 prevents overfitting to sparse one-hot city categories.
    """)

# ---------------------------------------------------------
# Model Benchmark Comparison (Cell 112 Ground Truth)
# ---------------------------------------------------------
st.markdown('<div class="section-title">🏆 Model Benchmark Comparison (Ground Truth from Notebook)</div>', unsafe_allow_html=True)

st.markdown("""
Below are the actual test-set performance metrics extracted directly from **Step 19 (Cell 112)** of the notebook:
""")

benchmarks = metrics_pack.get("benchmarks", {})
bench_df = pd.DataFrame(benchmarks).T[["Accuracy", "Precision", "Recall", "F1-score", "Threshold", "Balancing"]]

styled_bench = bench_df.copy()
for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
    styled_bench[col] = styled_bench[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x)

st.dataframe(styled_bench, use_container_width=True)

# Comparison Bar Chart with Altair
bench_melted = bench_df.reset_index().rename(columns={"index": "Model"}).melt(
    id_vars=["Model"],
    value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
    var_name="Metric",
    value_name="Score"
)
bench_melted["Score"] = bench_melted["Score"].astype(float).round(4)

bench_chart = alt.Chart(bench_melted).mark_bar().encode(
    x=alt.X('Metric:N', title=None),
    y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1.0]), title="Score"),
    color=alt.Color('Metric:N', legend=None),
    column=alt.Column('Model:N', title=None),
    tooltip=['Model:N', 'Metric:N', 'Score:Q']
).properties(width=160, height=220)

st.altair_chart(bench_chart)

st.markdown("""
---
<div style="text-align: center; color: #94A3B8; font-size: 0.85rem; padding: 15px 0;">
    Career Transition Predictor | HR Analytics Machine Learning Project | Powered by Streamlit & Scikit-Learn
</div>
""", unsafe_allow_html=True)
