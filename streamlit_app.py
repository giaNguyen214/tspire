import streamlit as st
import pandas as pd
import zipfile
import io
import os
from PIL import Image

# ===== IMPORT YOUR INFERENCE FUNCTIONS =====
from inference import (
    load_all_models,
    generate_zip_results,
    predict_single
)

# ===== LOAD MODEL AT START =====
load_all_models()

st.title("Techspire – Churn Prediction Dashboard")
st.markdown("""
This dashboard presents:
1. A summary of the dataset and training process  
2. Evaluation metrics produced during training  
3. Model explainability using SHAP  
4. An interactive tool to run predictions on custom input  

This interface is designed for non-technical users, with simple explanations and visual outputs.
""")

# ==========================
# SECTION: TABLE OF CONTENTS
# ==========================

st.markdown("## Table of Contents")
st.markdown("""
- [1. Training Results](#training-results)
- [2. Model SHAP Explainability](#model-shap-explainability)
- [3. Live Model Inference](#live-model-inference)
""")

# ==========================
# SECTION 1 – TRAINING RESULTS
# ==========================

st.markdown("## Training Results")
st.write("Below are the visual outputs generated during the model training phase. These charts help explain the data distribution, model performance, and evaluation metrics.")

# Load helper function
def show_image_row(title, image_paths, captions=None):
    st.markdown(f"### {title}")

    if captions is None:
        captions = [""] * len(image_paths)

    cols = st.columns(len(image_paths))

    for idx, img_path in enumerate(image_paths):
        with cols[idx]:
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
                if captions[idx]:
                    st.caption(captions[idx])
            else:
                st.write(f"(Placeholder for {os.path.basename(img_path)})")



# --- EDA images ---
show_image_row(
    "Exploratory Data Analysis (EDA)",
    [
        "outputs/eda/target_distribution.png",
        "outputs/eda/correlation_matrix.png",
        "outputs/eda/outlier_box_plots.png"
    ],
    captions=[
        "The dataset is imbalanced: around 75% active users and 25% churned.",
        "Features have normal and simple relationships. No feature pair is too strongly linked, so the model learns safely.",
        "Many features have natural outliers. This is normal in e-commerce because a few users are very active or spend a lot."
    ]
)


# --- Metrics images ---
# --- Metrics main ---
show_image_row(
    "Model Evaluation Metrics",
    [
        "outputs/metrics/evaluation/calibration_plot.png",
        "outputs/metrics/evaluation/confusion_matrix.png",
        "outputs/metrics/evaluation/roc_curve.png"
    ],
    captions=[
        "Predicted churn probabilities match real outcomes well. This means the scores are reliable for business use.",
        "Most users are classified correctly. Wrong predictions are very few, showing strong model accuracy.",
        "The ROC curve is close to perfect (AUC ~0.99). The model clearly separates churn vs. non-churn users."
    ]
)


# --- Sub-metrics ---
show_image_row(
    "Additional Performance Curves",
    [
        "outputs/metrics/evaluation/pr_curve.png",
        "outputs/metrics/evaluation/gain_chart.png",
        "outputs/metrics/evaluation/lift_curve.png"
    ],
    captions=[
        """
Precision stays very high across most recall levels, meaning when the model says a user will churn, it is almost always correct.
It only drops at very high recall, which is normal when including more uncertain users.
        """,
        """
The gain chart shows the model finds most churners within the top ~30% highest-risk users.
This means the model is very good at ranking users from high-risk to low-risk.
        """,
        """
The lift curve stays far above the baseline. The top user segment captures almost all churners.
This shows strong prioritization ability for targeting retention actions.
        """
    ]
)




# ==========================
# SECTION 2 – SHAP EXPLAINABILITY
# ==========================

st.markdown("## Model SHAP Explainability")
st.write("""
SHAP helps explain which features matter most for predictions. Below are the SHAP results for each model.
""")

# XGBoost
show_image_row("XGBoost SHAP", [
    "outputs/metrics/shap/xgb_summary.png",
    "outputs/metrics/shap/xgb_bar.png",
    "outputs/metrics/shap/xgb_waterfall_0.png"
],
    captions=[
        None,
        """
Top features include days_since_last_order and emails_open_rate_90d. These strongly affect churn risk.
Activity, support tickets, and refund behavior also play key roles.
        """,
        """
The waterfall shows this user is very low churn. High engagement pushes the score down, and risky features are small.
        """
    ]               
)

# LightGBM
show_image_row("LightGBM SHAP", [
    "outputs/metrics/shap/lgbm_summary.png",
    "outputs/metrics/shap/lgbm_bar.png",
    "outputs/metrics/shap/lgbm_waterfall_0.png"
],
    captions=[
        None,
        """
Important features include inactivity, email engagement, and support tickets.
Active and satisfied users naturally show lower churn in the model.
        """,
        """
The plot shows this user has strong engagement, so churn score goes down.
Only small positive factors slightly increase risk.
        """
    ] )

# CatBoost
show_image_row("CatBoost SHAP", [
    "outputs/metrics/shap/cat_summary.png",
    "outputs/metrics/shap/cat_bar.png",
    "outputs/metrics/shap/cat_waterfall_0.png"
],
    captions=[
        None,
        """
CatBoost focuses strongly on recency, email activity, and customer issues.
Refund rate and sessions also affect churn meaningfully.
        """,
        """
This user looks low-risk: good engagement pushes the score down.
Negative behaviors are small and do not increase churn much.
        """
    ])

st.markdown("""
XGBoost SHAP Summary:
XGBoost shows simple, clear patterns. Long inactivity increases churn, while strong engagement lowers it. Refunds and support tickets raise risk, while active sessions and orders reduce it.

LightGBM SHAP Summary:
LightGBM behaves similarly, with inactivity and email engagement as top factors. Refund behavior and support issues increase churn, while active users stay safe.

CatBoost SHAP Summary:
CatBoost puts more weight on recency and refund behavior. Email engagement and activity reduce churn. It also picks up extra signals like app version patterns.
""")


# ==========================
# SECTION 3 – LIVE MODEL INFERENCE
# ==========================

st.markdown("## Live Model Inference")
st.write("""
Enter customer information below.  
The model will compute churn probability and provide SHAP explanations for the inputs.
""")

# ===== DEFAULT SAMPLE =====
sample_0 = {
    "age": 20,
    "country": "Thailand",
    "city": "Bangkok",
    "reg_days": 262,
    "marketing_source": "ads_fb",
    "sessions_30d": 2,
    "sessions_90d": 4,
    "avg_session_duration_90d": 728.93,
    "median_pages_viewed_30d": 4.41,
    "search_queries_30d": 1,
    "device_mix_ratio": 0.861,
    "app_version_major": "3.x",
    "orders_30d": 0,
    "orders_90d": 1,
    "orders_2024": 4,
    "aov_2024": 18.95,
    "gmv_2024": 80.58,
    "category_diversity_2024": 0,
    "days_since_last_order": 55,
    "discount_rate_2024": 0.168,
    "refunds_count_2024": 0,
    "refund_rate_2024": 0,
    "support_tickets_2024": 1,
    "avg_csat_2024": 4.3,
    "emails_open_rate_90d": 0.252,
    "emails_click_rate_90d": 0.029,
    "review_count_2024": 0,
    "avg_review_stars_2024": 4.46,
    "rfm_recency": 55,
    "rfm_frequency": 4,
    "rfm_monetary": 80.58
}

COUNTRIES = ['Thailand', 'Indonesia', 'Malaysia', 'Vietnam', 'Philippines']
CITIES = [
    'Bangkok', 'Jakarta', 'Surabaya', 'Johor Bahru', 'Ho Chi Minh City',
    'Hanoi', 'Bandung', 'Manila', 'Cebu', 'Kuala Lumpur',
    'Davao', 'Chiang Mai', 'Da Nang', 'Phuket', 'Penang'
]
SOURCES = ['ads_fb', 'organic', 'referral', 'influencer', 'ads_ig']
VERSIONS = ['3.x', '2.x', '1.x']

with st.form("user_input_form"):
    st.subheader("Customer Features")

    user_data = {}

    for key, val in sample_0.items():

        if key == "country":
            user_data[key] = st.selectbox("country", COUNTRIES, index=0)

        elif key == "city":
            user_data[key] = st.selectbox("city", CITIES, index=0)

        elif key == "marketing_source":
            user_data[key] = st.selectbox("marketing_source", SOURCES, index=0)

        elif key == "app_version_major":
            user_data[key] = st.selectbox("app_version_major", VERSIONS, index=0)

        elif isinstance(val, (int, float)):
            user_data[key] = st.number_input(key, value=val)

    submit = st.form_submit_button("Predict")

# ===== RUN PREDICTION =====
if submit:
    st.info("Running the model and generating SHAP explanations. Please wait...")

    df = pd.DataFrame([user_data])

    zip_path, pred_df = generate_zip_results(df_raw=df, explain=True, max_rows=1)

    prediction = pred_df.iloc[0].to_dict()

    st.write("### Prediction Output")
    st.json(prediction)

    st.write("### SHAP Visualizations")
    with zipfile.ZipFile(zip_path, "r") as zf:
        img_files = [f for f in zf.namelist() if f.endswith(".png")]

        models = {
            "XGBoost": [],
            "LightGBM": [],
            "CatBoost": []
        }

        for img_name in img_files:
            lower = img_name.lower()
            if "xgb" in lower:
                models["XGBoost"].append(img_name)
            elif "lgb" in lower:
                models["LightGBM"].append(img_name)
            elif "cat" in lower:
                models["CatBoost"].append(img_name)

        for model_name, images in models.items():
            if not images:
                continue
            st.markdown(f"#### {model_name}")
            cols = st.columns(3)
            for idx, img_name in enumerate(images):
                img = Image.open(io.BytesIO(zf.read(img_name)))
                with cols[idx % 3]:
                    st.image(img, use_container_width=True)
                    st.caption(img_name)

    st.download_button(
        "Download SHAP ZIP",
        data=open(zip_path, "rb").read(),
        file_name="shap_outputs.zip"
    )
