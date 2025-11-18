"""
Inference script for Churn Model (XGB + LGBM + CatBoost + Ensemble + SHAP)
"""

import os
import pickle
import logging
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.preprocessor import ChurnPreprocessor
from utils.config import Config

# ============================================================
# CONFIG
# ============================================================

ENABLE_SHAP = True
SHAP_OUTPUT_DIR = "outputs/shap_inference"

# Auto create output folder
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD MODELS & PREPROCESSOR
# ============================================================

config = Config()
IN_API_ZIP_MODE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Inference")

preprocessor = None
xgb_model = None
lgbm_model = None
cat_model = None
top_features = None

def load_all_models():
    """
    Called by FastAPI at startup.
    Loads:
        - Preprocessor
        - XGB
        - LGBM
        - CatBoost
        - Ensemble (top_features)
    """

    global preprocessor, xgb_model, lgbm_model, cat_model, top_features

    logger = logging.getLogger("Inference")

    logger.info("Loading preprocessor...")
    with open(config.preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    logger.info("Loading XGB model...")
    with open(config.xgb_model_path, "rb") as f:
        xgb_model = pickle.load(f)

    logger.info("Loading LGBM model...")
    with open(config.lgbm_model_path, "rb") as f:
        lgbm_model = pickle.load(f)

    logger.info("Loading CatBoost model...")
    with open(config.cat_model_path, "rb") as f:
        cat_model = pickle.load(f)

    logger.info("Loading Ensemble...")
    with open(config.ensemble_model_path, "rb") as f:
        ens = pickle.load(f)
        top_features = ens["top_features"]

    logger.info("All inference models successfully loaded.")


# ============================================================
# INTERNAL PREDICTION
# ============================================================

def _predict_internal(X_fs: pd.DataFrame):
    """Predict with all models + ensemble"""
    p_xgb = xgb_model.predict_proba(X_fs)[:, 1]
    p_lgb = lgbm_model.predict_proba(X_fs)[:, 1]
    p_cat = cat_model.predict_proba(X_fs)[:, 1]

    p_ens = (p_xgb + p_lgb + p_cat) / 3.0
    return p_ens, p_xgb, p_lgb, p_cat


# ============================================================
# SHAP HELPERS
# ============================================================

def compute_shap(model_name: str, model, X_fs: pd.DataFrame):
    logger.info(f"[SHAP] Computing SHAP for {model_name}...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X_fs)
    return shap_vals


def save_shap_plots(model_name: str, shap_values, X_fs: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"[SHAP] Saving plots for {model_name} → {out_dir}")

    # Summary
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values.values, X_fs, show=False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/summary.png")
    plt.close()

    # Bar
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values.values, X_fs, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/bar.png")
    plt.close()

    # Waterfall only for first sample
    if len(X_fs) == 1:
        plt.figure(figsize=(8, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/waterfall.png")
        plt.close()


# ============================================================
# PUBLIC API – PREDICT BATCH
# ============================================================

def predict_batch(
    df_raw: pd.DataFrame,
    explain: bool = ENABLE_SHAP,
    max_shap_rows: int = 100
):
    """Predict many samples at once, DataFrame input"""

    df_raw = df_raw.copy()

    # Remove user_id
    if "user_id" in df_raw.columns:
        df_raw = df_raw.drop(columns=["user_id"])

    # Preprocess
    X_proc = preprocessor.transform(df_raw)
    X_fs = X_proc[top_features]

    # Predictions
    p_ens, p_xgb, p_lgb, p_cat = _predict_internal(X_fs)

    results = pd.DataFrame({
        "prob_ensemble": p_ens,
        "prob_xgb": p_xgb,
        "prob_lgbm": p_lgb,
        "prob_cat": p_cat,
        "label_pred": (p_ens >= 0.5).astype(int)
    })

    # ======================================================
    # SHAP optional
    # ======================================================
    if explain:
        n = min(len(X_fs), max_shap_rows)
        logger.info(f"[SHAP] Running SHAP on {n} rows...")

        X_shap = X_fs.iloc[:n]

        # Decide SHAP dir
        shap_dir = Config.shap_api_dir if IN_API_ZIP_MODE else SHAP_OUTPUT_DIR

        # Save to model subfolders
        shap_xgb = compute_shap("xgb", xgb_model, X_shap)
        save_shap_plots("xgb", shap_xgb, X_shap, f"{shap_dir}/xgb")

        shap_lgb = compute_shap("lgbm", lgbm_model, X_shap)
        save_shap_plots("lgbm", shap_lgb, X_shap, f"{shap_dir}/lgbm")

        shap_cat = compute_shap("cat", cat_model, X_shap)
        save_shap_plots("cat", shap_cat, X_shap, f"{shap_dir}/cat")

    return results


# ============================================================
# PUBLIC API – PREDICT SINGLE
# ============================================================

def predict_single(sample_dict: dict, explain: bool = ENABLE_SHAP):
    """Predict 1 record from python dict"""
    df = pd.DataFrame([sample_dict])
    result_df = predict_batch(df, explain=explain, max_shap_rows=1)
    return result_df.iloc[0].to_dict()


def generate_zip_results(
    df_raw: pd.DataFrame,
    explain: bool = True,
    max_rows: int = 100
):
    """
    API helper:
    - Gọi predict_batch() => tạo 9 ảnh SHAP
    - Gom ảnh + kết quả CSV vào 1 file ZIP
    - Trả về (path_zip, prediction_dataframe)
    """

    # 1) Run prediction (SHAP inside)
    global IN_API_ZIP_MODE
    # Turn on API ZIP mode
    IN_API_ZIP_MODE = True

    pred_df = predict_batch(df_raw, explain=explain, max_shap_rows=max_rows)

    # Turn off after done
    IN_API_ZIP_MODE = False

    # 2) Collect SHAP image paths WITH MODEL PREFIX
    image_paths = []
    for model in ["xgb", "lgbm", "cat"]:
        folder = f"{Config.shap_api_dir}/{model}"
        for name in ["summary.png", "bar.png", "waterfall.png"]:
            old_path = f"{folder}/{name}"
            if os.path.exists(old_path):
                # e.g. xgb_summary.png
                new_name = f"{model}_{name}"
                image_paths.append((old_path, new_name))

    # 3) Prepare ZIP path
    zip_path = f"{Config.shap_api_dir}/shap_outputs.zip"

    # Remove old file
    if os.path.exists(zip_path):
        os.remove(zip_path)

    # 4) Write ZIP
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:

        # Add prediction CSV
        pred_csv_path = f"{Config.shap_api_dir}/pred_results.csv"
        pred_df.to_csv(pred_csv_path, index=False)
        zipf.write(pred_csv_path, arcname="pred_results.csv")

        # Add SHAP images (renamed)
        for old_path, new_name in image_paths:
            zipf.write(old_path, arcname=new_name)

    return zip_path, pred_df


# ============================================================
# DEMO WHEN RUN DIRECT
# ============================================================

if __name__ == "__main__":
    print("===== Loading models =====")
    load_all_models()
    
    # -------------------------------------
    # Single
    # -------------------------------------
    print("===== Single prediction =====")
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
    
    sample_1 = {
        "age": 31,
        "country": "Indonesia",
        "city": "Surabaya",
        "reg_days": 406,
        "marketing_source": "referral",
        "sessions_30d": 0,
        "sessions_90d": 3,
        "avg_session_duration_90d": 493.29,
        "median_pages_viewed_30d": 2.58,
        "search_queries_30d": 1,
        "device_mix_ratio": 0.917,
        "app_version_major": "2.x",
        "orders_30d": 0,
        "orders_90d": 0,
        "orders_2024": 1,
        "aov_2024": 14.41,
        "gmv_2024": 11.95,
        "category_diversity_2024": 1,
        "days_since_last_order": 73,
        "discount_rate_2024": 0.488,
        "refunds_count_2024": 0,
        "refund_rate_2024": 0,
        "support_tickets_2024": 0,
        "avg_csat_2024": 4.35,
        "emails_open_rate_90d": 0.343,
        "emails_click_rate_90d": 0.014,
        "review_count_2024": 0,
        "avg_review_stars_2024": 4.59,
        "rfm_recency": 73,
        "rfm_frequency": 1,
        "rfm_monetary": 11.95
    }


    print(predict_single(sample_0))

    # -------------------------------------
    # BATCH DEMO (3 samples)
    # -------------------------------------
    print("\n===== Batch prediction (3 samples) =====")
    df_batch = pd.DataFrame([sample_0, sample_1, sample_0])
    print(predict_batch(df_batch, explain=True, max_shap_rows=3))
