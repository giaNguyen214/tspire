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

st.title("🛍️ Churn Prediction – Techspire")
st.write("Nhập thông tin khách hàng → Model trả về dự đoán + SHAP interpretability")

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

# ===== STREAMLIT FORM =====
with st.form("user_input_form"):
    st.subheader("📄 Customer Features")

    user_data = {}
    for key, val in sample_0.items():
        if isinstance(val, (int, float)):
            user_data[key] = st.number_input(key, value=val)
        else:
            user_data[key] = st.text_input(key, value=val)

    submit = st.form_submit_button("🚀 Predict")

# ===== RUN PREDICTION =====
if submit:
    st.info("Đang chạy model & SHAP... vui lòng chờ...")

    df = pd.DataFrame([user_data])

    # API logic tái hiện 100%
    zip_path, pred_df = generate_zip_results(df_raw=df, explain=True, max_rows=1)

    prediction = pred_df.iloc[0].to_dict()

    st.success("🎉 Prediction ready!")
    st.write("### 🔮 Prediction Output")
    st.json(prediction)

    # ===== SHOW SHAP IMAGES =====
    st.write("### 📊 SHAP Explainability Plots")
    st.write("(Lấy từ ZIP y như API)")

    with zipfile.ZipFile(zip_path, "r") as zf:
        img_files = [f for f in zf.namelist() if f.endswith(".png")]

        # Gom ảnh theo prefix model
        models = {
            "XGBoost": [],
            "LightGBM": [],
            "CatBoost": []
        }

        for img_name in img_files:
            lower = img_name.lower()
            if "xgb" in lower:
                models["XGBoost"].append(img_name)
            elif "lgb" in lower or "light" in lower:
                models["LightGBM"].append(img_name)
            elif "cat" in lower:
                models["CatBoost"].append(img_name)

        # Hiển thị từng model
        for model_name, image_list in models.items():
            if len(image_list) == 0:
                continue

            st.markdown(f"## 🔥 {model_name}")

            # 3 ảnh một dòng
            cols = st.columns(3)

            for idx, img_name in enumerate(image_list):
                img = Image.open(io.BytesIO(zf.read(img_name)))
                with cols[idx % 3]:
                    st.image(img, use_container_width=True)
                    st.caption(img_name)

    # ===== DOWNLOAD ZIP =====
    st.download_button(
        "📥 Download SHAP ZIP",
        data=open(zip_path, "rb").read(),
        file_name="shap_outputs.zip"
    )
