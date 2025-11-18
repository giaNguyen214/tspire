# Project README
## Overview

This project includes:
- A Streamlit web demo for running inference.
- A FastAPI service for API-based inference and SHAP explanations.
- Training and inference scripts.
- Output folders for SHAP images, model artifacts, and evaluation metrics.


## 1. Streamlit Web Demo
- Run locally: streamlit run streamlit_app.py
- Online demo: You can try the deployed version here https://techspire-gia-nguyen.streamlit.app/

## 2. FastAPI Service
- Run locally: uvicorn api:app --reload
- After running, open the API documentation at: http://127.0.0.1:8000/docs

- Main API Endpoints
  - 1. /predict-single/
    - Make a single prediction.
    - Parameter shap: choose whether to generate SHAP results or not.
    - Sample Request Body:
      {
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

  - 2. /download/{filename}
    - Download SHAP results as a ZIP file.
    - Example: /download/shap_outputs.zip

## 3. Training the Model
- Run training: python train.py
- This script trains the model and saves all outputs into outputs/artifacts.

## 4. Local Inference Script
- Run inference: python inference.py
- This uses the sample data file and prints the prediction result. SHAP results (if enabled) will be saved inside outputs/shap_inference.

## 5. Project Structure
project/
│
├── api.py                 # FastAPI application
├── train.py               # Train the model
├── inference.py           # Local inference script
├── streamlit_app.py       # Web demo
│
├── data/                  # Dataset files
│
├── outputs/
│   ├── shap_api/          # SHAP images from API calls
│   ├── shap_inference/    # SHAP images from inference.py
│   ├── artifacts/         # Trained model files
│   ├── eda/               # EDA images/reports
│   └── metrics/           # Model evaluation metrics
│
└── README.md
