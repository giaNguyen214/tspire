"""
File: config.py
Chứa Lớp Config (dataclass) để quản lý tất cả các tham số.
"""
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class Config:
    # --- Cấu trúc thư mục ---
    
    # SỬA: BASE_DIR bây giờ là thư mục cha của file này (vd: /utils)
    # .../Techspire/utils -> .../Techspire
    BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

    # --- Đường dẫn ---
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True) 

    # Đường dẫn local mặc định
    data_path: str = os.path.join(data_dir, "SkilioMall_Churn Dataset_50,000 Users.xlsx")
    
    sheet_name: int = 0 
    target_column: str = "churn_label"

    # --- Model Output Paths ---
    # (Đường dẫn này bây giờ đã đúng, vd: .../Techspire/outputs/artifacts)
    artifacts_dir = os.path.join(BASE_DIR, "outputs", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    preprocessor_path: str = os.path.join(artifacts_dir, "preprocessor.pkl")
    xgb_model_path: str    = os.path.join(artifacts_dir, "xgb_model.pkl")
    lgbm_model_path: str   = os.path.join(artifacts_dir, "lgbm_model.pkl")
    cat_model_path: str    = os.path.join(artifacts_dir, "cat_model.pkl")
    ensemble_model_path: str = os.path.join(artifacts_dir, "ensemble_model.pkl")
    
    # --- Metric Output Paths ---
    metrics_dir = os.path.join(BASE_DIR, "outputs", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    metric_path: str = os.path.join(metrics_dir, "metric.json")
    shap_dir = os.path.join(metrics_dir, "shap")
    os.makedirs(shap_dir, exist_ok=True)
    eval_plot_dir = os.path.join(metrics_dir, "evaluation")
    os.makedirs(eval_plot_dir, exist_ok=True)

    # --- Tham số Training ---
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42

    smote_k_neighbors: int = 5
    smote_sampling_strategy: float = 1.0  # 1.0 = balance

    shap_top_n_features: int = 30 # because of encoding, there are about 40 cols
    optuna_n_trials: int = 10
    optuna_timeout: Optional[int] = None  # seconds