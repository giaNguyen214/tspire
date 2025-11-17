"""
File: config.py
Chứa Lớp Config (dataclass) để quản lý tất cả các tham số.
Tương thích với cả chạy local (nếu bạn tự gán data_path) và API.
"""
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class Config:
    # --- Cấu trúc thư mục (Giữ nguyên của bạn) ---
    
    # Lấy thư mục gốc của dự án (ví dụ: Techspire/)
    # Giả định file config.py nằm trong 1 thư mục con (vd: /utils/config.py)
    # Nếu config.py nằm ở gốc, dùng: os.path.abspath(os.path.dirname(__file__))
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Nếu file config.py nằm ở thư mục gốc (cùng với api.py), 
    # hãy dùng dòng này thay cho dòng trên:
    # BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    # --- Đường dẫn cho API (Sửa đổi) ---
    
    # data_path sẽ được pipeline_runner.py ghi đè khi chạy từ API
    data_path: Optional[str] = None 
    
    # Định nghĩa nơi lưu file upload tạm thời
    temp_upload_path: str = os.path.join(BASE_DIR, "temp_uploaded_data.bin")

    # Sửa: Dùng sheet 0 (sheet đầu tiên) để API chấp nhận mọi file Excel
    sheet_name: int = 0 
    target_column: str = "churn_label"

    # --- Model Output Paths (Giữ nguyên của bạn) ---
    artifacts_dir = os.path.join(BASE_DIR, "outputs", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    preprocessor_path: str = os.path.join(artifacts_dir, "preprocessor.pkl")
    xgb_model_path: str    = os.path.join(artifacts_dir, "xgb_model.pkl")
    lgbm_model_path: str   = os.path.join(artifacts_dir, "lgbm_model.pkl")
    cat_model_path: str    = os.path.join(artifacts_dir, "cat_model.pkl")
    ensemble_model_path: str = os.path.join(artifacts_dir, "ensemble_model.pkl")
    
    # --- Metric Output Paths (Giữ nguyên của bạn) ---
    metrics_dir = os.path.join(BASE_DIR, "outputs", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    metric_path: str = os.path.join(metrics_dir, "metric.json")

    # --- Tham số Training (Giữ nguyên của bạn) ---
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42

    smote_k_neighbors: int = 5
    smote_sampling_strategy: float = 1.0  # 1.0 = balance

    shap_top_n_features: int = 20
    optuna_n_trials: int = 10
    optuna_timeout: Optional[int] = None  # seconds