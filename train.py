"""
File: pipeline_runner.py
Tách logic từ main.py cũ.
Chạy toàn bộ pipeline huấn luyện.
"""
import logging
import json
from typing import List
from joblib import Parallel, delayed
import os

from utils.config import Config
from utils.utils import set_global_seed, setup_logging
from utils.dataloader import ChurnDataLoader
from utils.preprocessor import ChurnPreprocessor
from utils.trainer import ChurnModelTrainer, SmoteBalancer
from utils.evaluator import ChurnEvaluator
import matplotlib
matplotlib.use("Agg")
from utils.eda import run_eda


# Đổi tên main() thành run_training_pipeline()
# Thêm tham số: data_file_path
def run_training_pipeline(data_file_path: str) -> List[str]:
    """
    Chạy toàn bộ pipeline huấn luyện trên file data được cung cấp.
    Lưu metrics vào JSON và trả về danh sách các file output.
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Churn Prediction Pipeline ---")
    
    config = Config()
    # Ghi đè data_path bằng đường dẫn file upload
    config.data_path = data_file_path
    
    set_global_seed(config.random_state)

    # Dùng để thu thập metrics
    master_metrics = {}

    try:
        # 1. Load data
        logger.info("\n--- STEP 1: Load Data ---")
        loader = ChurnDataLoader(config)
        df = loader.load_raw_data()
        numerical_cols, categorical_cols = loader.get_column_types(df)
        
        # NEW — Gọi EDA
        eda_results = run_eda(df, numerical_cols)
        logger.info(f"EDA files generated: {eda_results}")


        # 2. Split data
        logger.info("\n--- STEP 2: Split Data ---")
        X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(df)

        # 3. Preprocess
        logger.info("\n--- STEP 3: Preprocess Data ---")
        preproc = ChurnPreprocessor(numerical_cols, categorical_cols, label_encoded_cols=["city"])
        X_train_proc = preproc.fit_transform(X_train)
        X_val_proc = preproc.transform(X_val)
        X_test_proc = preproc.transform(X_test)
        preproc.save(config.preprocessor_path)
        
        feature_names = preproc.get_feature_names()
        X_train_proc.columns = feature_names
        X_val_proc.columns = feature_names
        X_test_proc.columns = feature_names

        # 4. SMOTE
        logger.info("\n--- STEP 4: Apply SMOTE ---")
        smote = SmoteBalancer(config)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

        # 5. Train Models & Feature Selection
        trainer = ChurnModelTrainer(config)

        # 5.1 Train base models (Parallel)
        logger.info("\n--- STEP 5.1: Train Base Models for SHAP FS (Parallel) ---")
        with Parallel(n_jobs=3, backend="threading") as parallel:
            base_models = parallel(
                delayed(func)(X_train_bal, y_train_bal, X_val_proc, y_val)
                for func in [
                    trainer.train_xgb_base,
                    trainer.train_lightgbm_base,
                    trainer.train_catboost_base
                ]
            )
        trainer.xgb_model = base_models[0]
        trainer.lgbm_model = base_models[1]
        trainer.cat_model = base_models[2]
        logger.info("Base models trained in parallel.")

        # 5.2 SHAP Feature Selection (Parallel)
        logger.info("\n--- STEP 5.2: Perform SHAP Feature Selection (Parallel) ---")
        models_to_shap = [
            (trainer.xgb_model, "xgb"),
            (trainer.lgbm_model, "lgbm"),
            (trainer.cat_model, "cat")
        ]
        with Parallel(n_jobs=3, backend="threading") as parallel:
            shap_results = parallel(
                delayed(ChurnModelTrainer._get_shap_values)(model, X_val_proc, m_type)
                for model, m_type in models_to_shap
            )
        top_features = trainer.process_shap_importances(shap_results)
        
        # 5.3 Filter features
        logger.info("\n--- STEP 5.3: Filtering datasets by top features ---")
        X_train_bal_fs = X_train_bal[top_features]
        X_val_proc_fs = X_val_proc[top_features]
        X_test_proc_fs = X_test_proc[top_features]

        # 5.4 Tune models (Optuna) (Parallel)
        logger.info("\n--- STEP 5.4: Tune Final Models with Optuna (Parallel) ---")
        with Parallel(n_jobs=3, backend="threading") as parallel:
            tuned_models = parallel(
                delayed(func)(X_train_bal_fs, y_train_bal, X_val_proc_fs, y_val)
                for func in [
                    trainer.tune_xgb_optuna,
                    trainer.tune_lgbm_optuna,
                    trainer.tune_cat_optuna
                ]
            )
        trainer.xgb_model = tuned_models[0]
        trainer.lgbm_model = tuned_models[1]
        trainer.cat_model = tuned_models[2]
        logger.info("Optuna tuning finished in parallel.")

        # 5.5 Build ensemble
        logger.info("\n--- STEP 5.5: Build Ensemble Model ---")
        trainer.build_ensemble()

        # 5.6 Calculate final SHAP
        logger.info("\n--- STEP 5.6: Calculate Final SHAP Values ---")
        trainer.calculate_final_shap_values(X_test_proc)

       # 6. Evaluate
        logger.info("\n--- STEP 6: Evaluate Models on Test Set ---")

        # XGB
        logger.info("\n=== Tuned XGB Test Metrics ===")
        eval_xgb = ChurnEvaluator(y_test, trainer.xgb_model.predict_proba(X_test_proc_fs)[:, 1])
        master_metrics["xgb"] = eval_xgb.evaluate_all()

        # LGBM
        logger.info("\n=== Tuned LGBM Test Metrics ===")
        eval_lgbm = ChurnEvaluator(y_test, trainer.lgbm_model.predict_proba(X_test_proc_fs)[:, 1])
        master_metrics["lgbm"] = eval_lgbm.evaluate_all()

        # CatBoost
        logger.info("\n=== Tuned CatBoost Test Metrics ===")
        eval_cat = ChurnEvaluator(y_test, trainer.cat_model.predict_proba(X_test_proc_fs)[:, 1])
        master_metrics["catboost"] = eval_cat.evaluate_all()

        # Ensemble
        logger.info("\n=== Ensemble Test Metrics ===")
        eval_ens = ChurnEvaluator(y_test, trainer.predict_proba_ensemble(X_test_proc))
        master_metrics["ensemble"] = eval_ens.evaluate_all()


        # 7. Save Models
        logger.info("\n--- STEP 7: Save All Models ---")
        trainer.save_xgb(config.xgb_model_path)
        trainer.save_lgbm(config.lgbm_model_path)
        trainer.save_cat(config.cat_model_path)
        trainer.save_ensemble(config.ensemble_model_path)
        
        # 7.1 (MỚI) Save Metrics
        logger.info(f"\n--- STEP 7.1: Save Metrics to JSON ---")
        try:
            with open(config.metric_path, 'w') as f:
                json.dump(master_metrics, f, indent=4)
            logger.info(f"Metrics saved to {config.metric_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

        logger.info("\n--- Churn Prediction Pipeline Finished Successfully ---")
        
        # (MỚI) Trả về danh sách các file đã tạo
        return [
            config.preprocessor_path,
            config.xgb_model_path,
            config.lgbm_model_path,
            config.cat_model_path,
            config.ensemble_model_path,
            config.metric_path
        ]

    except Exception as e:
        logger.error(f"--- Pipeline Failed: {e} ---", exc_info=True)
        # Trả về list rỗng nếu lỗi
        return []

# Hàm này để thiết lập logging khi chạy từ API
def initialize_logging():
    setup_logging()
    
    
if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("--- Running pipeline in LOCAL mode ---")
    
    config = Config()
    
    # Kiểm tra xem file data local trong config có tồn tại không
    if config.data_path and os.path.exists(config.data_path):
        logger.info(f"Using local data file: {config.data_path}")
        # Chạy pipeline với file local
        run_training_pipeline(config.data_path)
    else:
        # Báo lỗi nếu không tìm thấy file
        logger.error("="*50)
        logger.error("LOCAL RUN FAILED: `data_path` not found in config.py")
        logger.error(f"File not found: {config.data_path}")
        logger.error("Please edit 'config.py' to point to your local data file.")
        logger.error("="*50)