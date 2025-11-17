"""
File: trainer.py
Chứa Lớp SmoteBalancer và ChurnModelTrainer.
Quản lý SMOTE, training (base, optuna), SHAP (FS, final), và lưu model.
"""
import logging
import pickle
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier

import optuna
# from optuna.integration import TqdmCallback # <--- XÓA
import shap
from joblib import Parallel, delayed
# from tqdm.auto import tqdm # <--- XÓA

from config import Config

# ============================================================
# 4. SMOTE Wrapper
# ============================================================

class SmoteBalancer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.smote = SMOTE(
            sampling_strategy=self.config.smote_sampling_strategy,
            k_neighbors=self.config.smote_k_neighbors,
            random_state=self.config.random_state,
        )

    def fit_resample(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        self.logger.info("Applying SMOTE on preprocessed train data...")
        self.logger.info(f"Original train balance: {y_train.value_counts(normalize=True).to_dict()}")
        X_res, y_res = self.smote.fit_resample(X_train, y_train)
        self.logger.info(f"SMOTE done. Train size: {X_train.shape} -> {X_res.shape}")
        self.logger.info(f"Resampled train balance: {y_res.value_counts(normalize=True).to_dict()}")
        return X_res, y_res

# ============================================================
# 5. Model Trainer
# ============================================================

class ChurnModelTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.xgb_model: Optional[XGBClassifier] = None
        self.lgbm_model: Optional[LGBMClassifier] = None
        self.cat_model: Optional[CatBoostClassifier] = None
        self.ensemble_model: str = "soft_voting"
        self.top_features: Optional[List[str]] = None
        self.shap_values_xgb: Optional[np.ndarray] = None
        self.shap_values_lgbm: Optional[np.ndarray] = None
        self.shap_values_cat: Optional[np.ndarray] = None

    @staticmethod
    def _calc_scale_pos_weight(y: pd.Series) -> float:
        neg = (y == 0).sum()
        pos = (y == 1).sum()
        return neg / pos if pos > 0 else 1.0

    # ---------- Base Models (Dùng cho SHAP FS) ----------
    # Sửa: Xóa side-effect (self.xgb_model = model), chỉ return
    def train_xgb_base(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> XGBClassifier:
        spw = self._calc_scale_pos_weight(y_train)
        self.logger.info(f"[XGB-Base] Training... (scale_pos_weight = {spw:.2f})")
        params = dict(
            n_estimators=1000, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=self.config.random_state,
            eval_metric="auc", scale_pos_weight=spw, tree_method="hist",
            early_stopping_rounds=50,
        )
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self.logger.info(f"[XGB-Base] Trained with {model.best_iteration} iterations (AUC: {model.best_score:.4f}).")
        # self.xgb_model = model # Xóa side-effect
        return model

    # Sửa: Xóa side-effect
    def train_catboost_base(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> CatBoostClassifier:
        spw = self._calc_scale_pos_weight(y_train)
        self.logger.info(f"[CatBoost-Base] Training... (scale_pos_weight = {spw:.2f})")
        model = CatBoostClassifier(
            iterations=1000, depth=5, learning_rate=0.05,
            random_state=self.config.random_state, loss_function="Logloss",
            eval_metric="AUC", scale_pos_weight=spw, verbose=False,
            early_stopping_rounds=50,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        self.logger.info(f"[CatBoost-Base] Trained with {model.get_best_iteration()} iterations (AUC: {model.get_best_score()['validation']['AUC']:.4f}).")
        # self.cat_model = model # Xóa side-effect
        return model

    # Sửa: Xóa side-effect
    def train_lightgbm_base(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> LGBMClassifier:
        spw = self._calc_scale_pos_weight(y_train)
        self.logger.info(f"[LightGBM-Base] Training... (scale_pos_weight = {spw:.2f})")
        model = LGBMClassifier(
            n_estimators=1000, max_depth=5, learning_rate=0.05,
            random_state=self.config.random_state, objective="binary",
            metric="auc", scale_pos_weight=spw, verbose=-1,
            callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        self.logger.info(f"[LightGBM-Base] Trained with {model.best_iteration_} iterations (AUC: {model.best_score_['valid_0']['auc']:.4f}).")
        # self.lgbm_model = model # Xóa side-effect
        return model

    # ---------- SHAP Feature Selection (Parallel) ----------

    # Sửa: Chuyển thành staticmethod để main.py gọi
    @staticmethod
    def _get_shap_values(model: Any, X: pd.DataFrame, model_type: str) -> pd.Series:
        # Static method không có self.logger, dùng logging
        logging.info(f"[SHAP-FS] Calculating for {model_type}...")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X, check_additivity=False)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            importance = pd.Series(mean_abs_shap, index=X.columns, name=model_type)
            logging.info(f"[SHAP-FS] Done for {model_type}.")
            return importance
        except Exception as e:
            logging.warning(f"[SHAP-FS] Error for {model_type}: {e}")
            return pd.Series(dtype=float, name=model_type)

    # Sửa: Đổi tên và chỉ xử lý kết quả (logic parallel chuyển ra main.py)
    def process_shap_importances(self, shap_importance_results: List[pd.Series]) -> List[str]:
        """
        Nhận kết quả SHAP (tính song song) và xử lý (scale, aggregate, top-N).
        """
        self.logger.info("[SHAP-FS] Processing aggregated SHAP results...")
        
        df_importances = pd.concat(shap_importance_results, axis=1).fillna(0)
        
        # Chuẩn hóa (Min-Max)
        scaler = MinMaxScaler()
        df_importances_scaled = pd.DataFrame(
            scaler.fit_transform(df_importances),
            columns=df_importances.columns,
            index=df_importances.index
        )
        
        df_importances_scaled["mean_importance"] = df_importances_scaled.mean(axis=1)
        final_importances = df_importances_scaled["mean_importance"].sort_values(ascending=False)
        
        top_features = final_importances.head(self.config.shap_top_n_features).index.tolist()
        
        self.logger.info(f"[SHAP FS] Top 5 features: {final_importances.head(5).index.tolist()}")
        self.logger.info(f"[SHAP FS] Selected {len(top_features)} features.")
        
        self.top_features = top_features
        return top_features

    # ---------- Optuna (với show_progress_bar=True) ----------
    
    # XÓA: def _create_tqdm_callback(self) -> TqdmCallback: ...
    
    # Sửa: Xóa side-effect
    def tune_xgb_optuna(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> XGBClassifier:
        spw = self._calc_scale_pos_weight(y_train)
        self.logger.info(f"\n[Optuna-XGB] Tuning... (scale_pos_weight = {spw:.2f})")
        # XÓA: tqdm_callback = self._create_tqdm_callback()

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            }
            model = XGBClassifier(
                random_state=self.config.random_state, eval_metric="auc",
                tree_method="hist", scale_pos_weight=spw, early_stopping_rounds=50,
                **params,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            # XÓA: tqdm_callback.update_best_trial(trial.number, model.best_score)
            return model.best_score

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective, n_trials=self.config.optuna_n_trials,
            timeout=self.config.optuna_timeout,
            show_progress_bar=True # <--- THAY ĐỔI
            # callbacks=[tqdm_callback] # <--- XÓA
        )
        # XÓA: tqdm_callback.tqdm.close()
        self.logger.info(f"[Optuna-XGB] Best value (AUC): {study.best_value:.4f}")
        
        best_model = XGBClassifier(
            random_state=self.config.random_state, eval_metric="auc",
            tree_method="hist", scale_pos_weight=spw, early_stopping_rounds=50,
            **study.best_params,
        )
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        # self.xgb_model = best_model # Xóa side-effect
        return best_model

    # Sửa: Xóa side-effect
    def tune_lgbm_optuna(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> LGBMClassifier:
        spw = self._calc_scale_pos_weight(y_train)
        self.logger.info(f"\n[Optuna-LGBM] Tuning... (scale_pos_weight = {spw:.2f})")
        # XÓA: tqdm_callback = self._create_tqdm_callback()

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "num_leaves": trial.suggest_int("num_leaves", 10, 50),
            }
            model = LGBMClassifier(
                random_state=self.config.random_state, objective="binary", metric="auc",
                scale_pos_weight=spw, verbose=-1,
                callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
                **params,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            auc = model.best_score_['valid_0']['auc']
            # XÓA: tqdm_callback.update_best_trial(trial.number, auc)
            return auc

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective, n_trials=self.config.optuna_n_trials,
            timeout=self.config.optuna_timeout,
            show_progress_bar=True # <--- THAY ĐỔI
            # callbacks=[tqdm_callback] # <--- XÓA
        )
        # XÓA: tqdm_callback.tqdm.close()
        self.logger.info(f"[Optuna-LGBM] Best value (AUC): {study.best_value:.4f}")
        
        best_model = LGBMClassifier(
            random_state=self.config.random_state, objective="binary", metric="auc",
            scale_pos_weight=spw, verbose=-1,
            callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
            **study.best_params,
        )
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        # self.lgbm_model = best_model # Xóa side-effect
        return best_model

    # Sửa: Xóa side-effect
    def tune_cat_optuna(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> CatBoostClassifier:
        spw = self._calc_scale_pos_weight(y_train)
        self.logger.info(f"\n[Optuna-CatBoost] Tuning... (scale_pos_weight = {spw:.2f})")
        # XÓA: tqdm_callback = self._create_tqdm_callback()

        def objective(trial: optuna.Trial) -> float:
            params = {
                "iterations": trial.suggest_int("iterations", 300, 1500),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
            model = CatBoostClassifier(
                random_state=self.config.random_state, loss_function="Logloss",
                eval_metric="AUC", scale_pos_weight=spw, verbose=False,
                early_stopping_rounds=50,
                **params,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            auc = model.get_best_score()['validation']['AUC']
            # XÓA: tqdm_callback.update_best_trial(trial.number, auc)
            return auc

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective, n_trials=self.config.optuna_n_trials,
            timeout=self.config.optuna_timeout,
            show_progress_bar=True # <--- THAY ĐỔI
            # callbacks=[tqdm_callback] # <--- XÓA
        )
        # XÓA: tqdm_callback.tqdm.close()
        self.logger.info(f"[Optuna-CatBoost] Best value (AUC): {study.best_value:.4f}")

        best_model = CatBoostClassifier(
            random_state=self.config.random_state, loss_function="Logloss",
            eval_metric="AUC", scale_pos_weight=spw, verbose=False,
            early_stopping_rounds=50,
            **study.best_params,
        )
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        # self.cat_model = best_model # Xóa side-effect
        return best_model

    # ---------- Ensemble & Predict ----------
    
    def build_ensemble(self):
        if not (self.xgb_model and self.lgbm_model and self.cat_model):
            self.logger.error("Need tuned models for ensemble. Models were not set on trainer.")
            raise RuntimeError("Need tuned xgb, lgbm, cat models for ensemble.")
        if not self.top_features:
            self.logger.error("top_features must be set for ensemble.")
            raise RuntimeError("top_features (from SHAP) must be set.")
        self.ensemble_model = "soft_voting"
        self.logger.info("\nEnsemble model is ready (soft voting).")

    def predict_proba_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        if self.ensemble_model != "soft_voting":
            raise RuntimeError("Ensemble not built.")
        if self.top_features is None:
            raise RuntimeError("top_features must be set before predicting.")

        X_filtered = X[self.top_features]
        p_xgb = self.xgb_model.predict_proba(X_filtered)[:, 1]
        p_lgbm = self.lgbm_model.predict_proba(X_filtered)[:, 1]
        p_cat = self.cat_model.predict_proba(X_filtered)[:, 1]
        proba = (p_xgb + p_lgbm + p_cat) / 3.0
        return proba

    # ---------- SHAP (Final) ----------
    
    def calculate_final_shap_values(self, X_data: pd.DataFrame):
        if self.top_features is None:
            raise RuntimeError("top_features must be set.")
        if not (self.xgb_model and self.lgbm_model and self.cat_model):
            raise RuntimeError("Final models must be trained and set on trainer.")

        self.logger.info("\n[SHAP-Final] Calculating SHAP values for final models...")
        X_filtered = X_data[self.top_features]

        self.logger.info("[SHAP-Final] Calculating for XGB...")
        explainer_xgb = shap.TreeExplainer(self.xgb_model)
        self.shap_values_xgb = explainer_xgb(X_filtered)

        self.logger.info("[SHAP-Final] Calculating for LGBM...")
        explainer_lgbm = shap.TreeExplainer(self.lgbm_model)
        shap_vals_lgbm = explainer_lgbm.shap_values(X_filtered, check_additivity=False)
        self.shap_values_lgbm = shap.Explanation(
             values=shap_vals_lgbm[1], base_values=explainer_lgbm.expected_value,
             data=X_filtered, feature_names=X_filtered.columns
        )

        self.logger.info("[SHAP-Final] Calculating for CatBoost...")
        explainer_cat = shap.TreeExplainer(self.cat_model)
        shap_vals_cat = explainer_cat.shap_values(X_filtered, check_additivity=False)
        self.shap_values_cat = shap.Explanation(
             values=shap_vals_cat, base_values=explainer_cat.expected_value,
             data=X_filtered, feature_names=X_filtered.columns
        )
        self.logger.info("[SHAP-Final] All SHAP values calculated and stored.")

    # ---------- Save Models ----------
    
    def _save_model(self, model: Any, path: str, model_name: str):
        if model is None:
            self.logger.error(f"{model_name} model not trained. Cannot save.")
            raise RuntimeError(f"{model_name} model not trained.")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        self.logger.info(f"{model_name} model saved to {path}")

    def save_xgb(self, path: str):
        self._save_model(self.xgb_model, path, "XGB")

    def save_lgbm(self, path: str):
        self._save_model(self.lgbm_model, path, "LGBM")

    def save_cat(self, path: str):
        self._save_model(self.cat_model, path, "CatBoost")

    def save_ensemble(self, path: str):
        if self.ensemble_model is None or self.top_features is None:
            self.logger.error("Ensemble/top_features not ready. Cannot save.")
            raise RuntimeError("Ensemble or top_features not ready.")

        ensemble_data = {
            "xgb": self.xgb_model,
            "lgbm": self.lgbm_model,
            "cat": self.cat_model,
            "top_features": self.top_features,
        }
        
        with open(path, "wb") as f:
            pickle.dump(ensemble_data, f)
        self.logger.info(f"Ensemble (3 models + top_features) saved to {path}")