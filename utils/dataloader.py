"""
File: dataloader.py
Chứa Lớp ChurnDataLoader để tải và chia dữ liệu.
"""
import logging
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from config import Config

class ChurnDataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_raw_data(self) -> pd.DataFrame:
        self.logger.info(f"Loading raw data from: {self.config.data_path}")
        
        # Sửa: Cho phép CSV hoặc XLSX (dùng sheet_name từ config)
        if self.config.data_path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(
                self.config.data_path, 
                sheet_name=self.config.sheet_name # <--- SỬA: Dùng sheet_name từ config
            )
        elif self.config.data_path.lower().endswith(".csv"):
            df = pd.read_csv(self.config.data_path)
        else:
            msg = f"Unknown file format: {self.config.data_path}"
            self.logger.error(msg)
            raise ValueError(msg)

        if "user_id" in df.columns:
            df = df.drop(columns=["user_id"])
            self.logger.info("Dropped 'user_id' column.")
        
        if self.config.target_column in df.columns:
            df[self.config.target_column] = df[self.config.target_column].astype(int)
        
        self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df

    def get_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        # ... (Không thay đổi)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if self.config.target_column in numerical_cols:
            numerical_cols = [c for c in numerical_cols if c != self.config.target_column]

        self.logger.info(f"Found {len(numerical_cols)} numerical columns.")
        self.logger.info(f"Found {len(categorical_cols)} categorical columns.")
        return numerical_cols, categorical_cols

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        # ... (Không thay đổi)
        train_ratio = self.config.train_ratio
        val_ratio = self.config.val_ratio
        test_ratio = self.config.test_ratio

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            msg = (
                f"Train/Val/Test ratios must sum to 1.0. "
                f"Current sum: {train_ratio + val_ratio + test_ratio}"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]

        # Step 1: split test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_ratio,
            random_state=self.config.random_state,
            stratify=y,
        )

        # Step 2: split train/val
        val_ratio_of_remaining = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_ratio_of_remaining,
            random_state=self.config.random_state,
            stratify=y_temp,
        )

        total = len(df)
        self.logger.info(f"Dataset split ({total} samples):")
        self.logger.info(f"  - Train: {len(X_train)} ({len(X_train)/total:.1%})")
        self.logger.info(f"  - Val:   {len(X_val)} ({len(X_val)/total:.1%})")
        self.logger.info(f"  - Test:  {len(X_test)} ({len(X_test)/total:.1%})")

        return X_train, X_val, X_test, y_train, y_val, y_test