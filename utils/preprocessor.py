"""
File: preprocessor.py
Chứa Lớp ChurnPreprocessor để xử lý features.
"""
import logging
import pickle
import pandas as pd
from typing import List, Optional
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class ChurnPreprocessor:
    def __init__(
        self,
        numerical_cols: List[str],
        categorical_cols: List[str],
        label_encoded_cols: Optional[List[str]] = None,
    ):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.label_encoded_cols = label_encoded_cols or ["city"]
        self.onehot_cols = [c for c in categorical_cols if c not in self.label_encoded_cols]
        self.preprocessor: Optional[ColumnTransformer] = None
        self.logger = logging.getLogger(__name__)

    def build_preprocessor(self) -> ColumnTransformer:
        self.logger.info("Building preprocessor pipelines...")
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]
        )

        onehot_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]
        )

        label_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "label_encode",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.numerical_cols),
                ("ohe", onehot_transformer, self.onehot_cols),
                ("label", label_transformer, self.label_encoded_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")

        self.preprocessor = preprocessor
        self.logger.info("Preprocessor built successfully.")
        return preprocessor

    def fit(self, X_train: pd.DataFrame):
        if self.preprocessor is None:
            self.build_preprocessor()
        self.logger.info("Fitting preprocessor on X_train...")
        self.preprocessor.fit(X_train)
        self.logger.info("Preprocessor fitted.")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            self.logger.error("Preprocessor not fitted yet. Call fit() first.")
            raise RuntimeError("Preprocessor not fitted yet.")
        return self.preprocessor.transform(X)

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        self.fit(X_train)
        return self.transform(X_train)
    
    def get_feature_names(self) -> List[str]:
        if self.preprocessor is None:
            self.logger.error("Preprocessor not fitted yet.")
            raise RuntimeError("Preprocessor not fitted yet.")
        return self.preprocessor.get_feature_names_out()

    def save(self, path: str):
        if self.preprocessor is None:
            self.logger.error("Preprocessor not fitted yet. Cannot save.")
            raise RuntimeError("Preprocessor not fitted yet.")
        with open(path, "wb") as f:
            pickle.dump(self.preprocessor, f)
        self.logger.info(f"Preprocessor saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            self.preprocessor = pickle.load(f)
        self.logger.info(f"Preprocessor loaded from {path}")