"""
File: evaluator.py
Chứa Lớp ChurnEvaluator để tính toán và in metrics.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix,
    classification_report
)

class ChurnEvaluator:
    def __init__(self, y_true: pd.Series, y_proba: np.ndarray, threshold: float = 0.5):
        self.y_true = y_true
        self.y_proba = y_proba
        self.threshold = threshold
        self.y_pred = (self.y_proba >= self.threshold).astype(int)
        self.logger = logging.getLogger(__name__)

    def basic_metrics(self) -> Dict[str, float]:
        auc = roc_auc_score(self.y_true, self.y_proba)
        acc = accuracy_score(self.y_true, self.y_pred)
        prec = precision_score(self.y_true, self.y_pred, zero_division=0)
        rec = recall_score(self.y_true, self.y_pred, zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, zero_division=0)

        self.logger.info("------ Evaluation ------")
        self.logger.info(f"AUC:         {auc:.4f}")
        self.logger.info(f"Accuracy:    {acc:.4f}")
        self.logger.info(f"Precision:   {prec:.4f}")
        self.logger.info(f"Recall:      {rec:.4f}")
        self.logger.info(f"F1:          {f1:.4f}")
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        self.logger.info(f"Confusion Matrix:\n{cm}")
        
        report = classification_report(self.y_true, self.y_pred, digits=4)
        self.logger.info(f"Classification Report:\n{report}")

        return {"auc": auc, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    def recall_lift_at_k(self, top_k: float = 0.2) -> Dict[str, float]:
        df = pd.DataFrame({"y_true": self.y_true, "proba": self.y_proba})
        df = df.sort_values("proba", ascending=False)

        cutoff = int(len(df) * top_k)
        df_top = df.iloc[:cutoff]

        total_churns = df["y_true"].sum()
        recall_at_k = df_top["y_true"].sum() / total_churns if total_churns > 0 else 0.0

        churn_rate_overall = df["y_true"].mean()
        churn_rate_top = df_top["y_true"].mean()
        lift_at_k = churn_rate_top / churn_rate_overall if churn_rate_overall > 0 else 0.0

        self.logger.info(f"Recall@Top{int(top_k*100)}%: {recall_at_k:.4f}")
        self.logger.info(f"Lift@Top{int(top_k*100)}%:   {lift_at_k:.4f}")

        return {"recall_at_k": recall_at_k, "lift_at_k": lift_at_k}