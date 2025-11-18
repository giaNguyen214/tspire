"""
File: evaluator.py
Chứa Lớp ChurnEvaluator để tính toán và in metrics.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score,
    f1_score, 
    accuracy_score, 
    confusion_matrix,
    classification_report,
    auc, 
    precision_recall_curve,
    roc_curve
)
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import os
from utils.config import Config
import seaborn as sns
    
    
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

    def pr_auc(self) -> float:
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_proba)
        return auc(recall, precision)
    
    def calibration_plot(self):
        prob_true, prob_pred = calibration_curve(self.y_true, self.y_proba, n_bins=10)

        plt.figure()
        plt.plot(prob_pred, prob_true, marker='o')
        plt.plot([0,1], [0,1], "--")
        plt.xlabel("Predicted probability")
        plt.ylabel("True probability")
        plt.title("Calibration Plot")
        plt.savefig(os.path.join(Config.eval_plot_dir, "calibration_plot.png"), bbox_inches="tight")
        plt.close()

        self.logger.info("[Evaluator] Saved calibration plot.")


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
    
    
    def save_all_plots(self):
        """
        Lưu tất cả các biểu đồ đánh giá model:
        - ROC Curve
        - Precision-Recall Curve
        - Confusion Matrix
        - Lift Curve
        - Gain Chart
        """


        out_dir = Config.eval_plot_dir

        # ========== ROC CURVE ==========
        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(self.y_true, self.y_proba):.4f}")
        plt.plot([0,1], [0,1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(os.path.join(out_dir, "roc_curve.png"), bbox_inches="tight")
        plt.close()

        # ========== PRECISION-RECALL CURVE ==========
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_proba)
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.savefig(os.path.join(out_dir, "pr_curve.png"), bbox_inches="tight")
        plt.close()

        # ========== CONFUSION MATRIX ==========
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), bbox_inches="tight")
        plt.close()

        # ========== LIFT CURVE ==========
        df = pd.DataFrame({"y_true": self.y_true, "proba": self.y_proba})
        df = df.sort_values("proba", ascending=False)
        df["cum_churn"] = df["y_true"].cumsum()
        df["pct_samples"] = np.arange(1, len(df)+1) / len(df)
        df["pct_churns"] = df["cum_churn"] / df["y_true"].sum()

        plt.figure()
        plt.plot(df["pct_samples"], df["pct_churns"])
        plt.plot([0,1], [0,1], "--")
        plt.xlabel("Percentage of Samples")
        plt.ylabel("Percentage of Churns Captured")
        plt.title("Lift Curve (Cumulative Capture Curve)")
        plt.savefig(os.path.join(out_dir, "lift_curve.png"), bbox_inches="tight")
        plt.close()

        # ========== GAIN CHART ==========
        plt.figure()
        plt.plot(df["pct_samples"], df["pct_churns"])
        plt.xlabel("Percentage of Samples")
        plt.ylabel("Gain (Cumulative Response)")
        plt.title("Gain Chart")
        plt.savefig(os.path.join(out_dir, "gain_chart.png"), bbox_inches="tight")
        plt.close()
        
        self.calibration_plot()

        self.logger.info("[Evaluator] All evaluation plots saved successfully.")
    
    
    def evaluate_all(self):
        """
        Chạy toàn bộ evaluation pipeline theo yêu cầu đề bài:
        - ROC-AUC + PR-AUC
        - Precision/Recall/F1 @ threshold
        - Confusion matrix
        - Recall@Top20% & Lift@Top20%
        - Calibration plot
        - Save evaluation plots
        """
        metrics = self.basic_metrics()

        # Add PR-AUC
        pr_auc_val = self.pr_auc()
        metrics["pr_auc"] = pr_auc_val
        self.logger.info(f"PR-AUC: {pr_auc_val:.4f}")

        # Add Lift/Recall@Top20%
        lift_metrics = self.recall_lift_at_k(top_k=0.20)
        metrics.update(lift_metrics)

        # Save all evaluator plots
        self.save_all_plots()

        return metrics

