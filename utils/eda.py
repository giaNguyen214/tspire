import os
import matplotlib
matplotlib.use("Agg")   # quan trọng để tránh lỗi Tkinter + multi-thread

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.config import Config


# ============================================================
# 1. Correlation Matrix
# ============================================================
def plot_correlation_matrix(df: pd.DataFrame):
    numerical_df = df.select_dtypes(include=["int64", "float64"])
    
    corr_matrix = numerical_df.corr()

    plt.figure(figsize=(24, 20))
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        linewidths=0.5,
        cbar_kws={"label": "Pearson Correlation Coefficient"}
    )

    plt.title("Correlation Matrix of Numerical Features", fontsize=20)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    out_path = os.path.join(Config.eda_dir, "correlation_matrix.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


# ============================================================
# 2. Outlier Box Plots
# ============================================================
def check_outliers(df: pd.DataFrame, cols_to_check: list):

    n_cols = 3
    n_rows = (len(cols_to_check) + n_cols - 1) // n_cols

    plt.figure(figsize=(18, 5 * n_rows))

    for i, col in enumerate(cols_to_check):
        plt.subplot(n_rows, n_cols, i + 1)

        if df[col].max() > df[col].quantile(0.75) * 5:
            sns.boxplot(y=df[col], color="skyblue")
            plt.yscale("log")
            plt.title(f"Box Plot of {col} (Log Scale)")
        else:
            sns.boxplot(y=df[col], color="lightcoral")
            plt.title(f"Box Plot of {col}")

        plt.ylabel(col)

    plt.tight_layout()
    out_path = os.path.join(Config.eda_dir, "outlier_box_plots.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


# ============================================================
# 3. Target Distribution
# ============================================================
def check_target_distribution(df: pd.DataFrame, target_col: str):

    # --- tính toán ---
    counts = df[target_col].value_counts()
    ratio = df[target_col].value_counts(normalize=True).apply(lambda x: f"{x:.2%}")

    print("\n--- Target Distribution ---")
    print(counts)
    print("\nTarget Ratio:")
    print(ratio)

    # --- ép kiểu để seaborn không warning ---
    df[target_col] = df[target_col].astype(int)

    plt.figure(figsize=(7, 5))
    sns.countplot(
        x=target_col,
        data=df,
        hue=target_col,        # tránh warning palette
        palette="viridis",
        legend=False
    )

    plt.title(f"Distribution of {target_col} (0=Active, 1=Churned)")
    plt.xlabel("Churn Label")
    plt.ylabel("Count")
    plt.tight_layout()

    out_path = os.path.join(Config.eda_dir, "target_distribution.png")
    plt.savefig(out_path)
    plt.close()

    return out_path


# ============================================================
# MAIN EDA ENTRYPOINT
# ============================================================
def run_eda(df: pd.DataFrame, numerical_cols: list):

    os.makedirs(Config.eda_dir, exist_ok=True)

    corr_path = plot_correlation_matrix(df)
    outlier_path = check_outliers(df, cols_to_check=numerical_cols)
    target_path = check_target_distribution(df, target_col="churn_label")

    return {
        "correlation_matrix": corr_path,
        "outlier_plots": outlier_path,
        "target_distribution": target_path
    }
