"""
Postoperative Predictive Performance Visualization App

This Streamlit app visualizes classification performance for models using
precomputed confusion matrices at various thresholds.

Expected CSV Input Format:
- Each file must be in the './data/' directory
- Required columns: Threshold, TP, FP, FN, TN
- One row per threshold

Threshold: The cutoff used to separate class 0 and 1 (inclusive upper bound)
TP: True Positives at that threshold
FP: False Positives at that threshold
FN: False Negatives at that threshold
TN: True Negatives at that threshold

Expected files:
- news_scores.csv (used in Sepsis tab)
- gbs_scores.csv (used in Bleeding tab)
- sepsis_custom_scores.csv (optional alternative)
- bleeding_custom_scores.csv (optional alternative)
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Data utilities
# ----------------------------

def load_confusion_data(filename: str) -> pd.DataFrame:
    path = f"./data/{filename}"
    return pd.read_csv(path)

def compute_metrics_from_row(row) -> dict:
    tp, fp, fn, tn = row["TP"], row["FP"], row["FN"], row["TN"]
    total = tp + fp + fn + tn

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    return {
        "Threshold": row["Threshold"],
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "PPV (Precision)": precision,
        "NPV": npv,
        "Accuracy": accuracy,
        "F1 Score": f1
    }

def compute_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [compute_metrics_from_row(row) for _, row in df.iterrows()]
    return pd.DataFrame(metrics)

# ----------------------------
# Plotting utilities
# ----------------------------

def plot_roc_curve(metrics_df: pd.DataFrame):
    tpr = metrics_df["Sensitivity (Recall)"]
    fpr = 1 - metrics_df["Specificity"]

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.plot(fpr, tpr, marker="o")
    ax.set_xlabel("False Positive Rate", fontsize=6)
    ax.set_ylabel("True Positive Rate", fontsize=6)
    ax.set_title("ROC Curve", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True)
    return fig

def plot_pr_curve(metrics_df: pd.DataFrame):
    recall = metrics_df["Sensitivity (Recall)"]
    precision = metrics_df["PPV (Precision)"]

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.plot(recall, precision, marker="o")
    ax.set_xlabel("Recall", fontsize=6)
    ax.set_ylabel("Precision", fontsize=6)
    ax.set_title("Precision-Recall Curve", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True)
    return fig

# ----------------------------
# Streamlit App Layout
# ----------------------------

st.set_page_config(page_title="Postoperative Predictive Performance Visualization", layout="wide")
st.title("Postoperative Predictive Performance Visualization")

tabs = {
    "Postoperative Sepsis": "news_scores.csv",
    "Postoperative Bleeding": "gbs_scores.csv"
}

for label, filename in tabs.items():
    with st.tab(label):
        st.subheader(f"Model Output: {label}")
        try:
            df = load_confusion_data(filename)
        except FileNotFoundError:
            st.error(f"Could not find file './data/{filename}'")
            continue

        metrics_df = compute_all_metrics(df)

        st.markdown("#### Classification Metrics by Threshold")
        st.dataframe(metrics_df.round(3), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_roc_curve(metrics_df))
        with col2:
            st.pyplot(plot_pr_curve(metrics_df))