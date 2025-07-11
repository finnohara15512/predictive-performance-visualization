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

def plot_roc_curve(metrics_df: pd.DataFrame, selected_threshold: float):
    tpr = metrics_df["Sensitivity (Recall)"]
    fpr = 1 - metrics_df["Specificity"]
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.plot(fpr, tpr, marker="o")
    selected_idx = metrics_df["Threshold"] == selected_threshold
    ax.plot(
        (1 - metrics_df["Specificity"][selected_idx]),
        metrics_df["Sensitivity (Recall)"][selected_idx],
        'ro', label="Selected Threshold"
    )
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=0.8, label='Random Classifier')
    ax.set_xlabel("1 - Specificity (False Positive Rate)", fontsize=6)
    ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=6)
    ax.set_title("ROC Curve", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True)
    ax.legend(fontsize=6)
    return fig

def plot_pr_curve(metrics_df: pd.DataFrame, selected_threshold: float, df: pd.DataFrame):
    recall = metrics_df["Sensitivity (Recall)"]
    precision = metrics_df["PPV (Precision)"]
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.plot(recall, precision, marker="o")
    selected_idx = metrics_df["Threshold"] == selected_threshold
    ax.plot(
        metrics_df["Sensitivity (Recall)"][selected_idx],
        metrics_df["PPV (Precision)"][selected_idx],
        'ro', label="Selected Threshold"
    )
    # Prevalence bar
    try:
        prevalence = df["TP"].sum() / (df["TP"].sum() + df["FN"].sum())
        ax.axhline(y=prevalence, color='gray', linestyle='--', linewidth=0.8, label="Prevalence")
    except Exception:
        pass
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, label="Random Classifier")
    ax.set_xlabel("Recall", fontsize=6)
    ax.set_ylabel("Precision", fontsize=6)
    ax.set_title("Precision-Recall Curve", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True)
    ax.legend(fontsize=6)
    return fig

# ----------------------------
# Streamlit App Layout
# ----------------------------

st.set_page_config(page_title="Postoperative Predictive Performance Visualization", layout="wide")
st.title("Postoperative Predictive Performance Visualization")

# Create tabs
tab_sepsis, tab_bleeding = st.tabs(["Postoperative Sepsis", "Postoperative Bleeding"])

# Sepsis Tab
with tab_sepsis:
    st.subheader("Model Output: Postoperative Sepsis")
    try:
        df = load_confusion_data("news_scores.csv")
        metrics_df = compute_all_metrics(df)

        # --- Threshold selection box (Box 1) ---
        st.markdown("### Threshold Selection")
        with st.container(border=True):
            st.markdown("**What is a threshold?**  \nIn classification models, a threshold is the cutoff point at which a predicted probability is converted into a class label. Adjusting this affects the trade-off between sensitivity and specificity.")
            threshold_values = df["Threshold"].sort_values().unique().tolist()
            step_val = float(threshold_values[1] - threshold_values[0]) if len(threshold_values) > 1 else 0.01
            selected_threshold = st.slider(
                "Choose Threshold",
                min_value=float(min(threshold_values)),
                max_value=float(max(threshold_values)),
                value=float(threshold_values[0]),
                step=step_val
            )

        # --- Metrics and plots (Box 2) ---
        selected_row = metrics_df[metrics_df["Threshold"] == selected_threshold].iloc[0]
        col_l, col_m, col_r = st.columns([0.3, 0.4, 0.3])
        with col_l:
            st.markdown("**Metrics at Selected Threshold**")
            metrics_table = pd.DataFrame([selected_row]).T.reset_index()
            metrics_table.columns = ["Metric", "Value"]
            st.dataframe(metrics_table, use_container_width=True, hide_index=True)
        with col_m:
            fig_roc = plot_roc_curve(metrics_df, selected_threshold)
            fig_pr = plot_pr_curve(metrics_df, selected_threshold, df)
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig_roc)
            with col2:
                st.pyplot(fig_pr)

        # --- Box 3: About Sepsis ---
        st.markdown("### About Postoperative Sepsis")
        st.markdown("""
Sepsis is a life-threatening response to infection that can occur after surgery.
Early identification using predictive models is critical to initiate timely treatment.

This model uses physiological and lab data to anticipate sepsis onset based on trends in early recovery.
""")

    except FileNotFoundError:
        st.error("Missing file: ./data/news_scores.csv")

# Bleeding Tab
with tab_bleeding:
    st.subheader("Model Output: Postoperative Bleeding")
    try:
        df = load_confusion_data("gbs_scores.csv")
        metrics_df = compute_all_metrics(df)

        # --- Threshold selection box (Box 1) ---
        st.markdown("### Threshold Selection")
        with st.container(border=True):
            st.markdown("**What is a threshold?**  \nIn classification models, a threshold is the cutoff point at which a predicted probability is converted into a class label. Adjusting this affects the trade-off between sensitivity and specificity.")
            threshold_values = df["Threshold"].sort_values().unique().tolist()
            step_val = float(threshold_values[1] - threshold_values[0]) if len(threshold_values) > 1 else 0.01
            selected_threshold = st.slider(
                "Choose Threshold",
                min_value=float(min(threshold_values)),
                max_value=float(max(threshold_values)),
                value=float(threshold_values[0]),
                step=step_val
            )

        # --- Metrics and plots (Box 2) ---
        selected_row = metrics_df[metrics_df["Threshold"] == selected_threshold].iloc[0]
        col_l, col_m, col_r = st.columns([0.3, 0.4, 0.3])
        with col_l:
            st.markdown("**Metrics at Selected Threshold**")
            metrics_table = pd.DataFrame([selected_row]).T.reset_index()
            metrics_table.columns = ["Metric", "Value"]
            st.dataframe(metrics_table, use_container_width=True, hide_index=True)
        with col_m:
            fig_roc = plot_roc_curve(metrics_df, selected_threshold)
            fig_pr = plot_pr_curve(metrics_df, selected_threshold, df)
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig_roc)
            with col2:
                st.pyplot(fig_pr)

        # --- Box 3: About Bleeding ---
        st.markdown("### About Postoperative Bleeding")
        st.markdown("""
Postoperative bleeding is a serious complication that can lead to reoperation, transfusion, or longer ICU stays.
Accurate early prediction helps clinicians intervene proactively.

This model leverages vital signs and clinical scores to predict bleeding risk after surgery.
""")

    except FileNotFoundError:
        st.error("Missing file: ./data/gbs_scores.csv")