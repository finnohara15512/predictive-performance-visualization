import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, classification_report, roc_auc_score, average_precision_score
st.set_page_config(page_title="Postoperative Predictive Performance Visualization", layout="wide")

# ---- Generate Simulated Data ----
def generate_fake_data(seed=42):
    np.random.seed(seed)
    y_true = np.random.binomial(1, 0.3, 1000)
    y_scores = y_true * np.random.uniform(0.4, 1.0, 1000) + (1 - y_true) * np.random.uniform(0.0, 0.6, 1000)
    return y_true, y_scores

# ---- Compute Metrics ----
def compute_metrics(y_true, y_scores, threshold):
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    N = len(y_true)
    label_prev = np.mean(y_true)
    pred_prev = np.mean(y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / N
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    return pd.DataFrame([{
        "N": N,
        "Label Prevalence": label_prev,
        "Prediction Prevalence": pred_prev,
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "PPV (Precision)": ppv,
        "NPV": npv,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "ROC AUC": auc,
        "Average Precision": ap
    }]).T.rename(columns={0: "Value"}).round(3)

# ---- Plot ROC and PR ----
def plot_curves(y_true, y_scores, threshold):
    fpr, tpr, roc_thresh = roc_curve(y_true, y_scores)
    precision, recall, pr_thresh = precision_recall_curve(y_true, y_scores)

    # ROC
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label="ROC Curve")
    point_fpr, point_tpr, _ = roc_curve(y_true, y_scores >= threshold)
    ax_roc.plot(point_fpr[1], point_tpr[1], 'ro', label=f"Threshold = {threshold:.2f}")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()

    # PR
    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(recall, precision, label="PR Curve")
    pr_point = ((y_scores >= threshold).astype(int))
    tp = np.sum((pr_point == 1) & (y_true == 1))
    fp = np.sum((pr_point == 1) & (y_true == 0))
    fn = np.sum((pr_point == 0) & (y_true == 1))
    precision_point = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_point = tp / (tp + fn) if (tp + fn) > 0 else 0
    ax_pr.plot(recall_point, precision_point, 'ro', label=f"Threshold = {threshold:.2f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend()

    return fig_roc, fig_pr

# ---- Metric Descriptions ----
metric_descriptions = {
    "N": "The total number of predictions made, which equals the number of samples evaluated.",
    "Label Prevalence": "The proportion of positive cases in the true labels (i.e., how often the event actually happened).",
    "Prediction Prevalence": "The proportion of positive predictions made by the model at the selected threshold.",
    "Sensitivity (Recall)": "The proportion of true positives identified among all actual positive cases. High sensitivity means fewer false negatives.",
    "Specificity": "The proportion of true negatives correctly identified among all actual negative cases. High specificity means fewer false positives.",
    "PPV (Precision)": "The proportion of true positives among all positive predictions. It tells you how often a positive prediction is actually correct.",
    "NPV": "The proportion of true negatives among all negative predictions. It tells you how often a negative prediction is actually correct.",
    "Accuracy": "The proportion of correct predictions (both positives and negatives) out of all predictions made.",
    "F1 Score": "The harmonic mean of precision and recall. It balances both metrics and is especially useful in imbalanced datasets.",
    "ROC AUC": "Area under the ROC curve. A measure of the model's ability to discriminate between classes, with 1 being perfect.",
    "Average Precision": "The area under the precision-recall curve, summarizing the tradeoff between precision and recall at different thresholds."
}

# ---- Layout ----
st.title("📊 Postoperative Predictive Performance Visualization")

tab1, tab2 = st.tabs(["🩺 Postoperative Sepsis", "🩸 Postoperative Bleeding"])

for tab, label in zip([tab1, tab2], ["sepsis", "bleeding"]):
    with tab:
        st.subheader(f"Model Performance for Postoperative {label.capitalize()}")

        # Load data
        y_true, y_scores = generate_fake_data(seed=42 if label == "sepsis" else 24)

        # Threshold slider
        threshold = st.slider(f"Select probability threshold for {label} prediction", 0.0, 1.0, 0.5, 0.01)

        # Plot ROC and PR side by side
        col1, col2 = st.columns(2)
        fig_roc, fig_pr = plot_curves(y_true, y_scores, threshold)
        with col1:
            st.pyplot(fig_roc)
        with col2:
            st.pyplot(fig_pr)

        # Metrics Table
        metrics_df = compute_metrics(y_true, y_scores, threshold)
        for metric, row in metrics_df.iterrows():
            with st.expander(metric):
                st.markdown(f"**{metric}**: {row['Value']}")
                st.write(metric_descriptions.get(metric, "No description available."))