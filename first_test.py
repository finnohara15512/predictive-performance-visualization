import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score, average_precision_score

st.set_page_config(page_title="Postoperative Predictive Performance Visualization", layout="wide")

# ---- Simulated data
def generate_fake_data(seed=42):
    np.random.seed(seed)
    y_true = np.random.binomial(1, 0.3, 1000)
    y_scores = y_true * np.random.uniform(0.4, 1.0, 1000) + (1 - y_true) * np.random.uniform(0.0, 0.6, 1000)
    return y_true, y_scores

# ---- Metrics
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

    metrics = {
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
    }
    return metrics

# ---- Metric Descriptions
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

# ---- Plots
def plot_curves(y_true, y_scores, threshold):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    # ROC
    fig_roc, ax_roc = plt.subplots(figsize=(2.5, 2.5))
    ax_roc.plot(fpr, tpr)
    pred = (y_scores >= threshold).astype(int)
    roc_point = [np.mean(pred[y_true == 0]), np.mean(pred[y_true == 1])]
    ax_roc.plot(roc_point[0], roc_point[1], 'ro')
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC")
    ax_roc.grid(True)

    # PR
    fig_pr, ax_pr = plt.subplots(figsize=(2.5, 2.5))
    ax_pr.plot(recall, precision)
    tp = np.sum((pred == 1) & (y_true == 1))
    fp = np.sum((pred == 1) & (y_true == 0))
    fn = np.sum((pred == 0) & (y_true == 1))
    prec_point = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_point = tp / (tp + fn) if (tp + fn) > 0 else 0
    ax_pr.plot(rec_point, prec_point, 'ro')
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("PR")
    ax_pr.grid(True)

    return fig_roc, fig_pr

# ---- Prevalence bar
def draw_prevalence_bar(pred_prev):
    fig, ax = plt.subplots(figsize=(0.8, 2.5))
    low_pct = 1 - pred_prev
    high_pct = pred_prev

    ax.bar(0, high_pct, color='green', bottom=low_pct, width=0.5)
    ax.bar(0, low_pct, color='lightcoral', bottom=0, width=0.5)

    ax.text(0, 1 - high_pct / 2, f"High\n{high_pct*100:.1f}%", color='white', ha='center', va='center', fontsize=8)
    ax.text(0, low_pct / 2, f"Low\n{low_pct*100:.1f}%", color='white', ha='center', va='center', fontsize=8)

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig

# ---- App Layout
st.title("📊 Postoperative Predictive Performance Visualization")
tab1, tab2 = st.tabs(["🩺 Postoperative Sepsis", "🩸 Postoperative Bleeding"])

for tab, label in zip([tab1, tab2], ["sepsis", "bleeding"]):
    with tab:
        st.subheader(f"Model Performance for Postoperative {label.capitalize()}")
        y_true, y_scores = generate_fake_data(seed=42 if label == "sepsis" else 24)

        with st.container():
            st.markdown("<div style='max-width:900px;margin:auto;'>",