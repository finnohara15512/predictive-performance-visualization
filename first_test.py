import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score, average_precision_score

st.set_page_config(page_title="Postoperative Predictive Performance Visualization", layout="wide")

# ---- Simulated Data
def generate_fake_data(seed=42):
    np.random.seed(seed)
    y_true = np.random.binomial(1, 0.3, 1000)
    y_scores = y_true * np.random.uniform(0.4, 1.0, 1000) + (1 - y_true) * np.random.uniform(0.0, 0.6, 1000)
    return y_true, y_scores

# ---- Compute Metrics
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

# ---- Descriptions
metric_descriptions = {
    "N": "The total number of predictions made.",
    "Label Prevalence": "Proportion of positive cases in ground truth.",
    "Prediction Prevalence": "Proportion of positive predictions by model.",
    "Sensitivity (Recall)": "True positives / All actual positives.",
    "Specificity": "True negatives / All actual negatives.",
    "PPV (Precision)": "True positives / All predicted positives.",
    "NPV": "True negatives / All predicted negatives.",
    "Accuracy": "Correct predictions / All predictions.",
    "F1 Score": "Harmonic mean of precision and recall.",
    "ROC AUC": "Area under ROC curve.",
    "Average Precision": "Area under the PR curve."
}

# ---- Plot ROC and PR
def plot_curves(y_true, y_scores, threshold):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    pred = (y_scores >= threshold).astype(int)
    roc_point = [np.mean(pred[y_true == 0]), np.mean(pred[y_true == 1])]
    tp = np.sum((pred == 1) & (y_true == 1))
    fp = np.sum((pred == 1) & (y_true == 0))
    fn = np.sum((pred == 0) & (y_true == 1))
    prec_point = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_point = tp / (tp + fn) if (tp + fn) > 0 else 0

    # ROC
    fig_roc, ax_roc = plt.subplots(figsize=(2.2, 2.2))
    ax_roc.plot(fpr, tpr)
    ax_roc.plot(roc_point[0], roc_point[1], 'ro')
    ax_roc.set_title("ROC")
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.grid(True)

    # PR
    fig_pr, ax_pr = plt.subplots(figsize=(2.2, 2.2))
    ax_pr.plot(recall, precision)
    ax_pr.plot(rec_point, prec_point, 'ro')
    ax_pr.set_title("PR")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid(True)

    return fig_roc, fig_pr

# ---- Prevalence Bar
def draw_prevalence_bar(pred_prev):
    fig, ax = plt.subplots(figsize=(0.5, 2.2))
    low_pct = 1 - pred_prev
    high_pct = pred_prev

    ax.bar(0, high_pct, color='green', bottom=low_pct, width=0.4)
    ax.bar(0, low_pct, color='lightcoral', bottom=0, width=0.4)
    ax.text(0, 1 - high_pct / 2, f"High\n{high_pct*100:.1f}%", color='white', ha='center', va='center', fontsize=7)
    ax.text(0, low_pct / 2, f"Low\n{low_pct*100:.1f}%", color='white', ha='center', va='center', fontsize=7)

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig

# ---- HTML Tooltip Builder
def metric_label_html(name, desc):
    return f"""
    <div style="display: flex; align-items: center;">
      <span title="{desc}" style="cursor: help; color: #555; margin-right: 6px;">❓</span>
      <span>{name}</span>
    </div>
    """

# ---- Main App Layout
st.title("📊 Postoperative Predictive Performance Visualization")
tab1, tab2 = st.tabs(["🩺 Postoperative Sepsis", "🩸 Postoperative Bleeding"])

for tab, label in zip([tab1, tab2], ["sepsis", "bleeding"]):
    with tab:
        y_true, y_scores = generate_fake_data(seed=42 if label == "sepsis" else 24)
        st.subheader(f"{label.capitalize()} Model Output")

        threshold = st.slider(f"Threshold for {label} prediction", 0.0, 1.0, 0.5, 0.01)

        with st.container():
            st.markdown("<div style='max-width:1080px;margin:auto;'>", unsafe_allow_html=True)
            col_metrics, col_roc, col_pr, col_bar = st.columns([3, 2, 2, 1])

            with col_metrics:
                metrics = compute_metrics(y_true, y_scores, threshold)
                metric_table = pd.DataFrame([
                    {
                        "Metric": metric_label_html(k, metric_descriptions.get(k, "")),
                        "Value": round(v, 3)
                    } for k, v in metrics.items()
                ])
                st.markdown("### Classification Metrics")
                st.write(metric_table.to_html(escape=False, index=False), unsafe_allow_html=True)

            with col_roc:
                fig_roc, fig_pr = plot_curves(y_true, y_scores, threshold)
                st.pyplot(fig_roc)

            with col_pr:
                st.pyplot(fig_pr)

            with col_bar:
                st.markdown("<div style='padding-top:20px;'>", unsafe_allow_html=True)
                st.pyplot(draw_prevalence_bar(np.mean(y_scores >= threshold)))
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)