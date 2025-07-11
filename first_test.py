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

    fig_roc, ax_roc = plt.subplots(figsize=(2, 2))
    ax_roc.plot(fpr, tpr)
    ax_roc.plot(roc_point[0], roc_point[1], 'ro')
    ax_roc.set_title("ROC", fontsize=8)
    ax_roc.set_xlabel("FPR", fontsize=6)
    ax_roc.set_ylabel("TPR", fontsize=6)
    ax_roc.tick_params(labelsize=6)
    ax_roc.grid(True)

    fig_pr, ax_pr = plt.subplots(figsize=(2, 2))
    ax_pr.plot(recall, precision)
    ax_pr.plot(rec_point, prec_point, 'ro')
    ax_pr.set_title("PR", fontsize=8)
    ax_pr.set_xlabel("Recall", fontsize=6)
    ax_pr.set_ylabel("Precision", fontsize=6)
    ax_pr.tick_params(labelsize=6)
    ax_pr.grid(True)

    return fig_roc, fig_pr

# ---- Prevalence Bar
def draw_prevalence_bar(pred_prev):
    fig, ax = plt.subplots(figsize=(1.2, 2))
    low_pct = 1 - pred_prev
    high_pct = pred_prev

    ax.bar(0, high_pct, color='green', bottom=low_pct, width=0.8)
    ax.bar(0, low_pct, color='lightcoral', bottom=0, width=0.8)
    ax.text(0, 1 - high_pct / 2, f"High\n{high_pct*100:.1f}%", color='white', ha='center', va='center', fontsize=7)
    ax.text(0, low_pct / 2, f"Low\n{low_pct*100:.1f}%", color='white', ha='center', va='center', fontsize=7)

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig

# ---- App Layout
st.title("📊 Postoperative Predictive Performance Visualization")
tab1, tab2 = st.tabs(["🩺 Postoperative Sepsis", "🩸 Postoperative Bleeding"])

for tab, label in zip([tab1, tab2], ["sepsis", "bleeding"]):
    with tab:
        y_true, y_scores = generate_fake_data(seed=42 if label == "sepsis" else 24)
        st.subheader(f"{label.capitalize()} Model Output")
        threshold = st.slider(f"Threshold for {label} prediction", 0.0, 1.0, 0.5, 0.01)

        # Layout structure: empty | table | graphs | bar | empty
        col_empty_left, col_table, col_graphs, col_bar, col_empty_right = st.columns([0.175, 0.2, 0.3, 0.15, 0.175])

        # Table
        with col_table:
            metrics = compute_metrics(y_true, y_scores, threshold)
            df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
            st.markdown("<div style='font-size:14px'>", unsafe_allow_html=True)
            st.table(df.style.format(precision=3))
            st.markdown("</div>", unsafe_allow_html=True)

        # Graphs
        with col_graphs:
            fig_roc, fig_pr = plot_curves(y_true, y_scores, threshold)
            g1, g2 = st.columns(2)
            with g1: st.pyplot(fig_roc)
            with g2: st.pyplot(fig_pr)

        # Prevalence Bar
        with col_bar:
            st.pyplot(draw_prevalence_bar(np.mean(y_scores >= threshold)))