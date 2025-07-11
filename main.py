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
    pred_pos = tp + fp
    pred_neg = tn + fn
    total = pred_pos + pred_neg

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    label_prevalence = (tp + fn) / total if total > 0 else 0
    pred_prevalence = pred_pos / total if total > 0 else 0

    return {
        "Threshold": row["Threshold"],
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "PPV (Precision)": precision,
        "NPV": npv,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Label Prevalence": label_prevalence,
        "Prediction Prevalence": pred_prevalence,
    }

def compute_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [compute_metrics_from_row(row) for _, row in df.iterrows()]
    return pd.DataFrame(metrics)

# ----------------------------
# Plotting utilities
# ----------------------------

def plot_roc_curve(metrics_df: pd.DataFrame, selected_threshold: float, show_selected: bool = True):
    tpr = metrics_df["Sensitivity (Recall)"]
    fpr = 1 - metrics_df["Specificity"]
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.plot(fpr, tpr, marker="o")
    if show_selected:
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

def plot_pr_curve(metrics_df: pd.DataFrame, selected_threshold: float, df: pd.DataFrame, show_selected: bool = True):
    recall = metrics_df["Sensitivity (Recall)"]
    precision = metrics_df["PPV (Precision)"]
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.plot(recall, precision, marker="o")
    if show_selected:
        selected_idx = metrics_df["Threshold"] == selected_threshold
        ax.plot(
            metrics_df["Sensitivity (Recall)"][selected_idx],
            metrics_df["PPV (Precision)"][selected_idx],
            'ro', label="Selected Threshold"
        )
    # Random classifier line
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, label="Random Classifier")
    ax.set_xlabel("Recall", fontsize=6)
    ax.set_ylabel("Precision", fontsize=6)
    ax.set_title("Precision-Recall Curve", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True)
    ax.legend(fontsize=6)
    return fig

def plot_prediction_bar(*args, **kwargs):
    """
    Plot a vertical bar showing prediction prevalence zones.
    If called with a single row (legacy), plot LOW/HIGH only.
    If called with two values (pred_prev_t1, pred_prev_t2), plot LOW/MODERATE/HIGH.
    """
    if len(args) == 1 and isinstance(args[0], (dict, pd.Series)):
        # Legacy: single row, two zones (LOW/HIGH)
        row = args[0]
        pred_prev = row["Prediction Prevalence"]
        fig, ax = plt.subplots(figsize=(0.6, 0.9))
        # Reverse color assignments: LOW is green, HIGH is #800020
        ax.bar(0, 1 - pred_prev, color='green', width=0.5)
        ax.bar(0, pred_prev, bottom=1 - pred_prev, color='#800020', width=0.5)
        low_y = (1 - pred_prev) / 2
        ax.text(0, low_y, "LOW", ha='center', va='center', fontsize=5, color='white')
        ax.text(0.6, low_y, f"{(1 - pred_prev)*100:.1f}%", ha='left', va='center', fontsize=5, color='green')
        high_y = 1 - (pred_prev / 2)
        ax.text(0, high_y, "HIGH", ha='center', va='center', fontsize=5, color='white')
        ax.text(0.6, high_y, f"{pred_prev*100:.1f}%", ha='left', va='center', fontsize=5, color='#800020')
        ax.set_xlim(-0.5, 0.8)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return fig
    elif len(args) == 2:
        # Dual threshold, three zones (LOW/MODERATE/HIGH)
        pred_prev_t1, pred_prev_t2 = args
        fig, ax = plt.subplots(figsize=(0.6, 0.9))
        # Calculate zone heights
        low_val = 1 - pred_prev_t1
        mod_val = pred_prev_t1 - pred_prev_t2
        high_val = pred_prev_t2
        # Reverse colors: LOW is green, MODERATE gold, HIGH is #800020
        ax.bar(0, low_val, color='green', width=0.5, label="LOW")
        ax.bar(0, mod_val, bottom=low_val, color='gold', width=0.5, label="MODERATE")
        ax.bar(0, high_val, bottom=low_val + mod_val, color='#800020', width=0.5, label="HIGH")
        # LOW label
        low_y = low_val / 2
        ax.text(0, low_y, "LOW", ha='center', va='center', fontsize=5, color='white')
        ax.text(0.6, low_y, f"{low_val*100:.1f}%", ha='left', va='center', fontsize=5, color='green')
        # MODERATE label
        mod_y = low_val + (mod_val / 2)
        ax.text(0, mod_y, "MODERATE", ha='center', va='center', fontsize=5, color='black')
        ax.text(0.6, mod_y, f"{mod_val*100:.1f}%", ha='left', va='center', fontsize=5, color='gold')
        # HIGH label
        high_y = low_val + mod_val + (high_val / 2)
        ax.text(0, high_y, "HIGH", ha='center', va='center', fontsize=5, color='white')
        ax.text(0.6, high_y, f"{high_val*100:.1f}%", ha='left', va='center', fontsize=5, color='#800020')
        ax.set_xlim(-0.5, 0.8)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return fig
    else:
        raise ValueError("Invalid arguments for plot_prediction_bar")

# ----------------------------
# Streamlit App Layout
# ----------------------------

st.set_page_config(page_title="Postoperative Predictive Performance Visualization", layout="wide")
st.title("Postoperative Predictive Performance Visualization")

# Create tabs
tab_news, tab_gbs, tab_qsofa, tab_news_2t, tab_gbs_2t, tab_qsofa_2t = st.tabs([
    "NEWS (1T)", "GBS (1T)", "qSOFA (1T)", "NEWS (2T)", "GBS (2T)", "qSOFA (2T)"
])
with tab_qsofa_2t:
    st.subheader("Model Output: Postoperative Sepsis (Dual Thresholds)")
    try:
        df = load_confusion_data("qsofa_scores.csv")
        metrics_df = compute_all_metrics(df)

        # --- Dual Threshold selection (Box 1) ---
        st.markdown("### Dual Threshold Selection")
        with st.container(border=True):
            st.markdown("**What are dual thresholds?**  \nUsing two thresholds allows you to define three prediction zones: LOW (score < T1), MODERATE (T1 ≤ score < T2), and HIGH (score ≥ T2). This can help with risk stratification and clinical decision-making.")
            threshold_values = df["Threshold"].sort_values().unique().tolist()
            step_val = float(threshold_values[1] - threshold_values[0]) if len(threshold_values) > 1 else 0.01

            # Use session state to store t1 and t2 across reruns
            if "t1_qsofa2t" not in st.session_state:
                st.session_state.t1_qsofa2t = float(threshold_values[0])
            if "t2_qsofa2t" not in st.session_state:
                st.session_state.t2_qsofa2t = float(threshold_values[-1])

            t1 = st.slider(
                "Choose T1 (Lower Threshold)",
                min_value=float(min(threshold_values)),
                max_value=float(max(threshold_values)),
                value=st.session_state.t1_qsofa2t,
                step=step_val,
                key="threshold_qsofa2t_t1"
            )

            # Adjust t2 only if it's less than t1
            if st.session_state.t2_qsofa2t < t1:
                st.session_state.t2_qsofa2t = t1

            t2 = st.slider(
                "Choose T2 (Upper Threshold)",
                min_value=float(t1),
                max_value=float(max(threshold_values)),
                value=st.session_state.t2_qsofa2t,
                step=step_val,
                key="threshold_qsofa2t_t2"
            )

            # Update session state
            st.session_state.t1_qsofa2t = t1
            st.session_state.t2_qsofa2t = t2

        # --- Metrics and plots (Box 2) ---
        row_t1 = metrics_df[metrics_df["Threshold"] == t1].iloc[0]
        row_t2 = metrics_df[metrics_df["Threshold"] == t2].iloc[0]
        col_spacer1, col_metrics, col_roc, col_pr, col_prev, col_spacer2 = st.columns([0.05, 0.35, 0.2, 0.2, 0.2, 0.05])
        with col_metrics:
            st.markdown("**Metrics at T1 and T2**")
            metrics_table = pd.DataFrame({
                "Metric": metrics_df.columns.tolist()[1:],
                "T1 Value": [row_t1[m] for m in metrics_df.columns if m != "Threshold"],
                "T2 Value": [row_t2[m] for m in metrics_df.columns if m != "Threshold"],
            })
            st.dataframe(metrics_table, use_container_width=True, hide_index=True, height=270)

        with col_roc:
            fig_roc = plot_roc_curve(metrics_df, t1, show_selected=False)
            idx_t2 = metrics_df["Threshold"] == t2
            ax_roc = fig_roc.axes[0]
            # Overlay T1 and T2 points, with correct labeling
            ax_roc.plot(
                (1 - row_t1["Specificity"]),
                row_t1["Sensitivity (Recall)"],
                marker='o', color='#FF7F0E', markersize=5, label="T1"
            )
            ax_roc.plot(
                (1 - row_t2["Specificity"]),
                row_t2["Sensitivity (Recall)"],
                marker='o', color='#D62728', markersize=5, label="T2"
            )
            ax_roc.legend(fontsize=5)
            st.pyplot(fig_roc, use_container_width=True)

        with col_pr:
            fig_pr = plot_pr_curve(metrics_df, t1, df, show_selected=False)
            ax_pr = fig_pr.axes[0]
            ax_pr.plot(
                row_t1["Sensitivity (Recall)"],
                row_t1["PPV (Precision)"],
                marker='o', color='#FF7F0E', markersize=5, label="T1"
            )
            ax_pr.plot(
                row_t2["Sensitivity (Recall)"],
                row_t2["PPV (Precision)"],
                marker='o', color='#D62728', markersize=5, label="T2"
            )
            ax_pr.legend(fontsize=5)
            st.pyplot(fig_pr, use_container_width=True)

        # Define prediction prevalence for T1 and T2 for the bar plot
        pred_prev_t1 = row_t1["Prediction Prevalence"]
        pred_prev_t2 = row_t2["Prediction Prevalence"]
        with col_prev:
            fig_prev = plot_prediction_bar(pred_prev_t1, pred_prev_t2)
            st.pyplot(fig_prev, use_container_width=True)

        # --- Box 2.5: Case Study on Performance (Three Zones) ---
        st.markdown("### Case Study on Performance: Three Risk Zones")
        st.markdown("It is the first morning after primary bariatric surgery. Please consider the following model behaviour. For 1000 patients, how are they classified into LOW, MODERATE, and HIGH risk groups?")

        sample_size = 1000
        pred_prev_t1 = row_t1["Prediction Prevalence"]
        pred_prev_t2 = row_t2["Prediction Prevalence"]
        # Prevalence for each zone
        p_low = 1 - pred_prev_t1
        p_mod = pred_prev_t1 - pred_prev_t2
        p_high = pred_prev_t2
        # For each zone, estimate confusion matrix using differences
        label_prev = row_t1["Label Prevalence"]
        # For LOW zone: below T1, use T1 TN/FN rates
        tn_low = row_t1["Specificity"] * sample_size * (1 - label_prev) * p_low / (p_low + p_mod + p_high) if (p_low + p_mod + p_high) > 0 else 0
        fn_low = (1 - row_t1["Sensitivity (Recall)"]) * sample_size * label_prev * p_low / (p_low + p_mod + p_high) if (p_low + p_mod + p_high) > 0 else 0
        fp_low = 0
        tp_low = 0
        # For MODERATE zone: between T1 and T2, use difference between T1 and T2
        # Compute confusion matrix entries for MODERATE as difference
        tn_mod = (df.loc[df["Threshold"] == t2, "TN"].values[0] - df.loc[df["Threshold"] == t1, "TN"].values[0]) * sample_size / df["TN"].max() if df["TN"].max() > 0 else 0
        fn_mod = (df.loc[df["Threshold"] == t2, "FN"].values[0] - df.loc[df["Threshold"] == t1, "FN"].values[0]) * sample_size / df["FN"].max() if df["FN"].max() > 0 else 0
        fp_mod = (df.loc[df["Threshold"] == t2, "FP"].values[0] - df.loc[df["Threshold"] == t1, "FP"].values[0]) * sample_size / df["FP"].max() if df["FP"].max() > 0 else 0
        tp_mod = (df.loc[df["Threshold"] == t2, "TP"].values[0] - df.loc[df["Threshold"] == t1, "TP"].values[0]) * sample_size / df["TP"].max() if df["TP"].max() > 0 else 0
        # For HIGH zone: above T2, use T2 TP/FP rates
        tn_high = 0
        fn_high = 0
        fp_high = (1 - row_t2["Specificity"]) * sample_size * (1 - row_t2["Label Prevalence"]) * p_high / (p_low + p_mod + p_high) if (p_low + p_mod + p_high) > 0 else 0
        tp_high = row_t2["Sensitivity (Recall)"] * sample_size * row_t2["Label Prevalence"] * p_high / (p_low + p_mod + p_high) if (p_low + p_mod + p_high) > 0 else 0

        def figure_block(label, count, color):
            people = "👤" * min(int(count), 100)
            return f"**{label} (N={int(round(count))})**\n\n{people}\n\n"

        col_low, col_mod, col_high = st.columns(3)
        label_term = "sepsis"
        # LOW
        with col_low:
            st.markdown(f"#### Model Labels as <span style='color:green;'>LOW</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", tn_low, "green"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", fn_low, "green"), unsafe_allow_html=True)
        # MODERATE
        with col_mod:
            st.markdown(f"#### Model Labels as <span style='color:gold;'>MODERATE</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", tn_mod, "gold"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", fn_mod, "gold"), unsafe_allow_html=True)
        # HIGH
        with col_high:
            st.markdown(f"#### Model Labels as <span style='color:#800020;'>HIGH</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", fp_high, "#800020"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", tp_high, "#800020"), unsafe_allow_html=True)

        # --- Box 3: About Sepsis ---
        st.markdown("### About Postoperative Sepsis")
        st.markdown("""
Sepsis is a life-threatening response to infection that can occur after surgery.
Early identification using predictive models is critical to initiate timely treatment.

This model uses physiological and lab data to anticipate sepsis onset based on trends in early recovery.
""")

    except FileNotFoundError:
        st.error("Missing file: ./data/qsofa_scores.csv")
# NEWS (2T) Tab
with tab_news_2t:
    st.subheader("Model Output: Postoperative Sepsis (Dual Thresholds)")
    try:
        df = load_confusion_data("news_scores.csv")
        metrics_df = compute_all_metrics(df)

        # --- Dual Threshold selection (Box 1) ---
        st.markdown("### Dual Threshold Selection")
        with st.container(border=True):
            st.markdown("**What are dual thresholds?**  \nUsing two thresholds allows you to define three prediction zones: LOW (score < T1), MODERATE (T1 ≤ score < T2), and HIGH (score ≥ T2). This can help with risk stratification and clinical decision-making.")
            threshold_values = df["Threshold"].sort_values().unique().tolist()
            step_val = float(threshold_values[1] - threshold_values[0]) if len(threshold_values) > 1 else 0.01

            # Use session state to store t1 and t2 across reruns
            if "t1_news2t" not in st.session_state:
                st.session_state.t1_news2t = float(threshold_values[0])
            if "t2_news2t" not in st.session_state:
                st.session_state.t2_news2t = float(threshold_values[-1])

            t1 = st.slider(
                "Choose T1 (Lower Threshold)",
                min_value=float(min(threshold_values)),
                max_value=float(max(threshold_values)),
                value=st.session_state.t1_news2t,
                step=step_val,
                key="threshold_news2t_t1"
            )

            # Adjust t2 only if it's less than t1
            if st.session_state.t2_news2t < t1:
                st.session_state.t2_news2t = t1

            t2 = st.slider(
                "Choose T2 (Upper Threshold)",
                min_value=float(t1),
                max_value=float(max(threshold_values)),
                value=st.session_state.t2_news2t,
                step=step_val,
                key="threshold_news2t_t2"
            )

            # Update session state
            st.session_state.t1_news2t = t1
            st.session_state.t2_news2t = t2

        # --- Metrics and plots (Box 2) ---
        row_t1 = metrics_df[metrics_df["Threshold"] == t1].iloc[0]
        row_t2 = metrics_df[metrics_df["Threshold"] == t2].iloc[0]
        col_spacer1, col_metrics, col_roc, col_pr, col_prev, col_spacer2 = st.columns([0.05, 0.35, 0.2, 0.2, 0.2, 0.05])
        with col_metrics:
            st.markdown("**Metrics at T1 and T2**")
            metrics_table = pd.DataFrame({
                "Metric": metrics_df.columns.tolist()[1:],
                "T1 Value": [row_t1[m] for m in metrics_df.columns if m != "Threshold"],
                "T2 Value": [row_t2[m] for m in metrics_df.columns if m != "Threshold"],
            })
            st.dataframe(metrics_table, use_container_width=True, hide_index=True, height=270)

        with col_roc:
            fig_roc = plot_roc_curve(metrics_df, t1, show_selected=False)
            idx_t2 = metrics_df["Threshold"] == t2
            ax_roc = fig_roc.axes[0]
            # Overlay T1 and T2 points, with correct labeling
            ax_roc.plot(
                (1 - row_t1["Specificity"]),
                row_t1["Sensitivity (Recall)"],
                marker='o', color='#FF7F0E', markersize=5, label="T1"
            )
            ax_roc.plot(
                (1 - row_t2["Specificity"]),
                row_t2["Sensitivity (Recall)"],
                marker='o', color='#D62728', markersize=5, label="T2"
            )
            ax_roc.legend(fontsize=5)
            st.pyplot(fig_roc, use_container_width=True)

        with col_pr:
            fig_pr = plot_pr_curve(metrics_df, t1, df, show_selected=False)
            ax_pr = fig_pr.axes[0]
            ax_pr.plot(
                row_t1["Sensitivity (Recall)"],
                row_t1["PPV (Precision)"],
                marker='o', color='#FF7F0E', markersize=5, label="T1"
            )
            ax_pr.plot(
                row_t2["Sensitivity (Recall)"],
                row_t2["PPV (Precision)"],
                marker='o', color='#D62728', markersize=5, label="T2"
            )
            ax_pr.legend(fontsize=5)
            st.pyplot(fig_pr, use_container_width=True)

        # Define prediction prevalence for T1 and T2 for the bar plot
        pred_prev_t1 = row_t1["Prediction Prevalence"]
        pred_prev_t2 = row_t2["Prediction Prevalence"]
        with col_prev:
            fig_prev = plot_prediction_bar(pred_prev_t1, pred_prev_t2)
            st.pyplot(fig_prev, use_container_width=True)

        # --- Box 2.5: Case Study on Performance (Three Zones) ---
        st.markdown("### Case Study on Performance: Three Risk Zones")
        st.markdown("It is the first morning after primary bariatric surgery. Please consider the following model behaviour. For 1000 patients, how are they classified into LOW, MODERATE, and HIGH risk groups?")

        sample_size = 1000
        pred_prev_t1 = row_t1["Prediction Prevalence"]
        pred_prev_t2 = row_t2["Prediction Prevalence"]
        # Prevalence for each zone
        p_low = 1 - pred_prev_t1
        p_mod = pred_prev_t1 - pred_prev_t2
        p_high = pred_prev_t2
        # For each zone, estimate confusion matrix using differences
        label_prev = row_t1["Label Prevalence"]
        # For LOW zone: below T1, use T1 TN/FN rates
        tn_low = row_t1["Specificity"] * sample_size * (1 - label_prev) * p_low / (p_low + p_mod + p_high) if (p_low + p_mod + p_high) > 0 else 0
        fn_low = (1 - row_t1["Sensitivity (Recall)"]) * sample_size * label_prev * p_low / (p_low + p_mod + p_high) if (p_low + p_mod + p_high) > 0 else 0
        fp_low = 0
        tp_low = 0
        # For MODERATE zone: between T1 and T2, use difference between T1 and T2
        # Compute confusion matrix entries for MODERATE as difference
        tn_mod = (df.loc[df["Threshold"] == t2, "TN"].values[0] - df.loc[df["Threshold"] == t1, "TN"].values[0]) * sample_size / df["TN"].max() if df["TN"].max() > 0 else 0
        fn_mod = (df.loc[df["Threshold"] == t2, "FN"].values[0] - df.loc[df["Threshold"] == t1, "FN"].values[0]) * sample_size / df["FN"].max() if df["FN"].max() > 0 else 0
        fp_mod = (df.loc[df["Threshold"] == t2, "FP"].values[0] - df.loc[df["Threshold"] == t1, "FP"].values[0]) * sample_size / df["FP"].max() if df["FP"].max() > 0 else 0
        tp_mod = (df.loc[df["Threshold"] == t2, "TP"].values[0] - df.loc[df["Threshold"] == t1, "TP"].values[0]) * sample_size / df["TP"].max() if df["TP"].max() > 0 else 0
        # For HIGH zone: above T2, use T2 TP/FP rates
        tn_high = 0
        fn_high = 0
        fp_high = (1 - row_t2["Specificity"]) * sample_size * (1 - row_t2["Label Prevalence"]) * p_high / (p_low + p_mod + p_high) if (p_low + p_mod + p_high) > 0 else 0
        tp_high = row_t2["Sensitivity (Recall)"] * sample_size * row_t2["Label Prevalence"] * p_high / (p_low + p_mod + p_high) if (p_low + p_mod + p_high) > 0 else 0

        def figure_block(label, count, color):
            people = "👤" * min(int(count), 100)
            return f"**{label} (N={int(round(count))})**\n\n{people}\n\n"

        col_low, col_mod, col_high = st.columns(3)
        label_term = "sepsis"
        # LOW
        with col_low:
            st.markdown(f"#### Model Labels as <span style='color:green;'>LOW</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", tn_low, "green"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", fn_low, "green"), unsafe_allow_html=True)
        # MODERATE
        with col_mod:
            st.markdown(f"#### Model Labels as <span style='color:gold;'>MODERATE</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", tn_mod, "gold"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", fn_mod, "gold"), unsafe_allow_html=True)
        # HIGH
        with col_high:
            st.markdown(f"#### Model Labels as <span style='color:#800020;'>HIGH</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", fp_high, "#800020"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", tp_high, "#800020"), unsafe_allow_html=True)

        # --- Box 3: About Sepsis ---
        st.markdown("### About Postoperative Sepsis")
        st.markdown("""
Sepsis is a life-threatening response to infection that can occur after surgery.
Early identification using predictive models is critical to initiate timely treatment.

This model uses physiological and lab data to anticipate sepsis onset based on trends in early recovery.
""")

    except FileNotFoundError:
        st.error("Missing file: ./data/news_scores.csv")

# NEWS Tab
with tab_news:
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
                step=step_val,
                key="threshold_news"
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
            fig_roc = plot_roc_curve(metrics_df, selected_threshold, show_selected=True)
            fig_pr = plot_pr_curve(metrics_df, selected_threshold, df, show_selected=True)
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig_roc)
            with col2:
                st.pyplot(fig_pr)

        with col_r:
            fig_prev = plot_prediction_bar(selected_row)
            st.pyplot(fig_prev)

        # --- Box 2.5: Case Study on Performance ---
        st.markdown("### Case Study on Performance")
        st.markdown("It is the first morning after primary bariatric surgery. Please consider the following model behaviour. For 1000 patients...")

        sample_size = 1000
        tp = selected_row["Sensitivity (Recall)"] * sample_size * selected_row["Label Prevalence"]
        fn = (1 - selected_row["Sensitivity (Recall)"]) * sample_size * selected_row["Label Prevalence"]
        fp = (1 - selected_row["Specificity"]) * sample_size * (1 - selected_row["Label Prevalence"])
        tn = selected_row["Specificity"] * sample_size * (1 - selected_row["Label Prevalence"])

        def figure_block(label, count, color):
            people = "👤" * min(int(count), 100)
            return f"**{label} (N={int(round(count))})**\n\n{people}\n\n"

        col_low, col_high = st.columns(2)

        label_term = "sepsis"

        with col_low:
            st.markdown(f"#### Model Labels as <span style='color:green;'>LOW</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", tn, "green"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", fn, "green"), unsafe_allow_html=True)

        with col_high:
            st.markdown(f"#### Model Labels as <span style='color:#800020;'>HIGH</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", fp, "#800020"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", tp, "#800020"), unsafe_allow_html=True)

        # --- Box 3: About Sepsis ---
        st.markdown("### About Postoperative Sepsis")
        st.markdown("""
Sepsis is a life-threatening response to infection that can occur after surgery.
Early identification using predictive models is critical to initiate timely treatment.

This model uses physiological and lab data to anticipate sepsis onset based on trends in early recovery.
""")

    except FileNotFoundError:
        st.error("Missing file: ./data/news_scores.csv")
# qSOFA Tab (duplicate of NEWS, modified)
with tab_qsofa:
    st.subheader("Model Output: Postoperative Sepsis")
    try:
        df = load_confusion_data("qsofa_scores.csv")
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
                step=step_val,
                key="threshold_qsofa"
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
            fig_roc = plot_roc_curve(metrics_df, selected_threshold, show_selected=True)
            fig_pr = plot_pr_curve(metrics_df, selected_threshold, df, show_selected=True)
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig_roc)
            with col2:
                st.pyplot(fig_pr)

        with col_r:
            fig_prev = plot_prediction_bar(selected_row)
            st.pyplot(fig_prev)

        # --- Box 2.5: Case Study on Performance ---
        st.markdown("### Case Study on Performance")
        st.markdown("It is the first morning after primary bariatric surgery. Please consider the following model behaviour. For 1000 patients...")

        sample_size = 1000
        tp = selected_row["Sensitivity (Recall)"] * sample_size * selected_row["Label Prevalence"]
        fn = (1 - selected_row["Sensitivity (Recall)"]) * sample_size * selected_row["Label Prevalence"]
        fp = (1 - selected_row["Specificity"]) * sample_size * (1 - selected_row["Label Prevalence"])
        tn = selected_row["Specificity"] * sample_size * (1 - selected_row["Label Prevalence"])

        def figure_block(label, count, color):
            people = "👤" * min(int(count), 100)
            return f"**{label} (N={int(round(count))})**\n\n{people}\n\n"

        col_low, col_high = st.columns(2)

        label_term = "sepsis"

        with col_low:
            st.markdown(f"#### Model Labels as <span style='color:green;'>LOW</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", tn, "green"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", fn, "green"), unsafe_allow_html=True)

        with col_high:
            st.markdown(f"#### Model Labels as <span style='color:#800020;'>HIGH</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", fp, "#800020"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", tp, "#800020"), unsafe_allow_html=True)

        # --- Box 3: About Sepsis ---
        st.markdown("### About Postoperative Sepsis")
        st.markdown("""
Sepsis is a life-threatening response to infection that can occur after surgery.
Early identification using predictive models is critical to initiate timely treatment.

This model uses physiological and lab data to anticipate sepsis onset based on trends in early recovery.
""")

    except FileNotFoundError:
        st.error("Missing file: ./data/qsofa_scores.csv")

# GBS Tab
with tab_gbs:
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
                step=step_val,
                key="threshold_gbs"
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
            fig_roc = plot_roc_curve(metrics_df, selected_threshold, show_selected=True)
            fig_pr = plot_pr_curve(metrics_df, selected_threshold, df, show_selected=True)
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig_roc)
            with col2:
                st.pyplot(fig_pr)

        with col_r:
            fig_prev = plot_prediction_bar(selected_row)
            st.pyplot(fig_prev)

        # --- Box 2.5: Case Study on Performance ---
        st.markdown("### Case Study on Performance")
        st.markdown("It is the first morning after primary bariatric surgery. Please consider the following model behaviour. For 1000 patients...")

        sample_size = 1000
        tp = selected_row["Sensitivity (Recall)"] * sample_size * selected_row["Label Prevalence"]
        fn = (1 - selected_row["Sensitivity (Recall)"]) * sample_size * selected_row["Label Prevalence"]
        fp = (1 - selected_row["Specificity"]) * sample_size * (1 - selected_row["Label Prevalence"])
        tn = selected_row["Specificity"] * sample_size * (1 - selected_row["Label Prevalence"])

        def figure_block(label, count, color):
            people = "👤" * min(int(count), 100)
            return f"**{label} (N={int(round(count))})**\n\n{people}\n\n"

        col_low, col_high = st.columns(2)

        label_term = "bleeding"

        with col_low:
            st.markdown(f"#### Model Labels as <span style='color:green;'>LOW</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", tn, "green"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", fn, "green"), unsafe_allow_html=True)

        with col_high:
            st.markdown(f"#### Model Labels as <span style='color:#800020;'>HIGH</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", fp, "#800020"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", tp, "#800020"), unsafe_allow_html=True)

        # --- Box 3: About Bleeding ---
        st.markdown("### About Postoperative Bleeding")
        st.markdown("""
Postoperative bleeding is a serious complication that can lead to reoperation, transfusion, or longer ICU stays.
Accurate early prediction helps clinicians intervene proactively.

This model leverages vital signs and clinical scores to predict bleeding risk after surgery.
""")

    except FileNotFoundError:
        st.error("Missing file: ./data/gbs_scores.csv")


# GBS (2T) Tab
with tab_gbs_2t:
    st.subheader("Model Output: Postoperative Bleeding (Dual Thresholds)")
    try:
        df = load_confusion_data("gbs_scores.csv")
        metrics_df = compute_all_metrics(df)

        # --- Dual Threshold selection (Box 1) ---
        st.markdown("### Dual Threshold Selection")
        with st.container(border=True):
            st.markdown("**What are dual thresholds?**  \nUsing two thresholds allows you to define three prediction zones: LOW (score < T1), MODERATE (T1 ≤ score < T2), and HIGH (score ≥ T2). This can help with risk stratification and clinical decision-making.")
            threshold_values = df["Threshold"].sort_values().unique().tolist()
            step_val = float(threshold_values[1] - threshold_values[0]) if len(threshold_values) > 1 else 0.01

            # Use session state to store t1 and t2 across reruns
            if "t1_gbs2t" not in st.session_state:
                st.session_state.t1_gbs2t = float(threshold_values[0])
            if "t2_gbs2t" not in st.session_state:
                st.session_state.t2_gbs2t = float(threshold_values[-1])

            t1 = st.slider(
                "Choose T1 (Lower Threshold)",
                min_value=float(min(threshold_values)),
                max_value=float(max(threshold_values)),
                value=st.session_state.t1_gbs2t,
                step=step_val,
                key="threshold_gbs2t_t1"
            )

            # Adjust t2 only if it's less than t1
            if st.session_state.t2_gbs2t < t1:
                st.session_state.t2_gbs2t = t1

            t2 = st.slider(
                "Choose T2 (Upper Threshold)",
                min_value=float(t1),
                max_value=float(max(threshold_values)),
                value=st.session_state.t2_gbs2t,
                step=step_val,
                key="threshold_gbs2t_t2"
            )

            # Update session state
            st.session_state.t1_gbs2t = t1
            st.session_state.t2_gbs2t = t2

        # --- Metrics and plots (Box 2) ---
        row_t1 = metrics_df[metrics_df["Threshold"] == t1].iloc[0]
        row_t2 = metrics_df[metrics_df["Threshold"] == t2].iloc[0]
        col_spacer1, col_metrics, col_roc, col_pr, col_prev, col_spacer2 = st.columns([0.05, 0.35, 0.2, 0.2, 0.2, 0.05])
        with col_metrics:
            st.markdown("**Metrics at T1 and T2**")
            metrics_table = pd.DataFrame({
                "Metric": metrics_df.columns.tolist()[1:],
                "T1 Value": [row_t1[m] for m in metrics_df.columns if m != "Threshold"],
                "T2 Value": [row_t2[m] for m in metrics_df.columns if m != "Threshold"],
            })
            st.dataframe(metrics_table, use_container_width=True, hide_index=True, height=270)

        with col_roc:
            fig_roc = plot_roc_curve(metrics_df, t1, show_selected=False)
            idx_t2 = metrics_df["Threshold"] == t2
            ax_roc = fig_roc.axes[0]
            # Overlay T1 and T2 points, with correct labeling
            ax_roc.plot(
                (1 - row_t1["Specificity"]),
                row_t1["Sensitivity (Recall)"],
                marker='o', color='#FF7F0E', markersize=5, label="T1"
            )
            ax_roc.plot(
                (1 - row_t2["Specificity"]),
                row_t2["Sensitivity (Recall)"],
                marker='o', color='#D62728', markersize=5, label="T2"
            )
            ax_roc.legend(fontsize=5)
            st.pyplot(fig_roc, use_container_width=True)

        with col_pr:
            fig_pr = plot_pr_curve(metrics_df, t1, df, show_selected=False)
            ax_pr = fig_pr.axes[0]
            ax_pr.plot(
                row_t1["Sensitivity (Recall)"],
                row_t1["PPV (Precision)"],
                marker='o', color='#FF7F0E', markersize=5, label="T1"
            )
            ax_pr.plot(
                row_t2["Sensitivity (Recall)"],
                row_t2["PPV (Precision)"],
                marker='o', color='#D62728', markersize=5, label="T2"
            )
            ax_pr.legend(fontsize=5)
            st.pyplot(fig_pr, use_container_width=True)

        # Define prediction prevalence for T1 and T2 for the bar plot
        pred_prev_t1 = row_t1["Prediction Prevalence"]
        pred_prev_t2 = row_t2["Prediction Prevalence"]
        with col_prev:
            fig_prev = plot_prediction_bar(pred_prev_t1, pred_prev_t2)
            st.pyplot(fig_prev, use_container_width=True)

        # --- Box 2.5: Case Study on Performance (Three Zones) ---
        st.markdown("### Case Study on Performance: Three Risk Zones")
        st.markdown("It is the first morning after primary bariatric surgery. Please consider the following model behaviour. For 1000 patients, how are they classified into LOW, MODERATE, and HIGH risk groups?")

        sample_size = 1000
        pred_prev_t1 = row_t1["Prediction Prevalence"]
        pred_prev_t2 = row_t2["Prediction Prevalence"]
        # Prevalence for each zone
        p_low = 1 - pred_prev_t1
        p_mod = pred_prev_t1 - pred_prev_t2
        p_high = pred_prev_t2
        # For each zone, estimate confusion matrix using differences
        label_prev = row_t1["Label Prevalence"]
        # For LOW zone: below T1, use T1 TN/FN rates
        tn_low = row_t1["Specificity"] * sample_size * (1 - label_prev) * p_low / (p_low + p_mod + p_high) if (p_low + p_mod + p_high) > 0 else 0
        fn_low = (1 - row_t1["Sensitivity (Recall)"]) * sample_size * label_prev * p_low / (p_low + p_mod + p_high) if (p_low + p_mod + p_high) > 0 else 0
        fp_low = 0
        tp_low = 0
        # For MODERATE zone: between T1 and T2, use difference between T1 and T2
        # Compute confusion matrix entries for MODERATE as difference
        tn_mod = (df.loc[df["Threshold"] == t2, "TN"].values[0] - df.loc[df["Threshold"] == t1, "TN"].values[0]) * sample_size / df["TN"].max() if df["TN"].max() > 0 else 0
        fn_mod = (df.loc[df["Threshold"] == t2, "FN"].values[0] - df.loc[df["Threshold"] == t1, "FN"].values[0]) * sample_size / df["FN"].max() if df["FN"].max() > 0 else 0
        fp_mod = (df.loc[df["Threshold"] == t2, "FP"].values[0] - df.loc[df["Threshold"] == t1, "FP"].values[0]) * sample_size / df["FP"].max() if df["FP"].max() > 0 else 0
        tp_mod = (df.loc[df["Threshold"] == t2, "TP"].values[0] - df.loc[df["Threshold"] == t1, "TP"].values[0]) * sample_size / df["TP"].max() if df["TP"].max() > 0 else 0
        # For HIGH zone: above T2, use T2 TP/FP rates
        tn_high = 0
        fn_high = 0
        fp_high = (1 - row_t2["Specificity"]) * sample_size * (1 - row_t2["Label Prevalence"]) * p_high / (p_low + p_mod + p_high) if (p_low + p_mod + p_high) > 0 else 0
        tp_high = row_t2["Sensitivity (Recall)"] * sample_size * row_t2["Label Prevalence"] * p_high / (p_low + p_mod + p_high) if (p_low + p_mod + p_high) > 0 else 0

        def figure_block(label, count, color):
            people = "👤" * min(int(count), 100)
            return f"**{label} (N={int(round(count))})**\n\n{people}\n\n"

        col_low, col_mod, col_high = st.columns(3)
        label_term = "bleeding"
        # LOW
        with col_low:
            st.markdown(f"#### Model Labels as <span style='color:green;'>LOW</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", tn_low, "green"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", fn_low, "green"), unsafe_allow_html=True)
        # MODERATE
        with col_mod:
            st.markdown(f"#### Model Labels as <span style='color:gold;'>MODERATE</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", tn_mod, "gold"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", fn_mod, "gold"), unsafe_allow_html=True)
        # HIGH
        with col_high:
            st.markdown(f"#### Model Labels as <span style='color:#800020;'>HIGH</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", fp_high, "#800020"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", tp_high, "#800020"), unsafe_allow_html=True)

        # --- Box 3: About Bleeding ---
        st.markdown("### About Postoperative Bleeding")
        st.markdown("""
Postoperative bleeding is a serious complication that can lead to reoperation, transfusion, or longer ICU stays.
Accurate early prediction helps clinicians intervene proactively.

This model leverages vital signs and clinical scores to predict bleeding risk after surgery.
""")

    except FileNotFoundError:
        st.error("Missing file: ./data/gbs_scores.csv")