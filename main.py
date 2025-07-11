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
    # Random classifier line
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, label="Random Classifier")
    ax.set_xlabel("Recall", fontsize=6)
    ax.set_ylabel("Precision", fontsize=6)
    ax.set_title("Precision-Recall Curve", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True)
    ax.legend(fontsize=6)
    return fig

def plot_prediction_bar(row):
    pred_prev = row["Prediction Prevalence"]
    fig, ax = plt.subplots(figsize=(0.6, 0.9))
    # Draw burgundy LOW on bottom, green HIGH on top
    ax.bar(0, 1 - pred_prev, color='#800020', width=0.5)
    ax.bar(0, pred_prev, bottom=1 - pred_prev, color='green', width=0.5)
    # LOW label and percent (bottom)
    low_y = (1 - pred_prev) / 2
    ax.text(0, low_y, "LOW", ha='center', va='center', fontsize=5, color='white')
    ax.text(0.6, low_y, f"{(1 - pred_prev)*100:.1f}%", ha='left', va='center', fontsize=5, color='#800020')
    # HIGH label and percent (top)
    high_y = 1 - (pred_prev / 2)
    ax.text(0, high_y, "HIGH", ha='center', va='center', fontsize=5, color='white')
    ax.text(0.6, high_y, f"{pred_prev*100:.1f}%", ha='left', va='center', fontsize=5, color='green')
    ax.set_xlim(-0.5, 0.8)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig

# ----------------------------
# Streamlit App Layout
# ----------------------------

st.set_page_config(page_title="Postoperative Predictive Performance Visualization", layout="wide")
st.title("Postoperative Predictive Performance Visualization")

# Create tabs
tab_news, tab_gbs, tab_news_custom, tab_gbs_custom, tab_qsofa, tab_news_2t = st.tabs([
    "NEWS (1T)", "GBS (1T)", "Sepsis Custom (1T)", "Bleeding Custom (1T)", "qSOFA (1T)", "NEWS (2T)"
])
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
            t1 = st.slider(
                "Choose T1 (Lower Threshold)",
                min_value=float(min(threshold_values)),
                max_value=float(max(threshold_values)),
                value=float(threshold_values[0]),
                step=step_val,
                key="threshold_news2t_t1"
            )
            t2 = st.slider(
                "Choose T2 (Upper Threshold)",
                min_value=float(t1),
                max_value=float(max(threshold_values)),
                value=max(float(t1), float(threshold_values[-1])),
                step=step_val,
                key="threshold_news2t_t2"
            )

        # --- Metrics and plots (Box 2) ---
        row_t1 = metrics_df[metrics_df["Threshold"] == t1].iloc[0]
        row_t2 = metrics_df[metrics_df["Threshold"] == t2].iloc[0]
        col_l, col_m, col_r = st.columns([0.3, 0.4, 0.3])
        with col_l:
            st.markdown("**Metrics at T1 (Lower Threshold)**")
            metrics_table1 = pd.DataFrame([row_t1]).T.reset_index()
            metrics_table1.columns = ["Metric", "Value"]
            st.dataframe(metrics_table1, use_container_width=True, hide_index=True)
        with col_m:
            st.markdown("**Metrics at T2 (Upper Threshold)**")
            metrics_table2 = pd.DataFrame([row_t2]).T.reset_index()
            metrics_table2.columns = ["Metric", "Value"]
            st.dataframe(metrics_table2, use_container_width=True, hide_index=True)
        with col_r:
            # Custom prevalence bar showing three segments: LOW, MODERATE, HIGH
            pred_prev_t1 = row_t1["Prediction Prevalence"]
            pred_prev_t2 = row_t2["Prediction Prevalence"]
            # LOW = 1 - pred_prev_t1, MODERATE = pred_prev_t1 - pred_prev_t2, HIGH = pred_prev_t2
            fig, ax = plt.subplots(figsize=(0.7, 1.2))
            ax.bar(0, 1 - pred_prev_t1, color='#800020', width=0.5, label="LOW")
            ax.bar(0, pred_prev_t1 - pred_prev_t2, bottom=1 - pred_prev_t1, color='gold', width=0.5, label="MODERATE")
            ax.bar(0, pred_prev_t2, bottom=1 - pred_prev_t2, color='green', width=0.5, label="HIGH")
            # LOW label
            low_y = (1 - pred_prev_t1) / 2
            ax.text(0, low_y, "LOW", ha='center', va='center', fontsize=5, color='white')
            ax.text(0.6, low_y, f"{(1 - pred_prev_t1)*100:.1f}%", ha='left', va='center', fontsize=5, color='#800020')
            # MODERATE label
            mod_y = 1 - pred_prev_t1 + (pred_prev_t1 - pred_prev_t2) / 2
            ax.text(0, mod_y, "MODERATE", ha='center', va='center', fontsize=5, color='black')
            ax.text(0.6, mod_y, f"{(pred_prev_t1 - pred_prev_t2)*100:.1f}%", ha='left', va='center', fontsize=5, color='gold')
            # HIGH label
            high_y = 1 - (pred_prev_t2 / 2)
            ax.text(0, high_y, "HIGH", ha='center', va='center', fontsize=5, color='white')
            ax.text(0.6, high_y, f"{(pred_prev_t2)*100:.1f}%", ha='left', va='center', fontsize=5, color='green')
            ax.set_xlim(-0.5, 0.8)
            ax.set_ylim(0, 1)
            ax.axis("off")
            st.pyplot(fig)

        # --- ROC and PR curves with T1/T2 markers ---
        st.markdown("#### ROC and Precision-Recall Curves")
        fig_roc = plot_roc_curve(metrics_df, t1)
        # Overlay T2 marker
        ax_roc = fig_roc.axes[0]
        idx_t2 = metrics_df["Threshold"] == t2
        ax_roc.plot(
            (1 - metrics_df["Specificity"][idx_t2]),
            metrics_df["Sensitivity (Recall)"][idx_t2],
            marker='o', color='purple', markersize=7, label="T2"
        )
        handles, labels = ax_roc.get_legend_handles_labels()
        if "T2" not in labels:
            ax_roc.legend(fontsize=6)
        st.pyplot(fig_roc)

        fig_pr = plot_pr_curve(metrics_df, t1, df)
        # Overlay T2 marker
        ax_pr = fig_pr.axes[0]
        ax_pr.plot(
            metrics_df["Sensitivity (Recall)"][idx_t2],
            metrics_df["PPV (Precision)"][idx_t2],
            marker='o', color='purple', markersize=7, label="T2"
        )
        handles, labels = ax_pr.get_legend_handles_labels()
        if "T2" not in labels:
            ax_pr.legend(fontsize=6)
        st.pyplot(fig_pr)

        # --- Box 2.5: Case Study on Performance (Three Zones) ---
        st.markdown("### Case Study on Performance: Three Risk Zones")
        st.markdown("It is the first morning after primary bariatric surgery. Please consider the following model behaviour. For 1000 patients, how are they classified into LOW, MODERATE, and HIGH risk groups?")

        sample_size = 1000
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
        tn_mod = (row_t2["True Negatives"] - row_t1["True Negatives"]) * sample_size / df["True Negatives"].max() if df["True Negatives"].max() > 0 else 0
        fn_mod = (row_t2["False Negatives"] - row_t1["False Negatives"]) * sample_size / df["False Negatives"].max() if df["False Negatives"].max() > 0 else 0
        fp_mod = (row_t2["False Positives"] - row_t1["False Positives"]) * sample_size / df["False Positives"].max() if df["False Positives"].max() > 0 else 0
        tp_mod = (row_t2["True Positives"] - row_t1["True Positives"]) * sample_size / df["True Positives"].max() if df["True Positives"].max() > 0 else 0
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
            st.markdown(f"#### Model Labels as <span style='color:#800020;'>LOW</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", tn_low, "#800020"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", fn_low, "#800020"), unsafe_allow_html=True)
        # MODERATE
        with col_mod:
            st.markdown(f"#### Model Labels as <span style='color:gold;'>MODERATE</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", tn_mod, "gold"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", fn_mod, "gold"), unsafe_allow_html=True)
        # HIGH
        with col_high:
            st.markdown(f"#### Model Labels as <span style='color:green;'>HIGH</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", fp_high, "green"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", tp_high, "green"), unsafe_allow_html=True)

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
            fig_roc = plot_roc_curve(metrics_df, selected_threshold)
            fig_pr = plot_pr_curve(metrics_df, selected_threshold, df)
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
            st.markdown(f"#### Model Labels as <span style='color:#800020;'>LOW</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", tn, "#800020"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", fn, "#800020"), unsafe_allow_html=True)

        with col_high:
            st.markdown(f"#### Model Labels as <span style='color:green;'>HIGH</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", fp, "green"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", tp, "green"), unsafe_allow_html=True)

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
            fig_roc = plot_roc_curve(metrics_df, selected_threshold)
            fig_pr = plot_pr_curve(metrics_df, selected_threshold, df)
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
            st.markdown(f"#### Model Labels as <span style='color:#800020;'>LOW</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", tn, "#800020"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", fn, "#800020"), unsafe_allow_html=True)

        with col_high:
            st.markdown(f"#### Model Labels as <span style='color:green;'>HIGH</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", fp, "green"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", tp, "green"), unsafe_allow_html=True)

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
            fig_roc = plot_roc_curve(metrics_df, selected_threshold)
            fig_pr = plot_pr_curve(metrics_df, selected_threshold, df)
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
            st.markdown(f"#### Model Labels as <span style='color:#800020;'>LOW</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", tn, "#800020"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", fn, "#800020"), unsafe_allow_html=True)

        with col_high:
            st.markdown(f"#### Model Labels as <span style='color:green;'>HIGH</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", fp, "green"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", tp, "green"), unsafe_allow_html=True)

        # --- Box 3: About Bleeding ---
        st.markdown("### About Postoperative Bleeding")
        st.markdown("""
Postoperative bleeding is a serious complication that can lead to reoperation, transfusion, or longer ICU stays.
Accurate early prediction helps clinicians intervene proactively.

This model leverages vital signs and clinical scores to predict bleeding risk after surgery.
""")

    except FileNotFoundError:
        st.error("Missing file: ./data/gbs_scores.csv")

# NEWS Custom Tab
with tab_news_custom:
    st.subheader("Model Output: Postoperative Sepsis")
    try:
        df = load_confusion_data("sepsis_custom_scores.csv")
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
                key="threshold_news_custom"
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
            st.markdown(f"#### Model Labels as <span style='color:#800020;'>LOW</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", tn, "#800020"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", fn, "#800020"), unsafe_allow_html=True)

        with col_high:
            st.markdown(f"#### Model Labels as <span style='color:green;'>HIGH</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", fp, "green"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", tp, "green"), unsafe_allow_html=True)

        # --- Box 3: About Sepsis ---
        st.markdown("### About Postoperative Sepsis")
        st.markdown("""
Sepsis is a life-threatening response to infection that can occur after surgery.
Early identification using predictive models is critical to initiate timely treatment.

This model uses physiological and lab data to anticipate sepsis onset based on trends in early recovery.
""")

    except FileNotFoundError:
        st.error("Missing file: ./data/sepsis_custom_scores.csv")

# GBS Custom Tab
with tab_gbs_custom:
    st.subheader("Model Output: Postoperative Bleeding")
    try:
        df = load_confusion_data("bleeding_custom_scores.csv")
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
                key="threshold_gbs_custom"
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
            st.markdown(f"#### Model Labels as <span style='color:#800020;'>LOW</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", tn, "#800020"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", fn, "#800020"), unsafe_allow_html=True)

        with col_high:
            st.markdown(f"#### Model Labels as <span style='color:green;'>HIGH</span>", unsafe_allow_html=True)
            st.markdown(figure_block(f"Cases without {label_term}", fp, "green"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown(figure_block(f"Cases with {label_term}", tp, "green"), unsafe_allow_html=True)

        # --- Box 3: About Bleeding ---
        st.markdown("### About Postoperative Bleeding")
        st.markdown("""
Postoperative bleeding is a serious complication that can lead to reoperation, transfusion, or longer ICU stays.
Accurate early prediction helps clinicians intervene proactively.

This model leverages vital signs and clinical scores to predict bleeding risk after surgery.
""")

    except FileNotFoundError:
        st.error("Missing file: ./data/bleeding_custom_scores.csv")