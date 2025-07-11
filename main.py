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
tab_news, tab_gbs, tab_news_custom, tab_gbs_custom, tab_qsofa = st.tabs(["NEWS", "GBS", "Sepsis Custom", "Bleeding Custom", "qSOFA"])

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