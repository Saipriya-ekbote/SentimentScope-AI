"""SentimentScope AI Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.alerts import generate_alerts
from src.data_loader import load_csv
from src.preprocessing import preprocess_dataframe
from src.sentiment import (
    aggregate_sentiment_over_time,
    compare_models,
    get_or_train_model,
    get_sentiment_stats,
    predict_sentiment,
    SENTIMENT_LABELS,
)
from src.spike_detection import detect_negative_spikes

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sample_data.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.joblib"


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    """Load and preprocess the dataset once per session."""
    raw_df = load_csv(path)
    return preprocess_dataframe(raw_df)


@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_model(dataframe: pd.DataFrame, model_path: str):
    """Load an existing model or train and save a new one."""
    return get_or_train_model(dataframe, model_path=model_path)

@st.cache_data(show_spinner="Comparing sentiment models...")
def run_model_comparison(dataframe: pd.DataFrame):
    """Compare Logistic Regression and Linear SVM."""
    return compare_models(dataframe)


def platform_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize sentiment counts by platform."""
    summary = (
        df.groupby(["platform", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for label in ["positive", "negative", "neutral"]:
        if label not in summary.columns:
            summary[label] = 0
    summary["total_posts"] = summary[["positive", "negative", "neutral"]].sum(axis=1)
    return summary.sort_values("platform")


def main() -> None:
    st.set_page_config(page_title="SentimentScope AI", layout="wide")
    st.title("SentimentScope AI")
    st.caption("Multi-Platform Social Sentiment Monitor with Spike Alerts")
    st.info(
        "This dashboard uses **SYNTHETIC DEVELOPMENT DATA** from a local CSV file. "
        "It does not connect to live social-media platforms."
    )

    data_path = st.sidebar.text_input("Dataset path", value=str(DEFAULT_DATA_PATH))
    period = st.sidebar.selectbox("Time aggregation", options=["day", "hour"], index=0)
    z_threshold = st.sidebar.slider("Spike z-score threshold", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

    try:
        df = load_dataset(data_path)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    try:
        model, training_metrics = load_sentiment_model(df, str(DEFAULT_MODEL_PATH))
    except Exception as exc:  # noqa: BLE001 - dashboard must surface failures clearly
        st.error(f"Model loading/training failed: {exc}")
        st.stop()

    try:
        model_comparison = run_model_comparison(df)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Model comparison failed: {exc}")
        st.stop()

    st.subheader("Model Comparison")

    # Build a summary table
    comparison_rows = []
    for model_name, metrics in model_comparison.items():
        comparison_rows.append(
            {
                "Model": model_name,
                "Accuracy": round(metrics["accuracy"], 4),
                "Precision": round(metrics["precision"], 4),
                "Recall": round(metrics["recall"], 4),
                "F1 Score": round(metrics["f1_score"], 4),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df, use_container_width=True)

    # Highlight best model by F1 score
    if not comparison_df.empty:
        best_idx = comparison_df["F1 Score"].idxmax()
        best_model = comparison_df.loc[best_idx, "Model"]
        best_f1 = comparison_df.loc[best_idx, "F1 Score"]
        st.success(f"**Best model (by F1 Score): {best_model}** (F1 = {best_f1})")

    # Detailed diagnostics per model
    for model_name, metrics in model_comparison.items():
        st.subheader(f"{model_name} details")
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=SENTIMENT_LABELS, columns=SENTIMENT_LABELS)
        with st.expander("Confusion Matrix"):
            st.dataframe(cm_df)
        with st.expander("Classification Report"):
            st.text(metrics["classification_report"])
    
    stats = get_sentiment_stats(df)
    time_series = aggregate_sentiment_over_time(df, period=period)
    spikes = detect_negative_spikes(time_series, z_threshold=z_threshold)
    alerts = generate_alerts(spikes)

    st.subheader("Overview")
    overview_cols = st.columns(5)
    overview_cols[0].metric("Total Posts", stats["total_posts"])
    overview_cols[1].metric("Positive %", f"{stats['positive_pct']:.1f}%")
    overview_cols[2].metric("Negative %", f"{stats['negative_pct']:.1f}%")
    overview_cols[3].metric("Neutral %", f"{stats['neutral_pct']:.1f}%")
    overview_cols[4].metric("Detected Spikes", len(spikes))

    if training_metrics is not None:
        with st.expander("Model training metrics (latest training run)"):
            st.write(
                {
                    "accuracy": round(training_metrics["accuracy"], 4),
                    "precision": round(training_metrics["precision"], 4),
                    "recall": round(training_metrics["recall"], 4),
                    "f1_score": round(training_metrics["f1_score"], 4),
                }
            )
            st.text(training_metrics["classification_report"])

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.subheader("Sentiment Distribution")
        distribution_df = pd.DataFrame(
            {
                "sentiment": ["positive", "negative", "neutral"],
                "count": [
                    stats["positive_count"],
                    stats["negative_count"],
                    stats["neutral_count"],
                ],
            }
        )
        fig_distribution = px.bar(
            distribution_df,
            x="sentiment",
            y="count",
            color="sentiment",
            title="Sentiment Counts",
        )
        st.plotly_chart(fig_distribution, use_container_width=True)

    with chart_cols[1]:
        st.subheader("Sentiment Over Time")
        fig_time = px.line(
            time_series,
            x="period",
            y=["positive", "negative", "neutral"],
            markers=True,
            title=f"Sentiment Activity by {period}",
        )
        st.plotly_chart(fig_time, use_container_width=True)

    st.subheader("Platform Analysis")
    platform_df = platform_summary(df)
    st.dataframe(platform_df, use_container_width=True)

    st.subheader("Spike Alerts")
    if alerts:
        for alert in alerts:
            st.warning(
                f"**{alert['alert_type']}** | Severity: {alert['severity']} | "
                f"Time: {alert['timestamp']} | Observed: {alert['observed_value']:.0f} | "
                f"Baseline: {alert['baseline_value']:.2f} | "
                f"Threshold: {alert['threshold']}"
            )
            st.write(alert["message"])
    else:
        st.success("No unusual sentiment spikes detected.")

    st.subheader("Live Text Prediction")
    user_text = st.text_area(
        "Enter text to classify",
        value="I am extremely disappointed with this service.",
    )
    if st.button("Predict Sentiment"):
        if not user_text.strip():
            st.error("Please enter text before requesting a prediction.")
        else:
            try:
                result = predict_sentiment(user_text, model)
                st.write(f"Predicted sentiment: **{result['sentiment']}**")
                st.write(f"Confidence: **{result['confidence'] * 100:.1f}%**")
            except ValueError as exc:
                st.error(str(exc))

if __name__ == "__main__":
    main()
