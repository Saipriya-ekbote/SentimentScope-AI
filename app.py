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
    compare_models,
    get_or_train_model,
    get_sentiment_stats,
    predict_sentiment,
    SENTIMENT_LABELS,
)
from src.time_series import build_time_series
from src.spike_detection import detect_negative_spikes
from src.entity_detection import EntityDetector
from src.entity_spike_detection import detect_all_entity_spikes

PROJECT_ROOT = Path(__file__).resolve().parent


DATASET_CONFIGS = {
    "Sample Dataset": {
        "data_path": PROJECT_ROOT / "data" / "sample_data.csv",
        "model_path": PROJECT_ROOT / "models" / "sentiment_model_sample.joblib",
        "is_synthetic": True,
        "description": (
            "Synthetic development/demo dataset across multiple simulated "
            "platforms with an injected negative spike."
        ),
    },
    "Realistic Twitter Dataset": {
        "data_path": PROJECT_ROOT / "data" / "realistic_social_data.csv",
        "model_path": PROJECT_ROOT / "models" / "sentiment_model_realistic.joblib",
        "is_synthetic": False,
        "description": (
            "Real-world Twitter US Airline Sentiment dataset "
            "(14,485 deduplicated posts) prepared from data/raw/Tweets.csv."
        ),
    },
}


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

    summary["total_posts"] = summary[
        ["positive", "negative", "neutral"]
    ].sum(axis=1)

    return summary.sort_values("platform")


def main() -> None:
    """Run the Streamlit dashboard."""

    st.set_page_config(
        page_title="SentimentScope AI",
        layout="wide",
    )

    st.title("SentimentScope AI")
    st.caption("Multi-Platform Social Sentiment Monitor with Spike Alerts")

    # ---------------------------------------------------------
    # Dataset selection
    # ---------------------------------------------------------

    dataset_choice = st.sidebar.selectbox(
        "Select Dataset",
        options=list(DATASET_CONFIGS.keys()),
        index=0,
    )

    dataset_cfg = DATASET_CONFIGS[dataset_choice]

    data_path = dataset_cfg["data_path"]
    model_path = dataset_cfg["model_path"]
    is_synthetic = dataset_cfg["is_synthetic"]

    if is_synthetic:
        st.info(
            "This dashboard uses **SYNTHETIC DEVELOPMENT DATA** from a local "
            "CSV file. It does not connect to live social-media platforms."
        )
    else:
        st.info(
            "This dashboard uses the **REAL-WORLD TWITTER DATASET** "
            "(Twitter US Airline Sentiment, 14,485 deduplicated posts). "
            "It reflects authentic social-media noise and customer feedback."
        )

    # ---------------------------------------------------------
    # Dashboard controls
    # ---------------------------------------------------------

    period = st.sidebar.selectbox(
        "Time aggregation",
        options=["day", "hour"],
        index=0,
    )

    z_threshold = st.sidebar.slider(
        "Spike z-score threshold",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.1,
    )

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    try:
        df = load_dataset(str(data_path))
        entity_detector = EntityDetector()
        df = entity_detector.detect_entities_dataframe(df, overwrite=False)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    # ---------------------------------------------------------
    # Load sentiment model
    # ---------------------------------------------------------

    try:
        model, training_metrics = load_sentiment_model(
            df,
            str(model_path),
        )

    except Exception as exc:  # noqa: BLE001
        st.error(f"Model loading/training failed: {exc}")
        st.stop()

    # ---------------------------------------------------------
    # Model comparison
    # ---------------------------------------------------------

    try:
        model_comparison = run_model_comparison(df)

    except Exception as exc:  # noqa: BLE001
        st.error(f"Model comparison failed: {exc}")
        st.stop()

    st.subheader("Model Comparison")

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

    st.dataframe(
        comparison_df,
        use_container_width=True,
    )

    # Highlight best model
    if not comparison_df.empty:
        best_idx = comparison_df["F1 Score"].idxmax()
        best_model = comparison_df.loc[best_idx, "Model"]
        best_f1 = comparison_df.loc[best_idx, "F1 Score"]

        st.success(
            f"**Best model (by F1 Score): {best_model}** "
            f"(F1 = {best_f1})"
        )

    # ---------------------------------------------------------
    # Detailed model diagnostics
    # ---------------------------------------------------------

    for model_name, metrics in model_comparison.items():

        st.subheader(f"{model_name} details")

        cm = metrics["confusion_matrix"]

        cm_df = pd.DataFrame(
            cm,
            index=SENTIMENT_LABELS,
            columns=SENTIMENT_LABELS,
        )

        with st.expander("Confusion Matrix"):
            st.dataframe(cm_df)

        with st.expander("Classification Report"):
            st.text(metrics["classification_report"])

    # ---------------------------------------------------------
    # Sentiment statistics
    # ---------------------------------------------------------

    stats = get_sentiment_stats(df)

    # ---------------------------------------------------------
    # Phase 6 - Time Series
    # ---------------------------------------------------------

    freq = "daily" if period == "day" else "hourly"

    time_series = build_time_series(
        df,
        freq=freq,
    )

    # ---------------------------------------------------------
    # Phase 6 - Spike Detection
    # ---------------------------------------------------------

    # The existing spike detector expects the historical
    # column name "period", while the new time-series
    # architecture uses "timestamp".
    spike_series = time_series.rename(
        columns={"timestamp": "period"}
    )

    spikes = detect_negative_spikes(
        spike_series,
        value_column="negative",
        z_threshold=z_threshold,
    )
    entity_spikes = detect_all_entity_spikes(
        df,
        dimensions=["brand", "product", "topic"],
        z_threshold=z_threshold,
        freq=freq,
    )

    # ---------------------------------------------------------
    # Phase 6 - Alert Generation
    # ---------------------------------------------------------
    global_alerts = generate_alerts(spikes)
    entity_alerts = generate_alerts(entity_spikes)

    alerts = global_alerts + entity_alerts

    # ---------------------------------------------------------
    # Overview
    # ---------------------------------------------------------

    st.subheader("Overview")

    overview_cols = st.columns(5)

    overview_cols[0].metric(
        "Total Posts",
        stats["total_posts"],
    )

    overview_cols[1].metric(
        "Positive %",
        f"{stats['positive_pct']:.1f}%",
    )

    overview_cols[2].metric(
        "Negative %",
        f"{stats['negative_pct']:.1f}%",
    )

    overview_cols[3].metric(
        "Neutral %",
        f"{stats['neutral_pct']:.1f}%",
    )

    overview_cols[4].metric(
        "Detected Spikes",
        len(spikes) + len(entity_spikes),
        delta_color="normal",
    )

    # ---------------------------------------------------------
    # Model Training Metrics
    # ---------------------------------------------------------

    if training_metrics is not None:

        with st.expander(
            "Model training metrics (latest training run)"
        ):

            st.write(
                {
                    "accuracy": round(
                        training_metrics["accuracy"],
                        4,
                    ),
                    "precision": round(
                        training_metrics["precision"],
                        4,
                    ),
                    "recall": round(
                        training_metrics["recall"],
                        4,
                    ),
                    "f1_score": round(
                        training_metrics["f1_score"],
                        4,
                    ),
                }
            )

            st.text(
                training_metrics["classification_report"]
            )

    # ---------------------------------------------------------
    # Charts
    # ---------------------------------------------------------

    chart_cols = st.columns(2)

    # Sentiment Distribution
    with chart_cols[0]:

        st.subheader("Sentiment Distribution")

        distribution_df = pd.DataFrame(
            {
                "sentiment": [
                    "positive",
                    "negative",
                    "neutral",
                ],
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

        st.plotly_chart(
            fig_distribution,
            use_container_width=True,
        )

    # Sentiment Over Time
    with chart_cols[1]:

        st.subheader("Sentiment Over Time")

        fig_time = px.line(
            time_series,
            x="timestamp",
            y=[
                "positive",
                "negative",
                "neutral",
            ],
            markers=True,
            title=f"Sentiment Activity by {period}",
        )

        st.plotly_chart(
            fig_time,
            use_container_width=True,
        )

    # ---------------------------------------------------------
    # Platform Analysis
    # ---------------------------------------------------------

    st.subheader("Platform Analysis")

    platform_df = platform_summary(df)

    st.dataframe(
        platform_df,
        use_container_width=True,
    )

    # ---------------------------------------------------------
    # Spike Alerts
    # ---------------------------------------------------------

    st.subheader("Spike Alerts")

    if alerts:
        for alert in alerts:
            severity = alert["severity"]

            if severity == "CRITICAL":
                alert_box = st.error
            elif severity == "HIGH":
                alert_box = st.warning
            else:
                alert_box = st.info

            if "entity" in alert and "dimension" in alert:
                title = (
                    f"{severity} | "
                    f"{alert['dimension'].title()}: "
                    f"{alert['entity']}"
                )
            else:
                title = severity

            alert_box(
                f"**{title}**\n\n"
                f"{alert['message']}\n\n"
                f"**Observed:** {alert['observed_value']:.0f}  \n"
                f"**Baseline:** {alert['baseline_value']:.2f}  \n"
                f"**Increase:** {alert['increase_percent']:.1f}%  \n"
                f"**Z-score:** {alert['z_score']:.2f}  \n"
                f"**Priority:** {alert['priority']}  \n"
                f"**Threshold:** {alert['threshold']}  \n"
                f"**Time:** {alert['timestamp']}"
            )
    else:
        st.success(
            "No unusual sentiment spikes detected."
        )
    # ---------------------------------------------------------
    # Live Text Prediction
    # ---------------------------------------------------------

    st.subheader("Live Text Prediction")

    user_text = st.text_area(
        "Enter text to classify",
        value="I am extremely disappointed with this service.",
    )

    if st.button("Predict Sentiment"):

        if not user_text.strip():

            st.error(
                "Please enter text before requesting a prediction."
            )

        else:

            try:

                result = predict_sentiment(
                    user_text,
                    model,
                )

                st.write(
                    f"Predicted sentiment: "
                    f"**{result['sentiment']}**"
                )

                st.write(
                    f"Confidence: "
                    f"**{result['confidence'] * 100:.1f}%**"
                )

            except ValueError as exc:

                st.error(str(exc))


if __name__ == "__main__":
    main()


