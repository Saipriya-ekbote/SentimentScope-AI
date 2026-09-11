"""SentimentScope AI Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.alerts import generate_alerts
from src.data_loader import load_csv
from src.entity_detection import EntityDetector
from src.entity_spike_detection import detect_all_entity_spikes
from src.preprocessing import preprocess_dataframe
from src.sentiment import (
    SENTIMENT_LABELS,
    compare_models,
    get_or_train_model,
    get_sentiment_stats,
    predict_sentiment,
)
from src.spike_detection import detect_negative_spikes
from src.time_series import build_time_series

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
        width="stretch",
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
            width="stretch",
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
            width="stretch",
        )

    # ---------------------------------------------------------
    # Platform Analysis
    # ---------------------------------------------------------

    st.subheader("Platform Analysis")

    platform_df = platform_summary(df)

    st.dataframe(
        platform_df,
        width="stretch",
    )

    # ---------------------------------------------------------
    # Spike Alerts
    # ---------------------------------------------------------

    st.subheader("Spike Alerts")

    # Alert summary metrics
    total_alerts = len(alerts)

    critical_alerts = sum(
        1
        for alert in alerts
        if alert["severity"] == "CRITICAL"
    )

    high_alerts = sum(
        1
        for alert in alerts
        if alert["severity"] == "HIGH"
    )

    entity_alerts_count = sum(
        1
        for alert in alerts
        if "entity" in alert and "dimension" in alert
    )

    summary_cols = st.columns(4)

    summary_cols[0].metric(
        "Total Alerts",
        total_alerts,
    )

    summary_cols[1].metric(
        "Critical Alerts",
        critical_alerts,
    )

    summary_cols[2].metric(
        "High Alerts",
        high_alerts,
    )

    summary_cols[3].metric(
        "Entity Alerts",
        entity_alerts_count,
    )

    # Alert severity distribution
    if alerts:
        severity_counts = {
            "CRITICAL": critical_alerts,
            "HIGH": high_alerts,
            "MEDIUM": sum(
                1
                for alert in alerts
                if alert["severity"] == "MEDIUM"
            ),
        }

        severity_df = pd.DataFrame(
            {
                "Severity": severity_counts.keys(),
                "Count": severity_counts.values(),
            }
        )

        st.bar_chart(
            severity_df.set_index("Severity")
        )
    else:
        st.caption("No alert severity data available.")

    # Alert filters and sorting
    filter_cols = st.columns(4)
    with filter_cols[0]:
        severity_filter = st.selectbox(
            "Severity",
            ["All", "CRITICAL", "HIGH", "MEDIUM"],
            key="alert_severity_filter",
        )

    with filter_cols[1]:
        scope_filter = st.selectbox(
            "Alert Scope",
            ["All", "Global", "Entity"],
            key="alert_scope_filter",
        )

    with filter_cols[2]:
        dimension_filter = st.selectbox(
            "Dimension",
            ["All", "Brand", "Product", "Topic"],
            key="alert_dimension_filter",
        )

    with filter_cols[3]:
        sort_option = st.selectbox(
            "Sort Alerts By",
            [
                "Priority",
                "Latest",
                "Increase %",
                "Z-score",
            ],
            key="alert_sort_option",
        )

    # Apply filters
    filtered_alerts = []

    for alert in alerts:

        # Severity filter
        if (
            severity_filter != "All"
            and alert["severity"] != severity_filter
        ):
            continue

        # Scope filter
        is_entity_alert = (
            "entity" in alert
            and "dimension" in alert
        )

        if scope_filter == "Entity" and not is_entity_alert:
            continue

        if scope_filter == "Global" and is_entity_alert:
            continue

        # Dimension filter
        if dimension_filter != "All":

            if not is_entity_alert:
                continue

            if (
                alert["dimension"].title()
                != dimension_filter
            ):
                continue

        filtered_alerts.append(alert)

    # Sort filtered alerts
    if sort_option == "Priority":
        filtered_alerts.sort(
            key=lambda alert: (
                alert["priority"],
                alert["timestamp"],
            )
        )

    elif sort_option == "Latest":
        filtered_alerts.sort(
            key=lambda alert: alert["timestamp"],
            reverse=True,
        )

    elif sort_option == "Increase %":
        filtered_alerts.sort(
            key=lambda alert: alert["increase_percent"],
            reverse=True,
        )

    elif sort_option == "Z-score":
        filtered_alerts.sort(
            key=lambda alert: alert["z_score"],
            reverse=True,
        )

    # Display filtered alerts
    if filtered_alerts:

        st.caption(
            f"Showing {len(filtered_alerts)} "
            f"of {len(alerts)} alert(s)"
        )

        # Export filtered alerts as CSV
        alerts_df = pd.DataFrame(filtered_alerts)

        csv_data = alerts_df.to_csv(index=False)

        st.download_button(
            label="Download Alerts as CSV",
            data=csv_data,
            file_name="sentiment_alerts.csv",
            mime="text/csv",
        )

        # Alert history table
        history_rows = []

        for alert in filtered_alerts:
            history_rows.append(
                {
                    "Timestamp": alert["timestamp"],
                    "Severity": alert["severity"],
                    "Priority": alert["priority"],
                    "Scope": (
                        "Entity"
                        if "entity" in alert and "dimension" in alert
                        else "Global"
                    ),
                    "Dimension": (
                        alert.get("dimension", "Global").title()
                    ),
                    "Entity": alert.get("entity", "All"),
                    "Observed": alert["observed_value"],
                    "Baseline": alert["baseline_value"],
                    "Increase %": alert["increase_percent"],
                    "Z-score": alert["z_score"],
                }
            )

        history_df = pd.DataFrame(history_rows)

        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Observed": st.column_config.NumberColumn(
                    format="%.0f",
                ),
                "Baseline": st.column_config.NumberColumn(
                    format="%.2f",
                ),
                "Increase %": st.column_config.NumberColumn(
                    format="%.1f%%",
                ),
                "Z-score": st.column_config.NumberColumn(
                    format="%.2f",
                ),
            },
        )

        for alert in filtered_alerts:

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
                f"**Observed:** "
                f"{alert['observed_value']:.0f}  \n"
                f"**Baseline:** "
                f"{alert['baseline_value']:.2f}  \n"
                f"**Increase:** "
                f"{alert['increase_percent']:.1f}%  \n"
                f"**Z-score:** "
                f"{alert['z_score']:.2f}  \n"
                f"**Priority:** "
                f"{alert['priority']}  \n"
                f"**Threshold:** "
                f"{alert['threshold']}  \n"
                f"**Time:** "
                f"{alert['timestamp']}"
            )

    else:

        if alerts:
            st.info(
                "No alerts match the selected filters."
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


