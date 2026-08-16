"""Classical ML sentiment classification and time-series aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
RANDOM_STATE = 42
DEFAULT_MODEL_PATH = Path("models/sentiment_model.joblib")


def build_pipeline() -> Pipeline:
    """Create the TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_model(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "sentiment",
    test_size: float = 0.2,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train the sentiment model and return metrics from the held-out test set."""
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError("Training DataFrame must include text and sentiment columns.")

    texts = df[text_column].astype(str)
    labels = df[label_column].astype(str)

    if texts.empty:
        raise ValueError("Cannot train on an empty dataset.")

    unique_labels = sorted(labels.unique())
    if len(unique_labels) < 2:
        raise ValueError("Training requires at least two sentiment classes.")

    stratify = labels if labels.nunique() > 1 and labels.value_counts().min() >= 2 else None
    split_kwargs: dict[str, Any] = {
        "test_size": test_size,
        "random_state": RANDOM_STATE,
    }
    if stratify is not None:
        split_kwargs["stratify"] = labels

    x_train, x_test, y_train, y_test = train_test_split(texts, labels, **split_kwargs)

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=SENTIMENT_LABELS
        ).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=SENTIMENT_LABELS, zero_division=0
        ),
        "classes": list(pipeline.classes_),
        "train_size": len(x_train),
        "test_size": len(x_test),
    }
    return pipeline, metrics


def save_model(pipeline: Pipeline, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Persist a trained pipeline to disk."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    return model_path


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a trained pipeline from disk."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def get_or_train_model(
    df: pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    force_retrain: bool = False,
) -> tuple[Pipeline, dict[str, Any] | None]:
    """Load an existing model or train and save a new one."""
    model_path = Path(model_path)
    if model_path.exists() and not force_retrain:
        return load_model(model_path), None

    pipeline, metrics = train_model(df)
    save_model(pipeline, model_path)
    return pipeline, metrics


def predict_sentiment(
    text: str, pipeline: Pipeline
) -> dict[str, Any]:
    """Predict sentiment and confidence for a single text input."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Prediction text cannot be empty.")

    probabilities = pipeline.predict_proba([cleaned])[0]
    classes = list(pipeline.classes_)
    best_index = int(np.argmax(probabilities))
    predicted_label = classes[best_index]

    return {
        "sentiment": predicted_label,
        "confidence": float(probabilities[best_index]),
        "probabilities": {
            label: float(probability)
            for label, probability in zip(classes, probabilities)
        },
    }


def get_sentiment_stats(df: pd.DataFrame, label_column: str = "sentiment") -> dict[str, Any]:
    """Calculate overall sentiment counts and percentages."""
    counts = df[label_column].value_counts()
    total = len(df)

    stats = {
        "total_posts": total,
        "positive_count": int(counts.get("positive", 0)),
        "negative_count": int(counts.get("negative", 0)),
        "neutral_count": int(counts.get("neutral", 0)),
    }
    if total == 0:
        stats.update(
            {
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 0.0,
            }
        )
        return stats

    stats.update(
        {
            "positive_pct": round(stats["positive_count"] / total * 100, 2),
            "negative_pct": round(stats["negative_count"] / total * 100, 2),
            "neutral_pct": round(stats["neutral_count"] / total * 100, 2),
        }
    )
    return stats


def aggregate_sentiment_over_time(
    df: pd.DataFrame,
    period: str = "day",
    timestamp_column: str = "timestamp",
    label_column: str = "sentiment",
) -> pd.DataFrame:
    """Aggregate sentiment counts and percentages by time period."""
    if period not in {"hour", "day"}:
        raise ValueError("Period must be 'hour' or 'day'.")

    if timestamp_column not in df.columns or label_column not in df.columns:
        raise ValueError("DataFrame must include timestamp and sentiment columns.")

    working = df.copy()
    working[timestamp_column] = pd.to_datetime(working[timestamp_column])
    freq = "h" if period == "hour" else "D"
    working["period"] = working[timestamp_column].dt.floor(freq)

    grouped = (
        working.groupby(["period", label_column])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SENTIMENT_LABELS, fill_value=0)
        .reset_index()
    )

    grouped["total_posts"] = grouped[SENTIMENT_LABELS].sum(axis=1)
    for label in SENTIMENT_LABELS:
        grouped[f"{label}_pct"] = np.where(
            grouped["total_posts"] > 0,
            grouped[label] / grouped["total_posts"] * 100,
            0.0,
        )

    return grouped.sort_values("period").reset_index(drop=True)
