"""Tests for sentiment model training and prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.sentiment import (
    aggregate_sentiment_over_time,
    get_sentiment_stats,
    load_model,
    predict_sentiment,
    save_model,
    train_model,
)


def test_train_model_returns_real_metrics(training_df: pd.DataFrame) -> None:
    pipeline, metrics = train_model(training_df)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 3
    assert set(pipeline.classes_) <= {"positive", "negative", "neutral"}


def test_predict_sentiment_returns_valid_label(training_df: pd.DataFrame) -> None:
    pipeline, _ = train_model(training_df)
    result = predict_sentiment("I love this product", pipeline)
    assert result["sentiment"] in {"positive", "negative", "neutral"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert pytest.approx(sum(result["probabilities"].values()), rel=1e-6) == 1.0


def test_predict_sentiment_rejects_empty_text(training_df: pd.DataFrame) -> None:
    pipeline, _ = train_model(training_df)
    with pytest.raises(ValueError, match="Prediction text cannot be empty"):
        predict_sentiment("   ", pipeline)


def test_model_persistence(training_df: pd.DataFrame, tmp_path: Path) -> None:
    pipeline, _ = train_model(training_df)
    model_path = tmp_path / "sentiment_model.joblib"
    save_model(pipeline, model_path)
    loaded = load_model(model_path)
    assert list(loaded.classes_) == list(pipeline.classes_)


def test_get_sentiment_stats(training_df: pd.DataFrame) -> None:
    stats = get_sentiment_stats(training_df)
    assert stats["total_posts"] == len(training_df)
    assert stats["positive_pct"] + stats["negative_pct"] + stats["neutral_pct"] == pytest.approx(
        100.0, abs=0.1
    )


def test_aggregate_sentiment_over_time(training_df: pd.DataFrame) -> None:
    aggregated = aggregate_sentiment_over_time(training_df, period="day")
    assert "period" in aggregated.columns
    assert "total_posts" in aggregated.columns
    assert aggregated["total_posts"].sum() == len(training_df)
