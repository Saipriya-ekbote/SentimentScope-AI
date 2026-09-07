"""Tests for sentiment model training, prediction, continuous scoring, and evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.sentiment import (
    aggregate_sentiment_over_time,
    build_svm_pipeline,
    compute_continuous_score,
    get_sentiment_stats,
    load_model,
    predict_dataframe,
    predict_sentiment,
    predict_sentiment_with_score,
    save_model,
    train_model,
)


# --- Legacy Tests (Preserved 100%) ---

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

    # Verify predictions match between original and persisted model
    sample_text = "The customer service was great!"
    pred_orig = predict_sentiment(sample_text, pipeline)
    pred_loaded = predict_sentiment(sample_text, loaded)
    assert pred_orig["sentiment"] == pred_loaded["sentiment"]
    assert pytest.approx(pred_orig["sentiment_score"], abs=1e-4) == pred_loaded["sentiment_score"]


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


# --- Phase 4 Continuous Scoring & Batch Prediction Tests ---

def test_compute_continuous_score_properties() -> None:
    # 1. Pure positive
    assert compute_continuous_score({"positive": 1.0, "neutral": 0.0, "negative": 0.0}) == 1.0

    # 2. Pure negative
    assert compute_continuous_score({"positive": 0.0, "neutral": 0.0, "negative": 1.0}) == -1.0

    # 3. Pure neutral
    assert compute_continuous_score({"positive": 0.0, "neutral": 1.0, "negative": 0.0}) == 0.0

    # 4. Balanced positive and negative
    assert compute_continuous_score({"positive": 0.45, "neutral": 0.1, "negative": 0.45}) == 0.0

    # 5. Strongly leaning positive
    score_pos = compute_continuous_score({"positive": 0.85, "neutral": 0.1, "negative": 0.05})
    assert score_pos == 0.80

    # 6. Strictly bounded in [-1.0, +1.0]
    assert -1.0 <= compute_continuous_score({"positive": 0.99, "negative": 0.01}) <= 1.0
    assert -1.0 <= compute_continuous_score({"positive": 0.0, "negative": 0.99}) <= 1.0


def test_predict_sentiment_includes_continuous_score(training_df: pd.DataFrame) -> None:
    pipeline, _ = train_model(training_df)
    result = predict_sentiment("The service was fantastic!", pipeline)

    assert "sentiment_score" in result
    assert isinstance(result["sentiment_score"], float)
    assert -1.0 <= result["sentiment_score"] <= 1.0
    # Strong positive sentence should produce positive score
    assert result["sentiment_score"] > 0.0
    assert result["sentiment"] == "positive"


def test_predict_sentiment_with_score_alias(training_df: pd.DataFrame) -> None:
    pipeline, _ = train_model(training_df)
    result = predict_sentiment_with_score("Terrible awful failure", pipeline)

    assert result["sentiment"] == "negative"
    assert result["sentiment_score"] < 0.0
    assert -1.0 <= result["sentiment_score"] <= 1.0


def test_predict_sentiment_directional_consistency(training_df: pd.DataFrame) -> None:
    pipeline, _ = train_model(training_df)

    pos_res = predict_sentiment("I absolutely love this product!", pipeline)
    neg_res = predict_sentiment("I hate this product, it is broken.", pipeline)
    neu_res = predict_sentiment("The report was delivered yesterday.", pipeline)

    assert pos_res["sentiment_score"] > neu_res["sentiment_score"]
    assert neu_res["sentiment_score"] > neg_res["sentiment_score"]


def test_linear_svm_continuous_score(training_df: pd.DataFrame) -> None:
    # Verify continuous score works on LinearSVM (using calibrated decision function)
    svm_pipe = build_svm_pipeline()
    svm_pipe.fit(training_df["text"], training_df["sentiment"])

    res = predict_sentiment("Outstanding experience and very happy!", svm_pipe)
    assert res["sentiment"] in {"positive", "neutral", "negative"}
    assert -1.0 <= res["sentiment_score"] <= 1.0
    assert isinstance(res["confidence"], float)


def test_predict_dataframe_batch(training_df: pd.DataFrame) -> None:
    pipeline, _ = train_model(training_df)

    input_df = pd.DataFrame(
        {
            "post_id": ["p1", "p2", "p3"],
            "platform": ["Twitter", "Reddit", "YouTube"],
            "timestamp": ["2026-08-01 10:00:00", "2026-08-01 11:00:00", "2026-08-01 12:00:00"],
            "author": ["alice", "bob", "carol"],
            "text": [
                "I love this excellent update!",
                "Worst experience ever, totally broken.",
                "Meeting is at three today.",
            ],
            "brand": ["Apple", "Delta", "Amazon"],
            "provenance": ["Synthetic Demo", "Synthetic Demo", "Synthetic Demo"],
        }
    )

    batch_res = predict_dataframe(input_df, pipeline)

    # 1. Output row count matches input
    assert len(batch_res) == 3

    # 2. Original columns are preserved
    for col in ["post_id", "platform", "timestamp", "author", "brand", "provenance"]:
        assert col in batch_res.columns
        assert batch_res[col].tolist() == input_df[col].tolist()

    # 3. Sentiment & sentiment_score columns are populated
    assert "sentiment" in batch_res.columns
    assert "sentiment_score" in batch_res.columns
    assert batch_res.loc[0, "sentiment"] == "positive"
    assert batch_res.loc[0, "sentiment_score"] > 0.0
    assert batch_res.loc[1, "sentiment"] == "negative"
    assert batch_res.loc[1, "sentiment_score"] < 0.0


def test_predict_dataframe_handles_missing_and_empty_text(training_df: pd.DataFrame) -> None:
    pipeline, _ = train_model(training_df)

    input_df = pd.DataFrame(
        {
            "post_id": ["p1", "p2"],
            "text": [None, "   "],
        }
    )

    batch_res = predict_dataframe(input_df, pipeline)
    assert len(batch_res) == 2
    assert batch_res["sentiment"].tolist() == ["neutral", "neutral"]
    assert batch_res["sentiment_score"].tolist() == [0.0, 0.0]
