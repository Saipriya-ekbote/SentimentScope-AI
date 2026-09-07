"""SentimentScope AI sentiment analysis and scoring package."""

from __future__ import annotations

from src.sentiment.evaluator import (
    aggregate_sentiment_over_time,
    compare_models,
    get_or_train_model,
    get_sentiment_stats,
    train_model,
)
from src.sentiment.model import (
    DEFAULT_MODEL_PATH,
    RANDOM_STATE,
    SENTIMENT_LABELS,
    build_pipeline,
    build_svm_pipeline,
    load_model,
    predict_dataframe,
    predict_sentiment,
    predict_sentiment_with_score,
    save_model,
)
from src.sentiment.scorer import (
    compute_continuous_score,
    decision_function_to_probabilities,
    extract_probabilities,
)

__all__ = [
    "DEFAULT_MODEL_PATH",
    "RANDOM_STATE",
    "SENTIMENT_LABELS",
    "aggregate_sentiment_over_time",
    "build_pipeline",
    "build_svm_pipeline",
    "compare_models",
    "compute_continuous_score",
    "decision_function_to_probabilities",
    "extract_probabilities",
    "get_or_train_model",
    "get_sentiment_stats",
    "load_model",
    "predict_dataframe",
    "predict_sentiment",
    "predict_sentiment_with_score",
    "save_model",
    "train_model",
]
