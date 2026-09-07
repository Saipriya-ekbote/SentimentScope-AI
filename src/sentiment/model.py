"""Model definitions, pipelines, persistence, and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocessing import clean_text
from src.sentiment.scorer import compute_continuous_score, extract_probabilities

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


def build_svm_pipeline() -> Pipeline:
    """Create the TF-IDF + LinearSVC pipeline."""
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
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def save_model(pipeline: Pipeline, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Persist a trained pipeline to disk using joblib."""
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


def predict_sentiment(text: str, pipeline: Pipeline) -> dict[str, Any]:
    """Predict sentiment, confidence, probabilities, and continuous score for a text.

    Args:
        text: Input string.
        pipeline: Trained sklearn Pipeline.

    Returns:
        Dictionary containing:
        - 'sentiment': Predicted class ('positive', 'neutral', 'negative')
        - 'confidence': Probability or normalized decision margin of predicted class
        - 'probabilities': Dict of class probabilities
        - 'sentiment_score': Continuous score in range [-1.0, +1.0]

    Raises:
        ValueError: If prediction text is empty or whitespace-only.
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Prediction text cannot be empty.")

    # Apply preprocessing to handle emojis, elongation, etc.
    preprocessed = clean_text(cleaned)
    eval_text = preprocessed if preprocessed else cleaned

    probabilities = extract_probabilities(pipeline, eval_text)
    classes = list(pipeline.classes_)

    # Select class with highest probability
    best_class = max(probabilities.keys(), key=lambda c: probabilities[c])
    confidence = float(probabilities[best_class])

    # Compute continuous polarity score [-1.0, +1.0]
    score = compute_continuous_score(probabilities)

    return {
        "sentiment": best_class,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "sentiment_score": score,
    }


def predict_sentiment_with_score(text: str, pipeline: Pipeline) -> dict[str, Any]:
    """Explicit alias returning structured sentiment classification and continuous score."""
    return predict_sentiment(text, pipeline)


def predict_dataframe(
    df: pd.DataFrame,
    pipeline: Pipeline,
    text_column: str = "text",
    sentiment_col: str = "sentiment",
    score_col: str = "sentiment_score",
    clean: bool = True,
) -> pd.DataFrame:
    """Perform batch prediction on a DataFrame, populating sentiment and score columns.

    Preserves all existing columns (including post_id, platform, timestamp, author, etc.).

    Args:
        df: Input pandas DataFrame (e.g. from normalized schema).
        pipeline: Trained sklearn Pipeline.
        text_column: Name of text column to evaluate.
        sentiment_col: Name of column to store predicted class.
        score_col: Name of column to store continuous score.
        clean: Whether to apply Phase 3 preprocessing before classification.

    Returns:
        pd.DataFrame with populated sentiment and sentiment_score columns.
    """
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in DataFrame.")

    result = df.copy()

    sentiments: list[str] = []
    scores: list[float] = []

    for val in result[text_column]:
        if pd.isna(val) or not str(val).strip():
            # Graceful fallback for empty/whitespace rows in batch mode
            sentiments.append("neutral")
            scores.append(0.0)
            continue

        raw_str = str(val).strip()
        eval_str = clean_text(raw_str) if clean else raw_str
        if not eval_str:
            eval_str = raw_str

        probs = extract_probabilities(pipeline, eval_str)
        best_class = max(probs.keys(), key=lambda c: probs[c])
        score = compute_continuous_score(probs)

        sentiments.append(best_class)
        scores.append(score)

    result[sentiment_col] = sentiments
    result[score_col] = scores
    return result
