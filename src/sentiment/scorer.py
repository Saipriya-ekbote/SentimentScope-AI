"""Continuous sentiment scoring engine for SentimentScope AI.

Maps classification outputs into a bounded, deterministic continuous polarity
score in the range [-1.0, +1.0]:
  -1.0 = Strongly Negative
   0.0 = Neutral
  +1.0 = Strongly Positive
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_continuous_score(probabilities: dict[str, float]) -> float:
    """Compute continuous polarity score from class probabilities.

    Mathematical Formulation:
        score = P(positive) - P(negative)

    Properties:
    1. Bounded: Strictly in [-1.0, +1.0] since P(positive), P(negative) in [0.0, 1.0].
    2. Neutral Attenuation: If P(neutral) is high, both P(pos) and P(neg) are small,
       centering the score at 0.0.
    3. Monotonic & Explainable: Directly measures the directional probability margin
       without arbitrary tuning constants.

    Args:
        probabilities: Dictionary mapping class labels ('positive', 'neutral', 'negative')
                       to calibrated probabilities.

    Returns:
        Float score bounded in [-1.0, +1.0], rounded to 4 decimal places.
    """
    pos_prob = float(probabilities.get("positive", 0.0))
    neg_prob = float(probabilities.get("negative", 0.0))

    raw_score = pos_prob - neg_prob
    clamped = float(np.clip(raw_score, -1.0, 1.0))
    return round(clamped, 4)


def decision_function_to_probabilities(
    decision_scores: np.ndarray,
    classes: list[str],
) -> dict[str, float]:
    """Calibrate LinearSVC decision function margins into valid probabilities via Softmax.

    Linear Support Vector Classifiers output signed geometric distances (margins)
    to the separating hyperplanes rather than probabilities. Softmax with
    temperature scaling maps distances into a valid categorical probability distribution.

    Args:
        decision_scores: 1D array of decision values for each class.
        classes: Ordered list of class labels matching the decision scores.

    Returns:
        Dictionary mapping class label to normalized probability (sums to 1.0).
    """
    scores = np.asarray(decision_scores, dtype=float)
    # Subtract max for numerical stability (prevents overflow in exp)
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    sum_exp = np.sum(exp_scores)

    if sum_exp == 0 or np.isnan(sum_exp):
        uniform = 1.0 / len(classes)
        return {cls: uniform for cls in classes}

    probs = exp_scores / sum_exp
    return {
        cls: round(float(p), 4)
        for cls, p in zip(classes, probs)
    }


def extract_probabilities(pipeline: Any, text: str) -> dict[str, float]:
    """Extract or calibrate class probabilities for a single text using an sklearn Pipeline.

    Args:
        pipeline: Fitted sklearn Pipeline (with TfidfVectorizer + Classifier).
        text: Text string to evaluate.

    Returns:
        Dictionary of class probabilities for ['negative', 'neutral', 'positive'].
    """
    classes = list(pipeline.classes_)

    if hasattr(pipeline, "predict_proba"):
        # Logistic Regression or models with predict_proba
        raw_probs = pipeline.predict_proba([text])[0]
        return {
            cls: float(prob)
            for cls, prob in zip(classes, raw_probs)
        }

    if hasattr(pipeline, "decision_function"):
        # LinearSVC or margin-based classifiers
        decision = pipeline.decision_function([text])
        if decision.ndim > 1:
            decision = decision[0]
        return decision_function_to_probabilities(decision, classes)

    # Fallback if neither exists: predict class and assign discrete confidence
    pred_label = str(pipeline.predict([text])[0])
    return {
        cls: 1.0 if cls == pred_label else 0.0
        for cls in classes
    }
