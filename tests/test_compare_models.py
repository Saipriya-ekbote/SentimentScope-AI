import numpy as np
import pandas as pd
import pytest

from src.sentiment import SENTIMENT_LABELS, compare_models


def test_compare_models_basic():
    # Create a tiny dataset with clear sentiment labels
    data = {
        "text": [
            "I love this product",
            "This is terrible",
            "It is okay",
            "Absolutely fantastic!",
            "Worst experience ever",
            "Mediocre at best",
        ],
        "sentiment": [
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative",
            "neutral",
        ],
    }
    df = pd.DataFrame(data)
    results = compare_models(df, test_size=0.5)
    # Expect both models to be present
    assert "Logistic Regression" in results
    assert "Linear SVM" in results
    for model_name, metrics in results.items():
        # Basic metric keys should exist
        for key in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "confusion_matrix",
            "classification_report",
            "train_size",
            "test_size",
        ]:
            assert key in metrics
        # Accuracy should be a float between 0 and 1
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0
        assert metrics["train_size"] == 3
        assert metrics["test_size"] == 3


def test_compare_models_train_test_sizes(training_df: pd.DataFrame):
    total = len(training_df)
    results = compare_models(training_df, test_size=0.25)
    lr_metrics = results["Logistic Regression"]
    svm_metrics = results["Linear SVM"]

    expected_test_size = int(np.ceil(total * 0.25))
    expected_train_size = total - expected_test_size

    assert lr_metrics["train_size"] == expected_train_size
    assert lr_metrics["test_size"] == expected_test_size
    assert svm_metrics["train_size"] == expected_train_size
    assert svm_metrics["test_size"] == expected_test_size
    assert lr_metrics["train_size"] + lr_metrics["test_size"] == total


def test_compare_models_reproducibility(training_df: pd.DataFrame):
    run_1 = compare_models(training_df, test_size=0.2)
    run_2 = compare_models(training_df, test_size=0.2)

    for model_name in ["Logistic Regression", "Linear SVM"]:
        assert run_1[model_name]["accuracy"] == run_2[model_name]["accuracy"]
        assert run_1[model_name]["precision"] == run_2[model_name]["precision"]
        assert run_1[model_name]["recall"] == run_2[model_name]["recall"]
        assert run_1[model_name]["f1_score"] == run_2[model_name]["f1_score"]
        assert run_1[model_name]["confusion_matrix"] == run_2[model_name]["confusion_matrix"]
        assert run_1[model_name]["classification_report"] == run_2[model_name]["classification_report"]


def test_compare_models_stratification_preserves_distribution(training_df: pd.DataFrame):
    # training_df has 20 positive, 20 negative, 20 neutral (total 60)
    results = compare_models(training_df, test_size=0.2)
    for model_name in ["Logistic Regression", "Linear SVM"]:
        assert results[model_name]["train_size"] == 48
        assert results[model_name]["test_size"] == 12
        cm = results[model_name]["confusion_matrix"]
        # Confusion matrix is 3x3 with 4 test samples per class (12 total)
        total_test_samples_in_cm = sum(sum(row) for row in cm)
        assert total_test_samples_in_cm == 12


def test_compare_models_both_models_evaluated_on_same_split(training_df: pd.DataFrame):
    results = compare_models(training_df, test_size=0.2)
    lr = results["Logistic Regression"]
    svm = results["Linear SVM"]

    assert lr["train_size"] == svm["train_size"]
    assert lr["test_size"] == svm["test_size"]
    assert len(lr["confusion_matrix"]) == len(svm["confusion_matrix"])


def test_compare_models_input_validation():
    # Non-DataFrame input
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        compare_models("not a dataframe")  # type: ignore

    # Empty DataFrame
    with pytest.raises(ValueError, match="Cannot train or evaluate models on an empty DataFrame"):
        compare_models(pd.DataFrame())

    # Missing text or sentiment columns
    with pytest.raises(ValueError, match="Training DataFrame must include"):
        compare_models(pd.DataFrame({"text": ["a", "b"]}))
    with pytest.raises(ValueError, match="Training DataFrame must include"):
        compare_models(pd.DataFrame({"sentiment": ["positive", "negative"]}))

    # Invalid test_size
    valid_df = pd.DataFrame(
        {
            "text": ["good", "bad", "okay", "great"],
            "sentiment": ["positive", "negative", "neutral", "positive"],
        }
    )
    with pytest.raises(ValueError, match="test_size"):
        compare_models(valid_df, test_size=0.0)
    with pytest.raises(ValueError, match="test_size"):
        compare_models(valid_df, test_size=1.0)
    with pytest.raises(ValueError, match="test_size"):
        compare_models(valid_df, test_size=-0.2)
    with pytest.raises(ValueError, match="test_size"):
        compare_models(valid_df, test_size="invalid")  # type: ignore

    # Insufficient samples (< 2)
    with pytest.raises(ValueError, match="at least two samples"):
        compare_models(pd.DataFrame({"text": ["hi"], "sentiment": ["positive"]}))

    # Single class
    with pytest.raises(ValueError, match="at least two sentiment classes"):
        compare_models(
            pd.DataFrame(
                {"text": ["hi", "hello"], "sentiment": ["positive", "positive"]}
            )
        )

    # Empty/whitespace text column
    with pytest.raises(ValueError, match="Text column contains only empty or whitespace strings"):
        compare_models(
            pd.DataFrame(
                {"text": ["  ", " "], "sentiment": ["positive", "negative"]}
            )
        )
