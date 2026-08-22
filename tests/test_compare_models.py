import pandas as pd
from src.sentiment import compare_models

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
        for key in ["accuracy", "precision", "recall", "f1_score", "confusion_matrix", "classification_report", "train_size", "test_size"]:
            assert key in metrics
        # Accuracy should be a float between 0 and 1
        assert 0.0 <= metrics["accuracy"] <= 1.0
