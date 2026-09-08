"""Tests for brand, product, and topic entity detection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_collection import CSVLoader
from src.entity_detection import (
    detect_brand,
    detect_brands,
    detect_entities,
    detect_entities_dataframe,
    detect_product,
    detect_products,
    detect_topic,
    detect_topics,
)


# --- Brand Detection Tests ---

def test_brand_detection_basic() -> None:
    assert detect_brand("Apple iPhone battery is amazing") == "Apple"
    assert detect_brand("Samsung phone display is excellent") == "Samsung"
    assert detect_brand("Delta flight was delayed") == "Delta"
    assert detect_brand("Amazon delivery arrived early") == "Amazon"


def test_brand_detection_inferred_from_product() -> None:
    # No explicit word 'Apple', but product 'iPad' is mentioned
    assert detect_brand("My iPad has great battery life") == "Apple"
    # No explicit word 'Samsung', but 'Galaxy S25' is mentioned
    assert detect_brand("The new Galaxy S25 display is breathtaking") == "Samsung"
    # No explicit word 'Amazon', but 'Echo Show' is mentioned
    assert detect_brand("Setting up my Echo Show in the living room") == "Amazon"


def test_brand_detection_case_insensitivity() -> None:
    assert detect_brand("APPLE customer service") == "Apple"
    assert detect_brand("samsung galaxy camera") == "Samsung"
    assert detect_brand("DELTA sky club was great") == "Delta"


def test_brand_detection_word_boundary_avoids_false_positives() -> None:
    # 'application' should not trigger 'Apple'
    assert detect_brand("Please submit your job application online.") is None
    # 'pineapples' should not trigger 'Apple'
    assert detect_brand("I bought fresh pineapples at the market.") is None


# --- Product Detection Tests ---

def test_product_detection_specificity_preference() -> None:
    # 'iPhone 16' must be preferred over 'iPhone'
    assert detect_product("iPhone 16 camera is amazing") == "iPhone 16"

    # 'Galaxy S25 Ultra' must be preferred over 'Galaxy S25' or 'Galaxy'
    assert detect_product("Testing the Galaxy S25 Ultra optical zoom") == "Galaxy S25 Ultra"

    # 'AirPods Pro' must be preferred over 'AirPods'
    assert detect_product("Active noise cancellation on AirPods Pro") == "AirPods Pro"


def test_product_detection_with_brand_filter() -> None:
    text = "Comparing the Galaxy S25 against the iPhone 16"
    assert detect_product(text, brand="Samsung") == "Galaxy S25"
    assert detect_product(text, brand="Apple") == "iPhone 16"


def test_product_detection_general_fallback() -> None:
    assert detect_product("I just bought a new iPhone yesterday") == "iPhone"
    assert detect_product("My flight with Delta was comfortable") == "Flight"


# --- Topic / Aspect Detection Tests ---

def test_topic_detection_product_aspects() -> None:
    assert detect_topic("battery life is terrible") == "battery"
    assert detect_topic("The camera shutter is fast and photos are clear") == "camera"
    assert detect_topic("The OLED display brightness in sunlight is great") == "display"
    assert detect_topic("Very fast processor speed and high benchmark scores") == "performance"
    assert detect_topic("The price is too high and expensive") == "pricing"
    assert detect_topic("The titanium build quality feels premium") == "quality"
    assert detect_topic("Latest software update fixed the annoying bug") == "software"


def test_topic_detection_service_aspects() -> None:
    assert detect_topic("customer service was excellent") == "customer_service"
    assert detect_topic("Same day prime delivery arrived on my doorstep") == "delivery"
    assert detect_topic("My flight was delayed by 3 hours") == "delay"
    assert detect_topic("Flight cancellation grounded all planes") == "cancellation"
    assert detect_topic("Waiting at baggage claim for my luggage") == "baggage"
    assert detect_topic("Nationwide computer outage shut down all dispatch systems") == "outage"
    assert detect_topic("I received a full refund and flight credit") == "refund"


def test_topic_alias_normalization() -> None:
    # Multiple aliases mapping to customer_service
    assert detect_topic("The support team helped me quickly") == "customer_service"
    assert detect_topic("Help desk agent answered my call") == "customer_service"

    # Multiple aliases mapping to delay
    assert detect_topic("Sitting on the tarmac waiting for departure") == "delay"
    assert detect_topic("Severe weather delay at Atlanta airport") == "delay"


# --- Multiple Entity & Negative Tests ---

def test_multiple_topics_captured() -> None:
    text = "Camera is amazing but battery life is terrible"
    topics = detect_topics(text)
    assert "camera" in topics
    assert "battery" in topics
    assert len(topics) >= 2


def test_detect_entities_full_structure() -> None:
    text = "Samsung Galaxy S25 has an amazing camera but terrible battery life."
    res = detect_entities(text)

    assert res["brand"] == "Samsung"
    assert res["product"] == "Galaxy S25"
    assert res["topic"] in {"camera", "battery"}
    assert "camera" in res["topics"]
    assert "battery" in res["topics"]
    assert "Samsung" in res["brands"]
    assert "Galaxy S25" in res["products"]


def test_negative_cases_unrelated_text() -> None:
    text = "The weather was partly cloudy this morning."
    res = detect_entities(text)
    assert res["brand"] is None
    assert res["product"] is None
    assert res["topic"] is None
    assert res["brands"] == []
    assert res["products"] == []
    assert res["topics"] == []


# --- DataFrame Integration Tests ---

def test_detect_entities_dataframe_preserves_schema_and_populates() -> None:
    df = pd.DataFrame(
        {
            "post_id": ["1", "2", "3"],
            "platform": ["Twitter", "Reddit", "Instagram"],
            "timestamp": ["2026-08-01", "2026-08-02", "2026-08-03"],
            "text": [
                "Apple iPhone 16 battery is outstanding",
                "Delta flight was delayed 4 hours",
                "Just regular morning routine with coffee",
            ],
            "sentiment": ["positive", "negative", "neutral"],
            "sentiment_score": [0.85, -0.90, 0.0],
        }
    )

    enriched = detect_entities_dataframe(df)

    # 1. Row count and columns preserved
    assert len(enriched) == 3
    for col in ["post_id", "platform", "timestamp", "text", "sentiment", "sentiment_score"]:
        assert col in enriched.columns
        assert enriched[col].tolist() == df[col].tolist()

    # 2. Entity columns populated
    assert enriched.loc[0, "brand"] == "Apple"
    assert enriched.loc[0, "product"] == "iPhone 16"
    assert enriched.loc[0, "topic"] == "battery"

    assert enriched.loc[1, "brand"] == "Delta"
    assert enriched.loc[1, "topic"] == "delay"

    assert pd.isna(enriched.loc[2, "brand"]) or enriched.loc[2, "brand"] is None
    assert pd.isna(enriched.loc[2, "topic"]) or enriched.loc[2, "topic"] is None



def test_detect_entities_dataframe_preserves_existing_annotations() -> None:
    df = pd.DataFrame(
        {
            "post_id": ["1"],
            "text": ["iPhone 16 camera is great"],
            "brand": ["CustomBrand"],  # Already annotated
            "product": [None],
            "topic": [None],
        }
    )

    # When overwrite=False, existing brand should be preserved
    res = detect_entities_dataframe(df, overwrite=False)
    assert res.loc[0, "brand"] == "CustomBrand"
    assert res.loc[0, "product"] == "iPhone 16"
    assert res.loc[0, "topic"] == "camera"

    # When overwrite=True, brand should be overwritten by detected 'Apple'
    res_overwrite = detect_entities_dataframe(df, overwrite=True)
    assert res_overwrite.loc[0, "brand"] == "Apple"


# --- Multi-Platform Demo Dataset Integration Test ---

def test_multi_platform_demo_dataset_entity_detection() -> None:
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "sample" / "multi_platform_demo.csv"

    assert csv_path.exists()
    loader = CSVLoader()
    df = loader.load(csv_path)

    enriched = detect_entities_dataframe(df, overwrite=True)

    # Validate that entity detection runs across multi-platform rows
    assert len(enriched) == len(df)
    brands = set(enriched["brand"].dropna().unique())
    assert {"Apple", "Samsung", "Delta", "Amazon"}.issubset(brands)

    topics = set(enriched["topic"].dropna().unique())
    assert {"battery", "camera", "customer_service", "delay", "outage"}.intersection(topics)
