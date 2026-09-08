"""SentimentScope AI brand, product, and topic entity detection package."""

from __future__ import annotations

from src.entity_detection.detector import (
    DEFAULT_DETECTOR,
    EntityDetector,
    detect_brand,
    detect_brands,
    detect_entities,
    detect_entities_dataframe,
    detect_product,
    detect_products,
    detect_topic,
    detect_topics,
)
from src.entity_detection.dictionaries import (
    BRAND_CATALOG,
    PRODUCT_CATALOG,
    TOPIC_TAXONOMY,
)
from src.entity_detection.normalizer import (
    GLOBAL_REGISTRY,
    PatternRegistry,
    build_phrase_regex,
    normalize_string,
)

__all__ = [
    "BRAND_CATALOG",
    "DEFAULT_DETECTOR",
    "EntityDetector",
    "GLOBAL_REGISTRY",
    "PRODUCT_CATALOG",
    "PatternRegistry",
    "TOPIC_TAXONOMY",
    "build_phrase_regex",
    "detect_brand",
    "detect_brands",
    "detect_entities",
    "detect_entities_dataframe",
    "detect_product",
    "detect_products",
    "detect_topic",
    "detect_topics",
    "normalize_string",
]
