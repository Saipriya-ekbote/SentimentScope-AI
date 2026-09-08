"""Modular Brand, Product, and Topic detection engine for SentimentScope AI."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.entity_detection.dictionaries import (
    BRAND_CATALOG,
    PRODUCT_CATALOG,
    TOPIC_TAXONOMY,
)
from src.entity_detection.normalizer import GLOBAL_REGISTRY, normalize_string


class EntityDetector:
    """Configurable rule-based entity and topic detector.

    Identifies brands, specific products, and operational topics using
    boundary-enforced regex pattern matching and alias resolution.
    """

    def __init__(
        self,
        brand_catalog: dict[str, list[str]] | None = None,
        product_catalog: dict[str, list[str]] | None = None,
        topic_taxonomy: dict[str, list[str]] | None = None,
    ) -> None:
        self.brand_catalog = brand_catalog if brand_catalog is not None else BRAND_CATALOG
        self.product_catalog = product_catalog if product_catalog is not None else PRODUCT_CATALOG
        self.topic_taxonomy = topic_taxonomy if topic_taxonomy is not None else TOPIC_TAXONOMY

        # Build reverse product-to-brand lookup map
        self._product_to_brand: dict[str, str] = {}
        for brand, products in self.product_catalog.items():
            for product in products:
                self._product_to_brand[product] = brand

    def detect_brands(self, text: str) -> list[str]:
        """Detect all matching canonical brand names in text.

        Returns:
            List of unique canonical brand names in order of appearance.
        """
        if not text:
            return []

        clean_text = normalize_string(text)
        detected: list[str] = []

        for canonical_brand, aliases in self.brand_catalog.items():
            for alias in aliases:
                pattern = GLOBAL_REGISTRY.get(alias)
                if pattern.search(clean_text):
                    if canonical_brand not in detected:
                        detected.append(canonical_brand)
                    break  # Matched this brand, continue to next

        return detected

    def detect_brand(self, text: str) -> str | None:
        """Detect the primary canonical brand name in text.

        If direct brand alias is not mentioned but a specific product is detected,
        infers the brand from the product catalog.
        """
        brands = self.detect_brands(text)
        if brands:
            return brands[0]

        # Inferred fallback: check if a known product is mentioned
        products = self.detect_products(text)
        if products:
            inferred_brand = self._product_to_brand.get(products[0])
            if inferred_brand:
                return inferred_brand

        return None

    def detect_products(self, text: str, brand: str | None = None) -> list[str]:
        """Detect all matching products in text, preferring more specific names.

        Args:
            text: Input text string.
            brand: Optional brand filter. If specified, restricts product search
                   to that brand's catalog.

        Returns:
            List of matching product names ordered from most specific to general.
        """
        if not text:
            return []

        clean_text = normalize_string(text)
        detected: list[str] = []

        # Determine which brand catalogs to search
        catalogs_to_search: list[tuple[str, list[str]]]
        if brand and brand in self.product_catalog:
            catalogs_to_search = [(brand, self.product_catalog[brand])]
        else:
            catalogs_to_search = list(self.product_catalog.items())

        for _, products in catalogs_to_search:
            for product in products:
                pattern = GLOBAL_REGISTRY.get(product)
                if pattern.search(clean_text):
                    if product not in detected:
                        detected.append(product)

        # Sort matches so longer/more specific product names appear first
        detected.sort(key=len, reverse=True)
        return detected

    def detect_product(self, text: str, brand: str | None = None) -> str | None:
        """Detect the most specific matching product in text."""
        products = self.detect_products(text, brand=brand)
        return products[0] if products else None

    def detect_topics(self, text: str) -> list[str]:
        """Detect all matching canonical topic/aspect categories in text.

        Returns:
            List of unique canonical topic names.
        """
        if not text:
            return []

        clean_text = normalize_string(text)
        detected: list[str] = []

        for canonical_topic, triggers in self.topic_taxonomy.items():
            for trigger in triggers:
                pattern = GLOBAL_REGISTRY.get(trigger)
                if pattern.search(clean_text):
                    if canonical_topic not in detected:
                        detected.append(canonical_topic)
                    break

        return detected

    def detect_topic(self, text: str) -> str | None:
        """Detect the primary matching canonical topic/aspect in text."""
        topics = self.detect_topics(text)
        return topics[0] if topics else None

    def detect_entities(self, text: str) -> dict[str, Any]:
        """Identify all brand, product, and topic entities from text.

        Returns:
            Dictionary with both primary entity assignments and full lists:
            {
                'brand': primary_brand,
                'product': primary_product,
                'topic': primary_topic,
                'brands': [all detected brands],
                'products': [all detected products],
                'topics': [all detected topics],
            }
        """
        brands = self.detect_brands(text)
        primary_brand = brands[0] if brands else None

        products = self.detect_products(text, brand=primary_brand)
        primary_product = products[0] if products else None

        # If brand was not found in text, infer it from the product
        if primary_brand is None and primary_product is not None:
            primary_brand = self._product_to_brand.get(primary_product)
            if primary_brand and primary_brand not in brands:
                brands.append(primary_brand)

        topics = self.detect_topics(text)
        primary_topic = topics[0] if topics else None

        return {
            "brand": primary_brand,
            "product": primary_product,
            "topic": primary_topic,
            "brands": brands,
            "products": products,
            "topics": topics,
        }

    def detect_entities_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """Enrich a pandas DataFrame by populating 'brand', 'product', and 'topic'.

        Args:
            df: Input pandas DataFrame (e.g., from Phase 2 normalized schema).
            text_column: Column containing post text.
            overwrite: If False, keeps existing non-empty values in brand/product/topic.

        Returns:
            DataFrame with enriched entity columns, preserving all original columns.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in DataFrame.")

        enriched = df.copy()

        # Ensure target columns exist
        for col in ["brand", "product", "topic"]:
            if col not in enriched.columns:
                enriched[col] = None

        new_brands: list[str | None] = []
        new_products: list[str | None] = []
        new_topics: list[str | None] = []

        for idx, row in enriched.iterrows():
            text_val = str(row[text_column]) if pd.notna(row[text_column]) else ""
            detection = self.detect_entities(text_val)

            # Brand assignment
            curr_brand = row.get("brand")
            if not overwrite and pd.notna(curr_brand) and str(curr_brand).strip():
                new_brands.append(str(curr_brand).strip())
            else:
                new_brands.append(detection["brand"])

            # Product assignment
            curr_prod = row.get("product")
            if not overwrite and pd.notna(curr_prod) and str(curr_prod).strip():
                new_products.append(str(curr_prod).strip())
            else:
                new_products.append(detection["product"])

            # Topic assignment
            curr_topic = row.get("topic")
            if not overwrite and pd.notna(curr_topic) and str(curr_topic).strip():
                new_topics.append(str(curr_topic).strip())
            else:
                new_topics.append(detection["topic"])

        enriched["brand"] = new_brands
        enriched["product"] = new_products
        enriched["topic"] = new_topics
        return enriched


# Default singleton instance for simple functional imports
DEFAULT_DETECTOR = EntityDetector()

detect_brand = DEFAULT_DETECTOR.detect_brand
detect_brands = DEFAULT_DETECTOR.detect_brands
detect_product = DEFAULT_DETECTOR.detect_product
detect_products = DEFAULT_DETECTOR.detect_products
detect_topic = DEFAULT_DETECTOR.detect_topic
detect_topics = DEFAULT_DETECTOR.detect_topics
detect_entities = DEFAULT_DETECTOR.detect_entities
detect_entities_dataframe = DEFAULT_DETECTOR.detect_entities_dataframe
