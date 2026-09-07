"""Tests for Phase 2 modular data ingestion and normalized schema."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from config import (
    NORMALIZED_COLUMNS,
    PROVENANCE_IMPORTED,
    PROVENANCE_REAL,
    PROVENANCE_SYNTHETIC,
)
from src.data_collection import (
    CSVLoader,
    ConfigurationError,
    DataIngestionError,
    InstagramLoader,
    JSONLoader,
    RedditLoader,
    TwitterLoader,
    YouTubeLoader,
)
from src.data_loader import load_normalized_data


# --- CSV Loader Tests ---

def test_csv_loader_valid(tmp_path: Path) -> None:
    csv_file = tmp_path / "valid.csv"
    pd.DataFrame(
        {
            "post_id": ["p1", "p2"],
            "platform": ["Twitter", "Reddit"],
            "text": ["Love the new battery", "Great update"],
            "timestamp": ["2026-08-01 10:00:00", "2026-08-01 11:00:00"],
            "author": ["alice", "bob"],
            "brand": ["Apple", "Google"],
            "sentiment": ["positive", "positive"],
        }
    ).to_csv(csv_file, index=False)

    loader = CSVLoader()
    df = loader.load(csv_file, provenance=PROVENANCE_IMPORTED)

    assert len(df) == 2
    assert list(df.columns) == NORMALIZED_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert (df["provenance"] == PROVENANCE_IMPORTED).all()
    assert df.loc[0, "brand"] == "Apple"
    assert df.loc[1, "platform"] == "Reddit"


def test_csv_loader_missing_file() -> None:
    loader = CSVLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("non_existent_file.csv")


def test_csv_loader_missing_text_column(tmp_path: Path) -> None:
    csv_file = tmp_path / "missing_text.csv"
    pd.DataFrame(
        {
            "post_id": ["p1"],
            "timestamp": ["2026-08-01 10:00:00"],
        }
    ).to_csv(csv_file, index=False)

    loader = CSVLoader()
    with pytest.raises(DataIngestionError, match="lacks a 'text'"):
        loader.load(csv_file)


def test_csv_loader_custom_column_mapping(tmp_path: Path) -> None:
    csv_file = tmp_path / "custom_cols.csv"
    pd.DataFrame(
        {
            "tweet_num": ["101"],
            "msg_body": ["Delayed flight once again."],
            "time_sent": ["2026-08-02 09:30:00"],
            "airline_tag": ["Delta"],
        }
    ).to_csv(csv_file, index=False)

    loader = CSVLoader()
    df = loader.load(
        csv_file,
        platform="Twitter",
        column_mapping={
            "tweet_num": "post_id",
            "msg_body": "text",
            "time_sent": "timestamp",
            "airline_tag": "brand",
        },
    )

    assert len(df) == 1
    assert df.loc[0, "post_id"] == "101"
    assert df.loc[0, "text"] == "Delayed flight once again."
    assert df.loc[0, "brand"] == "Delta"
    assert df.loc[0, "platform"] == "Twitter"


def test_csv_loader_filters_empty_text_and_bad_timestamps(tmp_path: Path) -> None:
    csv_file = tmp_path / "dirty.csv"
    pd.DataFrame(
        {
            "post_id": ["p1", "p2", "p3"],
            "text": ["Valid post", "   ", "Another valid post"],
            "timestamp": ["2026-08-01 10:00:00", "2026-08-01 11:00:00", "invalid-date"],
        }
    ).to_csv(csv_file, index=False)

    loader = CSVLoader()
    df = loader.load(csv_file)
    assert len(df) == 1
    assert df.loc[0, "post_id"] == "p1"


# --- JSON Loader Tests ---

def test_json_loader_array_format(tmp_path: Path) -> None:
    json_file = tmp_path / "posts.json"
    records = [
        {"post_id": "j1", "text": "Excited about the launch!", "timestamp": "2026-08-03 14:00:00", "platform": "YouTube"},
        {"post_id": "j2", "text": "Sound quality is disappointing", "timestamp": "2026-08-03 15:30:00", "platform": "YouTube"},
    ]
    json_file.write_text(json.dumps(records), encoding="utf-8")

    loader = JSONLoader()
    df = loader.load(json_file, provenance=PROVENANCE_REAL)

    assert len(df) == 2
    assert list(df.columns) == NORMALIZED_COLUMNS
    assert (df["provenance"] == PROVENANCE_REAL).all()
    assert (df["platform"] == "YouTube").all()


def test_json_loader_nested_data_key(tmp_path: Path) -> None:
    json_file = tmp_path / "nested.json"
    payload = {
        "status": "success",
        "data": [
            {"id": "n1", "content": "Camera is stunning", "created_at": "2026-08-04 12:00:00"}
        ],
    }
    json_file.write_text(json.dumps(payload), encoding="utf-8")

    loader = JSONLoader()
    df = loader.load(json_file)

    assert len(df) == 1
    assert df.loc[0, "post_id"] == "n1"
    assert df.loc[0, "text"] == "Camera is stunning"


def test_json_loader_json_lines_format(tmp_path: Path) -> None:
    jsonl_file = tmp_path / "feed.jsonl"
    lines = (
        '{"id": "l1", "text": "Line 1 post", "timestamp": "2026-08-01 08:00:00"}\n'
        '{"id": "l2", "text": "Line 2 post", "timestamp": "2026-08-01 09:00:00"}\n'
    )
    jsonl_file.write_text(lines, encoding="utf-8")

    loader = JSONLoader()
    df = loader.load(jsonl_file)
    assert len(df) == 2
    assert df["post_id"].tolist() == ["l1", "l2"]


def test_json_loader_malformed_json(tmp_path: Path) -> None:
    bad_json = tmp_path / "corrupt.json"
    bad_json.write_text("{this is not valid json", encoding="utf-8")

    loader = JSONLoader()
    with pytest.raises(DataIngestionError, match="Malformed JSON"):
        loader.load(bad_json)


def test_json_loader_empty_file(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("   ", encoding="utf-8")

    loader = JSONLoader()
    with pytest.raises(DataIngestionError, match="empty"):
        loader.load(empty_file)


# --- Platform Loader Stubs & Provenance Tests ---

def test_twitter_loader_api_stub_requires_credentials() -> None:
    loader = TwitterLoader()
    with pytest.raises(ConfigurationError, match="Live X/Twitter API ingestion is not configured"):
        loader.load_from_api(query="from:Apple")


def test_reddit_loader_api_stub_requires_credentials() -> None:
    loader = RedditLoader()
    with pytest.raises(ConfigurationError, match="Live Reddit API ingestion is not configured"):
        loader.load_from_api(subreddit="technology")


def test_instagram_loader_api_stub_requires_credentials() -> None:
    loader = InstagramLoader()
    with pytest.raises(ConfigurationError, match="Live Instagram Graph API ingestion is not configured"):
        loader.load_from_api(hashtag_or_account="apple")


def test_youtube_loader_api_stub_requires_credentials() -> None:
    loader = YouTubeLoader()
    with pytest.raises(ConfigurationError, match="Live YouTube Data API v3 ingestion is not configured"):
        loader.load_from_api(video_id="video123")


def test_platform_loader_loads_from_file(tmp_path: Path) -> None:
    reddit_file = tmp_path / "reddit_posts.csv"
    pd.DataFrame(
        {
            "id": ["r1"],
            "text": ["Reddit discussion on smartphone prices"],
            "timestamp": ["2026-08-05 10:00:00"],
        }
    ).to_csv(reddit_file, index=False)

    reddit_loader = RedditLoader()
    df = reddit_loader.load_from_file(reddit_file)

    assert len(df) == 1
    assert df.loc[0, "platform"] == "Reddit"
    assert list(df.columns) == NORMALIZED_COLUMNS


def test_provenance_marking_synthetic(tmp_path: Path) -> None:
    demo_file = tmp_path / "demo.csv"
    pd.DataFrame(
        {
            "id": ["d1"],
            "text": ["Demo sample post"],
            "timestamp": ["2026-08-01 12:00:00"],
        }
    ).to_csv(demo_file, index=False)

    df = load_normalized_data(demo_file, provenance=PROVENANCE_SYNTHETIC)
    assert df.loc[0, "provenance"] == "Synthetic Demo"


# --- Sample Multi-Platform Dataset Verification ---

def test_multi_platform_demo_dataset_conforms_to_schema() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sample_csv = project_root / "data" / "sample" / "multi_platform_demo.csv"

    assert sample_csv.exists(), f"Sample dataset missing at {sample_csv}"

    loader = CSVLoader()
    df = loader.load(sample_csv, provenance=PROVENANCE_SYNTHETIC)

    # 1. Column compliance
    assert list(df.columns) == NORMALIZED_COLUMNS

    # 2. Row count and no empty required values
    assert len(df) >= 50
    assert df["post_id"].notna().all()
    assert df["text"].notna().all()
    assert df["timestamp"].notna().all()

    # 3. Multiple platforms present
    platforms = set(df["platform"].unique())
    assert {"Twitter", "Reddit", "Instagram", "YouTube"}.issubset(platforms)

    # 4. Multiple brands present
    brands = set(df["brand"].dropna().unique())
    assert {"Apple", "Samsung", "Delta", "Amazon"}.issubset(brands)

    # 5. Provenance is explicitly 'Synthetic Demo'
    assert (df["provenance"] == PROVENANCE_SYNTHETIC).all()
