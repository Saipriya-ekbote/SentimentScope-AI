"""SentimentScope AI data collection package."""

from __future__ import annotations

from config import (
    NORMALIZED_COLUMNS,
    PROVENANCE_IMPORTED,
    PROVENANCE_REAL,
    PROVENANCE_SYNTHETIC,
    SUPPORTED_PLATFORMS,
)
from src.data_collection.base import (
    BaseDataLoader,
    ConfigurationError,
    DataIngestionError,
    DEFAULT_COLUMN_MAPPINGS,
)
from src.data_collection.csv_loader import CSVLoader
from src.data_collection.instagram_loader import InstagramLoader
from src.data_collection.json_loader import JSONLoader
from src.data_collection.reddit_loader import RedditLoader
from src.data_collection.twitter_loader import TwitterLoader
from src.data_collection.youtube_loader import YouTubeLoader

__all__ = [
    "BaseDataLoader",
    "CSVLoader",
    "ConfigurationError",
    "DataIngestionError",
    "DEFAULT_COLUMN_MAPPINGS",
    "InstagramLoader",
    "JSONLoader",
    "NORMALIZED_COLUMNS",
    "PROVENANCE_IMPORTED",
    "PROVENANCE_REAL",
    "PROVENANCE_SYNTHETIC",
    "RedditLoader",
    "SUPPORTED_PLATFORMS",
    "TwitterLoader",
    "YouTubeLoader",
]
