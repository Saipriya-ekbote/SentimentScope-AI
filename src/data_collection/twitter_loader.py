"""Twitter/X data loader and API stub for SentimentScope AI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from config import PROVENANCE_IMPORTED
from src.data_collection.base import BaseDataLoader, ConfigurationError
from src.data_collection.csv_loader import CSVLoader
from src.data_collection.json_loader import JSONLoader


class TwitterLoader(BaseDataLoader):
    """Data loader for X/Twitter posts.
    
    Supports:
    1. Offline ingestion from exported Twitter CSV and JSON datasets.
    2. Documented stub interface for live Twitter API v2 ingestion.

    Required Credentials for Live API:
    - Environment Variable: `TWITTER_BEARER_TOKEN` (for App-Only read access)
    - Optional User Context: `TWITTER_API_KEY`, `TWITTER_API_SECRET`,
      `TWITTER_ACCESS_TOKEN`, `TWITTER_ACCESS_TOKEN_SECRET`
    - API Endpoint: `https://api.twitter.com/2/tweets/search/recent`
    - Documentation: https://developer.x.com/en/docs/twitter-api
    """

    def __init__(self, default_provenance: str = PROVENANCE_IMPORTED) -> None:
        super().__init__(default_platform="Twitter", default_provenance=default_provenance)

    def load_from_file(
        self,
        file_path: str | Path,
        provenance: str | None = None,
        column_mapping: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Load Twitter data from an exported CSV or JSON file."""
        path = Path(file_path)
        if path.suffix.lower() == ".json":
            loader: BaseDataLoader = JSONLoader(default_platform="Twitter", default_provenance=self.default_provenance)
        else:
            loader = CSVLoader(default_platform="Twitter", default_provenance=self.default_provenance)

        return loader.load(
            source=path,
            platform="Twitter",
            provenance=provenance or self.default_provenance,
            column_mapping=column_mapping,
        )

    def load_from_api(
        self,
        query: str,
        max_results: int = 100,
    ) -> pd.DataFrame:
        """Fetch live tweets from Twitter API v2.
        
        Raises:
            ConfigurationError: When TWITTER_BEARER_TOKEN is not configured in environment.
            NotImplementedError: To prevent mock/fake network calls without explicit user credentials.
        """
        bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        if not bearer_token:
            raise ConfigurationError(
                "Live X/Twitter API ingestion is not configured.\n"
                "To enable live collection, set the TWITTER_BEARER_TOKEN environment variable.\n"
                "For local development and testing, use load_from_file() with exported CSV/JSON data."
            )

        raise NotImplementedError(
            "Live Twitter API v2 ingestion driver is a stub in this environment. "
            "Install a verified HTTP client or tweepy to stream live endpoints."
        )

    def load(self, source: Any, **kwargs: Any) -> pd.DataFrame:
        """Universal entrypoint: loads from file path or checks API stub."""
        if isinstance(source, (str, Path)) and (Path(source).exists() or str(source).endswith((".csv", ".json"))):
            return self.load_from_file(source, **kwargs)
        return self.load_from_api(str(source), **kwargs)
