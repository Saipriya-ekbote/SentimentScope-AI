"""YouTube data loader and API stub for SentimentScope AI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from config import PROVENANCE_IMPORTED
from src.data_collection.base import BaseDataLoader, ConfigurationError
from src.data_collection.csv_loader import CSVLoader
from src.data_collection.json_loader import JSONLoader


class YouTubeLoader(BaseDataLoader):
    """Data loader for YouTube video comments and descriptions.

    Supports:
    1. Offline ingestion from exported YouTube CSV and JSON datasets.
    2. Documented stub interface for live YouTube Data API v3 ingestion.

    Required Credentials for Live API:
    - Environment Variable: `YOUTUBE_API_KEY` (Google Cloud Console API key with YouTube Data API v3 enabled)
    - API Endpoints: `https://www.googleapis.com/youtube/v3/commentThreads`
    - Documentation: https://developers.google.com/youtube/v3/docs/commentThreads/list
    """

    def __init__(self, default_provenance: str = PROVENANCE_IMPORTED) -> None:
        super().__init__(default_platform="YouTube", default_provenance=default_provenance)

    def load_from_file(
        self,
        file_path: str | Path,
        provenance: str | None = None,
        column_mapping: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Load YouTube data from an exported CSV or JSON file."""
        path = Path(file_path)
        if path.suffix.lower() == ".json":
            loader: BaseDataLoader = JSONLoader(default_platform="YouTube", default_provenance=self.default_provenance)
        else:
            loader = CSVLoader(default_platform="YouTube", default_provenance=self.default_provenance)

        return loader.load(
            source=path,
            platform="YouTube",
            provenance=provenance or self.default_provenance,
            column_mapping=column_mapping,
        )

    def load_from_api(
        self,
        video_id: str,
        max_comments: int = 100,
    ) -> pd.DataFrame:
        """Fetch live comments for a YouTube video using the Data API v3.

        Raises:
            ConfigurationError: When YOUTUBE_API_KEY is missing from environment.
            NotImplementedError: To prevent mock network calls without verified credentials.
        """
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "Live YouTube Data API v3 ingestion is not configured.\n"
                "To enable live collection, configure the YOUTUBE_API_KEY environment variable.\n"
                "For local testing and evaluation, use load_from_file() with exported CSV/JSON data."
            )

        raise NotImplementedError(
            f"Live YouTube comment thread ingestion driver for video '{video_id}' is a stub in this environment."
        )

    def load(self, source: Any, **kwargs: Any) -> pd.DataFrame:
        """Universal entrypoint: loads from file path or checks API stub."""
        if isinstance(source, (str, Path)) and (Path(source).exists() or str(source).endswith((".csv", ".json"))):
            return self.load_from_file(source, **kwargs)
        return self.load_from_api(str(source), **kwargs)
