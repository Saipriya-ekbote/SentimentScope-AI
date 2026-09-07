"""Reddit data loader and API stub for SentimentScope AI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from config import PROVENANCE_IMPORTED
from src.data_collection.base import BaseDataLoader, ConfigurationError
from src.data_collection.csv_loader import CSVLoader
from src.data_collection.json_loader import JSONLoader


class RedditLoader(BaseDataLoader):
    """Data loader for Reddit submissions and comments.

    Supports:
    1. Offline ingestion from exported Reddit CSV and JSON datasets.
    2. Documented stub interface for live Reddit API (OAuth2 / PRAW) ingestion.

    Required Credentials for Live API:
    - Environment Variables:
      * `REDDIT_CLIENT_ID`: Script app client ID
      * `REDDIT_CLIENT_SECRET`: Script app client secret
      * `REDDIT_USER_AGENT`: Descriptive user agent string (e.g., 'SentimentScopeAI:v1.0 (by /u/...)')
    - API Endpoints: `https://oauth.reddit.com/r/{subreddit}/new` or `/r/{subreddit}/search`
    - Documentation: https://www.reddit.com/dev/api
    """

    def __init__(self, default_provenance: str = PROVENANCE_IMPORTED) -> None:
        super().__init__(default_platform="Reddit", default_provenance=default_provenance)

    def load_from_file(
        self,
        file_path: str | Path,
        provenance: str | None = None,
        column_mapping: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Load Reddit data from an exported CSV or JSON file."""
        path = Path(file_path)
        if path.suffix.lower() == ".json":
            loader: BaseDataLoader = JSONLoader(default_platform="Reddit", default_provenance=self.default_provenance)
        else:
            loader = CSVLoader(default_platform="Reddit", default_provenance=self.default_provenance)

        return loader.load(
            source=path,
            platform="Reddit",
            provenance=provenance or self.default_provenance,
            column_mapping=column_mapping,
        )

    def load_from_api(
        self,
        subreddit: str,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch live submissions from Reddit API.

        Raises:
            ConfigurationError: When REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET is missing.
            NotImplementedError: To prevent mock network calls without verified credentials.
        """
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT")

        if not client_id or not client_secret:
            raise ConfigurationError(
                "Live Reddit API ingestion is not configured.\n"
                "To enable live collection, configure the following environment variables:\n"
                "  - REDDIT_CLIENT_ID\n"
                "  - REDDIT_CLIENT_SECRET\n"
                f"  - REDDIT_USER_AGENT (current: '{user_agent or 'not set'}')\n"
                "For offline evaluation, use load_from_file() with exported CSV/JSON data."
            )

        raise NotImplementedError(
            f"Live Reddit ingestion driver for r/{subreddit} is a stub in this environment. "
            "Configure 'praw' or an authenticated session to fetch live posts."
        )

    def load(self, source: Any, **kwargs: Any) -> pd.DataFrame:
        """Universal entrypoint: loads from file path or checks API stub."""
        if isinstance(source, (str, Path)) and (Path(source).exists() or str(source).endswith((".csv", ".json"))):
            return self.load_from_file(source, **kwargs)
        return self.load_from_api(str(source), **kwargs)
