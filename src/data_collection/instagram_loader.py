"""Instagram data loader and API stub for SentimentScope AI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from config import PROVENANCE_IMPORTED
from src.data_collection.base import BaseDataLoader, ConfigurationError
from src.data_collection.csv_loader import CSVLoader
from src.data_collection.json_loader import JSONLoader


class InstagramLoader(BaseDataLoader):
    """Data loader for Instagram media captions and comments.

    Supports:
    1. Offline ingestion from exported Instagram CSV and JSON datasets.
    2. Documented stub interface for live Meta Graph API (Instagram Graph API) ingestion.

    Required Credentials for Live API:
    - Environment Variables:
      * `INSTAGRAM_ACCESS_TOKEN`: Long-lived Meta User or Page Access Token
      * `INSTAGRAM_BUSINESS_ACCOUNT_ID`: Connected Instagram Business / Creator Account ID
    - API Endpoints: `https://graph.facebook.com/v19.0/{ig-user-id}/media`
    - Documentation: https://developers.facebook.com/docs/instagram-api
    """

    def __init__(self, default_provenance: str = PROVENANCE_IMPORTED) -> None:
        super().__init__(default_platform="Instagram", default_provenance=default_provenance)

    def load_from_file(
        self,
        file_path: str | Path,
        provenance: str | None = None,
        column_mapping: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Load Instagram data from an exported CSV or JSON file."""
        path = Path(file_path)
        if path.suffix.lower() == ".json":
            loader: BaseDataLoader = JSONLoader(default_platform="Instagram", default_provenance=self.default_provenance)
        else:
            loader = CSVLoader(default_platform="Instagram", default_provenance=self.default_provenance)

        return loader.load(
            source=path,
            platform="Instagram",
            provenance=provenance or self.default_provenance,
            column_mapping=column_mapping,
        )

    def load_from_api(
        self,
        hashtag_or_account: str,
        limit: int = 50,
    ) -> pd.DataFrame:
        """Fetch live media from Instagram Graph API.

        Raises:
            ConfigurationError: When INSTAGRAM_ACCESS_TOKEN or account ID is missing.
            NotImplementedError: To prevent mock network calls without verified credentials.
        """
        access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN")
        account_id = os.getenv("INSTAGRAM_BUSINESS_ACCOUNT_ID")

        if not access_token or not account_id:
            raise ConfigurationError(
                "Live Instagram Graph API ingestion is not configured.\n"
                "To enable live collection, configure the following environment variables:\n"
                "  - INSTAGRAM_ACCESS_TOKEN\n"
                "  - INSTAGRAM_BUSINESS_ACCOUNT_ID\n"
                "For local testing and evaluation, use load_from_file() with exported CSV/JSON data."
            )

        raise NotImplementedError(
            f"Live Instagram Graph API ingestion driver for '{hashtag_or_account}' is a stub in this environment."
        )

    def load(self, source: Any, **kwargs: Any) -> pd.DataFrame:
        """Universal entrypoint: loads from file path or checks API stub."""
        if isinstance(source, (str, Path)) and (Path(source).exists() or str(source).endswith((".csv", ".json"))):
            return self.load_from_file(source, **kwargs)
        return self.load_from_api(str(source), **kwargs)
