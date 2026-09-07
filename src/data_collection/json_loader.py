"""JSON data loader for SentimentScope AI multi-platform ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import PROVENANCE_IMPORTED
from src.data_collection.base import BaseDataLoader, DataIngestionError


class JSONLoader(BaseDataLoader):
    """Loader for JSON-formatted social media datasets."""

    def __init__(
        self,
        default_platform: str = "Other",
        default_provenance: str = PROVENANCE_IMPORTED,
    ) -> None:
        super().__init__(
            default_platform=default_platform,
            default_provenance=default_provenance,
        )

    def load(
        self,
        source: str | Path,
        platform: str | None = None,
        provenance: str | None = None,
        column_mapping: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load a JSON file and return a normalized 12-column DataFrame.

        Supports:
        - List of record dictionaries: `[{"post_id": 1, "text": "foo", ...}, ...]`
        - Dict with a list under 'data', 'records', 'posts', or 'tweets'
        - JSON Lines (newline-delimited JSON objects)

        Args:
            source: Path to the JSON file.
            platform: Optional platform override.
            provenance: Optional provenance tag override ('Real', 'Imported', 'Synthetic Demo').
            column_mapping: Optional dictionary mapping JSON keys to normalized fields.
            **kwargs: Extra arguments passed to json.load or pd.read_json.

        Returns:
            Normalized pandas DataFrame.

        Raises:
            FileNotFoundError: If the source file does not exist.
            DataIngestionError: If the file is empty, malformed, or cannot be normalized.
        """
        file_path = Path(source)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        raw_content = file_path.read_text(encoding="utf-8").strip()
        if not raw_content:
            raise DataIngestionError(f"JSON file is empty: {file_path}")

        records: list[dict[str, Any]]

        try:
            # Try parsing standard JSON first
            parsed = json.loads(raw_content)
            if isinstance(parsed, list):
                records = parsed
            elif isinstance(parsed, dict):
                # Check for common container keys
                for key in ["data", "records", "posts", "tweets", "items", "comments"]:
                    if key in parsed and isinstance(parsed[key], list):
                        records = parsed[key]
                        break
                else:
                    # Single record wrapped in dict
                    records = [parsed]
            else:
                raise DataIngestionError(f"Unexpected JSON root element type: {type(parsed).__name__}")
        except json.JSONDecodeError:
            # Try JSON Lines format
            lines = [line.strip() for line in raw_content.splitlines() if line.strip()]
            records = []
            for line_idx, line in enumerate(lines, 1):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as err:
                    raise DataIngestionError(
                        f"Malformed JSON in {file_path} at line {line_idx}: {err}"
                    ) from err

        if not records:
            raise DataIngestionError(f"JSON file contains no record entries: {file_path}")

        df = pd.DataFrame(records)
        if df.empty:
            raise DataIngestionError(f"JSON records produced an empty DataFrame: {file_path}")

        return self.normalize_dataframe(
            df=df,
            platform=platform,
            provenance=provenance,
            column_mapping=column_mapping,
        )
