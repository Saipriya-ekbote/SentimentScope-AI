"""CSV data loader for SentimentScope AI multi-platform ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import PROVENANCE_IMPORTED
from src.data_collection.base import BaseDataLoader, DataIngestionError


class CSVLoader(BaseDataLoader):
    """Loader for CSV-formatted social media datasets."""

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
        """Load a CSV file and return a normalized 12-column DataFrame.

        Args:
            source: Path to the CSV file.
            platform: Optional platform override.
            provenance: Optional provenance tag override ('Real', 'Imported', 'Synthetic Demo').
            column_mapping: Optional dictionary mapping CSV column names to normalized fields.
            **kwargs: Extra arguments passed to pd.read_csv.

        Returns:
            Normalized pandas DataFrame.

        Raises:
            FileNotFoundError: If the source file does not exist.
            DataIngestionError: If the file is empty, malformed, or cannot be normalized.
        """
        file_path = Path(source)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        try:
            df = pd.read_csv(file_path, **kwargs)
        except pd.errors.EmptyDataError as exc:
            raise DataIngestionError(f"CSV file contains no data: {file_path}") from exc
        except pd.errors.ParserError as exc:
            raise DataIngestionError(f"Malformed CSV file: {file_path}") from exc

        if df.empty:
            raise DataIngestionError(f"CSV dataset contains zero rows: {file_path}")

        return self.normalize_dataframe(
            df=df,
            platform=platform,
            provenance=provenance,
            column_mapping=column_mapping,
        )
