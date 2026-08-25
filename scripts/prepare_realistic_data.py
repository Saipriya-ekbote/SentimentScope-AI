"""Prepare realistic Twitter US Airline Sentiment dataset for SentimentScope AI."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

SOURCE_REQUIRED_COLUMNS = ["tweet_id", "text", "tweet_created", "airline_sentiment"]
TARGET_COLUMNS = ["id", "platform", "text", "timestamp", "sentiment"]
VALID_SENTIMENTS = {"positive", "negative", "neutral"}


def prepare_realistic_data(
    source_path: Path | str,
    output_path: Path | str,
) -> pd.DataFrame:
    """Convert raw Twitter US Airline Sentiment CSV to SentimentScope schema.

    Parameters:
        source_path: Path to raw Tweets.csv.
        output_path: Destination path for realistic_social_data.csv.

    Returns:
        Converted and validated pandas DataFrame.
    """
    source_file = Path(source_path)
    output_file = Path(output_path)

    if not source_file.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_file}")

    df_raw = pd.read_csv(source_file)
    source_rows = len(df_raw)

    # 1. Validate required source columns
    missing_cols = [col for col in SOURCE_REQUIRED_COLUMNS if col not in df_raw.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required source columns: {', '.join(missing_cols)}"
        )

    # 2. De-duplicate on tweet_id, keeping the first occurrence
    df_dedup = df_raw.drop_duplicates(subset=["tweet_id"], keep="first").copy()
    dedup_removed = source_rows - len(df_dedup)

    # 3. Map and format columns
    id_series = df_dedup["tweet_id"].astype(str).str.strip()
    platform_series = pd.Series(["Twitter"] * len(df_dedup), index=df_dedup.index)
    text_series = df_dedup["text"].astype(str).str.strip()
    sentiment_series = (
        df_dedup["airline_sentiment"].astype(str).str.strip().str.lower()
    )

    parsed_timestamps = pd.to_datetime(df_dedup["tweet_created"], errors="coerce")
    timestamp_series = parsed_timestamps.dt.strftime("%Y-%m-%d %H:%M:%S")

    converted_df = pd.DataFrame(
        {
            "id": id_series,
            "platform": platform_series,
            "text": text_series,
            "timestamp": timestamp_series,
            "sentiment": sentiment_series,
        }
    )

    # 4. Validate output constraints
    invalid_mask = (
        converted_df["id"].eq("")
        | converted_df["platform"].eq("")
        | converted_df["text"].eq("")
        | parsed_timestamps.isna()
        | ~converted_df["sentiment"].isin(VALID_SENTIMENTS)
    )
    invalid_count = int(invalid_mask.sum())
    if invalid_count > 0:
        converted_df = converted_df.loc[~invalid_mask].reset_index(drop=True)

    if converted_df["id"].duplicated().any():
        raise ValueError("Duplicate IDs encountered in converted dataset.")

    if converted_df.empty:
        raise ValueError("No usable records remain after conversion.")

    # 5. Save output CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    converted_df.to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL)

    # 6. Summary reporting
    print("=" * 60)
    print("Realistic Social Data Preparation Summary")
    print("=" * 60)
    print(f"Source file:             {source_file}")
    print(f"Output file:             {output_file}")
    print(f"Source rows:             {source_rows:,}")
    print(f"Duplicate IDs removed:   {dedup_removed:,}")
    print(f"Invalid rows dropped:    {invalid_count:,}")
    print(f"Total converted rows:    {len(converted_df):,}")
    print(f"Timestamp range:         {converted_df['timestamp'].min()} to {converted_df['timestamp'].max()}")
    print("\nSentiment distribution:")
    sentiment_counts = converted_df["sentiment"].value_counts()
    for sentiment, count in sentiment_counts.items():
        count_int = int(count)
        pct = (count_int / len(converted_df)) * 100
        print(f"  - {sentiment:<10}: {count_int:>6d} ({pct:5.2f}%)")
    print("=" * 60)

    return converted_df


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    source_csv = project_root / "data" / "raw" / "Tweets.csv"
    output_csv = project_root / "data" / "realistic_social_data.csv"
    prepare_realistic_data(source_csv, output_csv)


if __name__ == "__main__":
    main()
