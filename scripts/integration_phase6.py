"""Integration test: Phase 6 aggregation on multi_platform_demo.csv."""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

from src.data_collection import CSVLoader
from src.preprocessing.cleaner import preprocess_dataframe
from src.aggregation import (
    aggregate_sentiment,
    aggregate_by_brand,
    aggregate_by_platform,
    aggregate_sentiment_over_time,
    aggregate_by_dimension_over_time,
    aggregate_by_entities,
)
from src.time_series import build_time_series, add_rolling_statistics

# ---- Load -------------------------------------------------------------------
loader = CSVLoader()
df = loader.load("data/sample/multi_platform_demo.csv")
print(f"Loaded: {len(df)} rows, {df['platform'].nunique()} platforms")

# ---- Preprocess -------------------------------------------------------------
df = preprocess_dataframe(df, target_column="text")

# ---- Ensure sentiment columns exist (from CSV or fallback) ------------------
if "sentiment" not in df.columns or df["sentiment"].isna().all():
    rng = np.random.default_rng(42)
    df["sentiment"] = rng.choice(["positive", "neutral", "negative"], len(df))
    df["sentiment_score"] = rng.uniform(-1, 1, len(df)).round(4)

# ---- Overall ----------------------------------------------------------------
overall = aggregate_sentiment(df)
assert overall["post_count"] == len(df), "post_count mismatch"
print(f"Overall: {overall['post_count']} posts, avg_score={overall['average_sentiment_score']:.3f}")

# ---- By brand ---------------------------------------------------------------
by_brand = aggregate_by_brand(df)
print("\nBy brand:")
print(by_brand[["brand", "post_count", "average_sentiment_score"]].to_string(index=False))
assert len(by_brand) >= 1

# ---- By platform ------------------------------------------------------------
by_platform = aggregate_by_platform(df)
print("\nBy platform:")
print(by_platform[["platform", "post_count"]].to_string(index=False))
assert set(by_platform["platform"].str.lower()).issubset({"twitter", "reddit", "instagram", "youtube", "news"})

# ---- Time-series (daily) ----------------------------------------------------
ts = aggregate_sentiment_over_time(df, freq="daily")
print(f"\nTime-series rows: {len(ts)}, zero-count rows: {(ts['post_count'] == 0).sum()}")
# Empty-bucket NaN rule
empty = ts[ts["post_count"] == 0]
if len(empty):
    assert empty["average_sentiment_score"].isna().all(), "Empty bucket avg_score should be NaN"

# ---- Brand x time -----------------------------------------------------------
by_brand_ts = aggregate_by_dimension_over_time(df, dimension="brand", freq="daily")
print(f"Brand x time rows: {len(by_brand_ts)}")
assert "brand" in by_brand_ts.columns
assert "timestamp" in by_brand_ts.columns

# ---- Multi-entity -----------------------------------------------------------
entity_ts = aggregate_by_entities(df, freq="daily")
print(f"Entity dims: {list(entity_ts.keys())}")
for dim, result_df in entity_ts.items():
    assert isinstance(result_df, pd.DataFrame), f"{dim} should be DataFrame"

# ---- Time-series builder ----------------------------------------------------
full_ts = build_time_series(df, freq="daily")
print(f"Full TS rows: {len(full_ts)}, gap rows: {(full_ts['post_count'] == 0).sum()}")
assert "timestamp" in full_ts.columns
sorted_ts = pd.to_datetime(full_ts["timestamp"])
assert sorted_ts.is_monotonic_increasing, "Time-series not sorted"

# ---- Rolling statistics -----------------------------------------------------
ts_rolling = add_rolling_statistics(full_ts, "post_count", window=3)
roll_cols = [c for c in ts_rolling.columns if "rolling" in c]
print(f"Rolling columns: {roll_cols}")
assert "post_count_rolling_mean" in roll_cols
assert "post_count_rolling_std" in roll_cols
# Verify causality: row 0 mean == row 0 value
assert abs(ts_rolling["post_count_rolling_mean"].iloc[0] - ts_rolling["post_count"].iloc[0]) < 1e-9

print()
print("=== INTEGRATION TEST PASSED ===")
