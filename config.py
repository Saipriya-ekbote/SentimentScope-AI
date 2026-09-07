# Project configuration

APP_TITLE = "SentimentScope-AI"

MAX_DATA_ROWS = 1000

SPIKE_THRESHOLD = 2.0

# Provenance Constants
PROVENANCE_REAL = "Real"
PROVENANCE_IMPORTED = "Imported"
PROVENANCE_SYNTHETIC = "Synthetic Demo"

# Supported Platforms
SUPPORTED_PLATFORMS = [
    "Twitter",
    "Reddit",
    "Instagram",
    "YouTube",
    "Facebook",
    "News",
    "Other",
]

# Standard Normalized Schema (12 fields)
NORMALIZED_COLUMNS = [
    "post_id",
    "platform",
    "timestamp",
    "author",
    "text",
    "brand",
    "product",
    "topic",
    "sentiment",
    "sentiment_score",
    "engagement_metrics",
    "provenance",
]