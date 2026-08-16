"""Generate SYNTHETIC DEVELOPMENT DATA for SentimentScope AI."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

PLATFORMS = ["Twitter", "Reddit", "YouTube", "Instagram", "Facebook"]

import random

POSITIVE_SUBJECTS = ["I", "My team", "This product", "The service", "The app", "Customer support", "The new update", "Everything", "The interface", "The design"]
POSITIVE_VERBS = ["love", "enjoy", "appreciate", "recommend", "am thrilled with", "am happy with", "am satisfied with", "am pleased with", "admire", "value"]
POSITIVE_ADJECTIVES = ["great", "fantastic", "amazing", "excellent", "good", "perfect", "wonderful", "outstanding", "brilliant", "smooth", "reliable"]

NEGATIVE_SUBJECTS = ["I", "My team", "This product", "The service", "The app", "Customer support", "The new update", "Everything", "The interface", "The delivery"]
NEGATIVE_VERBS = ["hate", "dislike", "regret", "can't stand", "am disappointed with", "am unhappy with", "am frustrated by", "am angry about", "struggle with", "abandoned"]
NEGATIVE_ADJECTIVES = ["terrible", "awful", "bad", "horrible", "unacceptable", "poor", "frustrating", "confusing", "broken", "unreliable", "useless"]

NEUTRAL_SUBJECTS = ["I", "The user", "The package", "The meeting", "The app", "The report", "The email", "The store", "The invoice", "The setting"]
NEUTRAL_VERBS = ["received", "opened", "closed", "started", "scheduled", "read", "viewed", "checked", "updated", "downloaded"]
NEUTRAL_OBJECTS = ["today", "yesterday", "tomorrow", "this morning", "in the afternoon", "at nine", "on Monday", "from the website", "in my inbox", "once a week"]

SPIKE_NEGATIVE_TEXTS = [
    "This outage is unacceptable.",
    "The app has been broken all day.",
    "I am furious about the downtime.",
    "Support still has not replied.",
    "This bug ruined my workflow.",
    "The latest patch introduced new errors.",
    "I lost data because of this failure.",
    "This is the third crash today.",
    "Billing charged me twice.",
    "The service is completely unreliable."
]


def _build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    row_id = 1
    start = datetime(2026, 8, 1, 8, 0, 0)
    
    # Use fixed seed for reproducible synthetic data
    rng = random.Random(42)

    def generate_texts(subjects, verbs, endings, count):
        texts = set()
        while len(texts) < count:
            texts.add(f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(endings)}.")
        return list(texts)

    positive_texts = generate_texts(POSITIVE_SUBJECTS, POSITIVE_VERBS, POSITIVE_ADJECTIVES, 100)
    neutral_texts = generate_texts(NEUTRAL_SUBJECTS, NEUTRAL_VERBS, NEUTRAL_OBJECTS, 100)
    negative_texts = generate_texts(NEGATIVE_SUBJECTS, NEGATIVE_VERBS, NEGATIVE_ADJECTIVES, 100)

    def add_rows(texts: list[str], sentiment: str, day_offset: int, hour_step: int) -> None:
        nonlocal row_id
        for index, text in enumerate(texts):
            timestamp = start + timedelta(days=day_offset, hours=index * hour_step)
            platform = PLATFORMS[index % len(PLATFORMS)]
            rows.append(
                {
                    "id": str(row_id),
                    "platform": platform,
                    "text": text,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "sentiment": sentiment,
                }
            )
            row_id += 1

    # Add 20 texts per day for 5 days
    for day in range(5):
        add_rows(positive_texts[day*20:(day+1)*20], "positive", day, 1)
        add_rows(neutral_texts[day*20:(day+1)*20], "neutral", day, 1)
        
        if day == 3:
            # Inject spike data on day 3
            spike_texts = negative_texts[day*20:(day+1)*20] + SPIKE_NEGATIVE_TEXTS
            add_rows(spike_texts, "negative", day, 1)
        else:
            add_rows(negative_texts[day*20:(day+1)*20], "negative", day, 1)

    return rows


def main() -> None:
    import sys

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from src.data_loader import load_csv
    from src.preprocessing import preprocess_dataframe

    output_path = project_root / "data" / "sample_data.csv"
    processed_path = project_root / "data" / "processed_data.csv"
    rows = _build_rows()
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["id", "platform", "text", "timestamp", "sentiment"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")

    processed_df = preprocess_dataframe(load_csv(output_path))
    processed_df.to_csv(processed_path, index=False)
    print(f"Wrote {len(processed_df)} rows to {processed_path}")


if __name__ == "__main__":
    main()
