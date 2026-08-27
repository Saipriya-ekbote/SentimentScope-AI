# SentimentScope AI

## Problem Statement

Organizations need a way to understand how customers and audiences feel about products, services, or brands across social platforms. SentimentScope AI provides a local-first sentiment analysis and monitoring platform. It loads social-media-style CSV datasets, cleans noisy social text, classifies sentiment using classical machine learning models (Logistic Regression vs. Linear SVM), aggregates trends over time, detects unusual negative sentiment spikes using rolling statistical baselines, and presents diagnostic insights in an interactive Streamlit dashboard.

---

## Key Features

- **Multi-Dataset Support**: Seamlessly switch between a synthetic multi-platform development dataset and a real-world Twitter dataset from the sidebar.
- **Robust Social-Media Preprocessing**: Cleans messy social data (HTML entity decoding, URL removal, `@mention` handle stripping, hashtag `#` symbol stripping, character elongation compression).
- **Side-by-Side Model Comparison**: Evaluates Logistic Regression and Linear Support Vector Classifier (LinearSVC) on the exact same stratified train/test split.
- **Leak-Free Evaluation Pipeline**: Vectorizers and classifiers are strictly fitted only on training data, reporting honest Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and Classification Reports.
- **Dataset-Isolated Model Persistence**: Dedicated model persistence paths per dataset (`sentiment_model_sample.joblib` and `sentiment_model_realistic.joblib`) to prevent cross-dataset contamination.
- **Time-Series Sentiment Tracking**: Aggregates post volume and sentiment percentages by hour or day.
- **Statistical Spike Detection & Alerts**: Uses rolling z-scores over negative sentiment volume to flag anomalies and generate severity-graded alerts (`CRITICAL`, `HIGH`, `MEDIUM`).
- **Interactive Streamlit Dashboard**: Comprehensive web UI featuring overview metrics, model comparison diagnostics, Plotly charts, platform breakdowns, and single-text inference.

---

## Architecture & Data Flow

```text
┌────────────────────────────────────────────────────────┐
│                        CSV Data                        │
│   (Sample Dataset  /  Realistic Twitter Airline Data)  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│             Data Loader (src/data_loader.py)           │
│   Schema validation, ID de-duplication, format checks  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│        Preprocessing (src/preprocessing.py)            │
│   HTML decode, @mention / URL strip, char compression  │
└─────────────┬────────────────────────────┬─────────────┘
              │                            │
              ▼                            ▼
┌───────────────────────────┐┌───────────────────────────┐
│     Machine Learning      ││   Time-Series & Alerts    │
│    (src/sentiment.py)     ││ (src/spike_detection.py)  │
│ ───────────────────────── ││ ───────────────────────── │
│ • 80/20 Stratified Split  ││ • Hourly / Daily Buckets  │
│ • TF-IDF (Train Fit Only) ││ • Rolling Mean & Std Dev  │
│ • Logistic Regression     ││ • Rolling Z-Score Spikes  │
│ • Linear SVM              ││ • Alert Generation        │
└─────────────┬─────────────┘└─────────────┬─────────────┘
              │                            │
              └─────────────┬──────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│         Streamlit Dashboard (app.py)                   │
│   • Dataset Selector & Dynamic Information Banners     │
│   • Model Comparison Table & Classification Reports    │
│   • Sentiment Distributions & Time-Series Charts       │
│   • Platform Breakdown & Spike Alert Banners           │
│   • Live Single-Text Sentiment Prediction              │
└────────────────────────────────────────────────────────┘
```

---

## Project Structure

```text
SentimentScope AI
├── data/
│   ├── raw/
│   │   └── Tweets.csv                  # Raw Twitter US Airline Sentiment dataset
│   ├── processed_data.csv              # Legacy processed data artifact
│   ├── realistic_social_data.csv       # Converted realistic dataset (14,485 rows)
│   └── sample_data.csv                 # Synthetic development dataset (310 rows)
├── models/
│   └── .gitkeep                        # Holds dataset-specific .joblib models
├── scripts/
│   ├── generate_sample_data.py         # Generates synthetic multi-platform dataset
│   └── prepare_realistic_data.py       # Converts raw tweets to standard schema
├── src/
│   ├── __init__.py
│   ├── alerts.py                       # Structured alert generation from spikes
│   ├── data_loader.py                  # Schema validation & dataset loading
│   ├── data_validation.py              # Basic CSV data validation utility
│   ├── preprocessing.py                # Social media text normalization & cleaning
│   ├── sentiment.py                    # Classical ML pipelines & model comparison
│   └── spike_detection.py              # Rolling z-score statistical spike detection
├── tests/
│   ├── conftest.py                     # Pytest fixtures
│   ├── test_compare_models.py          # Phase 2 model comparison tests
│   ├── test_data_loader.py             # Data loading & preparation tests
│   ├── test_preprocessing.py           # Text cleaning tests
│   ├── test_sentiment.py               # Model training & prediction tests
│   └── test_spike_detection.py         # Statistical anomaly detection tests
├── app.py                              # Streamlit multi-dataset web application
├── config.py                           # Project-level configuration constants
├── diagnostic.py                       # Development diagnostic & OOD testing script
├── README.md                           # Project documentation
└── requirements.txt                    # Project dependencies
```

---

## Standard Data Schema

Both datasets strictly conform to the 5-column schema validated by `src/data_loader.py`:

| Column | Type | Description | Validation Constraints |
| :--- | :--- | :--- | :--- |
| `id` | String | Unique record identifier | Must be unique across all rows; non-empty |
| `platform` | String | Social media platform origin | Non-empty string (e.g., `Twitter`, `Reddit`, `YouTube`) |
| `text` | String | Post content | Non-empty string |
| `timestamp` | String / Datetime | Post creation timestamp | Valid datetime parseable to `YYYY-MM-DD HH:MM:SS` |
| `sentiment` | String | Labeled sentiment class | Strictly `positive`, `negative`, or `neutral` |

---

## Datasets

### 1. Sample Dataset (`data/sample_data.csv`)
* **Purpose**: Development, automated testing, and deterministic spike-alert demonstration.
* **Row Count**: 310 rows.
* **Platforms**: 5 simulated platforms (`Twitter`, `Reddit`, `YouTube`, `Instagram`, `Facebook`) with 62 posts each.
* **Sentiment Distribution**: 110 negative (35.5%), 100 neutral (32.3%), 100 positive (32.3%).
* **Temporal Range**: 2026-08-01 08:00:00 to 2026-08-06 03:00:00.
* **Injected Anomaly**: On Day 3 (August 4–5, 2026), 10 outage-themed negative records are injected to create a measurable negative spike.

### 2. Realistic Twitter Dataset (`data/realistic_social_data.csv`)
* **Source**: Real-world [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) dataset (`data/raw/Tweets.csv`).
* **Source Volume**: 14,640 raw tweets.
* **Deduplication**: 155 duplicate `tweet_id` rows removed.
* **Final Usable Volume**: 14,485 unique records.
* **Platform**: `Twitter` exclusively.
* **Sentiment Distribution**:
  * **`negative`**: 9,082 (62.7%)
  * **`neutral`**: 3,069 (21.2%)
  * **`positive`**: 2,334 (16.1%)
* **Temporal Range**: 2015-02-16 to 2015-02-24 (8.5 continuous days with natural customer volume peaks on Feb 22–23).
* **Characteristics**: Contains authentic real-world social-media noise (100% `@mention` tags, 8.0% URLs, 16.2% hashtags, 4.9% HTML entities, and 11.3% elongated words).

---

## Social-Media Text Preprocessing

Phase 3.1 introduced a modular text cleaning pipeline in `src/preprocessing.py`:
1. **HTML Entity Decoding**: Unescapes entities like `&amp;` → `&`, `&lt;` → `<`, and `&quot;` → `"`.
2. **URL Removal**: Strips standard `http://`, `https://`, and `www.` web links.
3. **Handle Removal**: Strips `@username` mentions while preserving internal email addresses.
4. **Hashtag Normalization**: Strips the leading `#` symbol while preserving the sentiment-bearing token (e.g. `#terrible` → `terrible`).
5. **Character Elongation Compression**: Compresses sequences of 3 or more repeated characters down to 2 (e.g., `sooooo` → `soo`, `delayeeeed` → `delayeed`), preserving valid English double-letter spellings (`good`, `coffee`).
6. **Case & Whitespace Normalization**: Converts text to lowercase and strips redundant whitespace.

---

## Machine Learning & Model Comparison

The system uses classical, interpretable machine learning pipelines implemented in `src/sentiment.py`:

1. **Feature Extraction**: `TfidfVectorizer` (max features = 5,000, unigrams and bigrams `(1, 2)`, min document frequency = 1).
2. **Classifiers**:
   - **Logistic Regression**: `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)`
   - **Linear SVM**: `LinearSVC(C=1.0, class_weight="balanced", random_state=42)`
3. **Leak-Free Evaluation Methodology**:
   - Data is split into 80% train and 20% test using stratified sampling (`random_state=42`).
   - The TF-IDF vectorizer is strictly fitted **only on the training split** and transforms the test split to eliminate data leakage.
   - Both models are evaluated on the identical held-out test split.
   - Reports Accuracy, Weighted Precision, Weighted Recall, Weighted F1 Score, Confusion Matrix, and Full Classification Reports.

---

## Model Persistence & Dataset Isolation

To ensure complete isolation between datasets:
* **Sample Dataset Model**: Stored at `models/sentiment_model_sample.joblib`.
* **Realistic Twitter Model**: Stored at `models/sentiment_model_realistic.joblib`.
* **Why `.joblib` files are ignored by Git**:
  - `models/*.joblib` files are excluded in `.gitignore` to keep the repository lightweight and prevent binary merge conflicts or cross-platform pickle deserialization issues.
  - The application automatically fits and persists the appropriate model on its first run in under 1 second.

---

## Spike Detection & Alerts

Negative sentiment spikes are detected using a statistical baseline in `src/spike_detection.py`:
1. Post volume is aggregated by hour or day.
2. A rolling window ($w = 3$) calculates the rolling mean and standard deviation of negative post counts, lagged by 1 period to establish a baseline.
3. The current observation's z-score is computed:
   $$z = \frac{\text{observed} - \text{baseline}}{\sigma}$$
4. If $z \ge \text{threshold}$ (default 2.0), an alert record is emitted with severity:
   - **`CRITICAL`**: $z \ge 4.0$
   - **`HIGH`**: $3.0 \le z < 4.0$
   - **`MEDIUM`**: $2.0 \le z < 3.0$

---

## Benchmark Results

### 1. Synthetic Demonstration Dataset (`sample_data.csv`, Test Size = 62)
| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1 Score (Weighted) | Best Model |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9839 | 0.9845 | 0.9839 | 0.9839 | Tied |
| **Linear SVM** | 0.9839 | 0.9845 | 0.9839 | 0.9839 | Tied |

*Note: High scores reflect formulaic synthetic development sentences designed for pipeline testing.*

### 2. Realistic Twitter Dataset (`realistic_social_data.csv`, Test Size = 2,897)
| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1 Score (Weighted) | Best Model |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7746 | 0.7906 | 0.7746 | **0.7804** | **Best Model** |
| **Linear SVM** | 0.7746 | 0.7762 | 0.7746 | 0.7752 | - |

---

## Installation & Running

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Saipriya-ekbote/SentimentScope-AI.git
cd SentimentScope-AI

# Create and activate Python 3.12 virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. (Optional) Regenerate Datasets
```bash
# Regenerate synthetic sample data
python scripts/generate_sample_data.py

# Regenerate realistic Twitter dataset from raw data
python scripts/prepare_realistic_data.py
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. Use the sidebar dropdown to toggle between **Sample Dataset** and **Realistic Twitter Dataset**.

---

## Running Tests

Run the full pytest suite:

```bash
python -m pytest -q
```

All **35 automated unit and integration tests** verify data loader constraints, text cleaning rules, model training reproducibility, model comparison metrics, and spike alert generation.

---

## Limitations & Future Improvements

### Current Limitations
- **Local CSV Processing**: Operates on static CSV files rather than live streaming APIs.
- **Single-Domain Realistic Data**: The realistic dataset is currently specific to Twitter airline customer service interactions.
- **Baseline Spike Detection**: The rolling z-score baseline requires sufficient contiguous time periods ($N \ge 3$) to compute variance.

### Planned Improvements
- **Additional Real-World Datasets**: Integrate additional labeled social-media datasets covering different domains and platforms.
- **Advanced Feature Engineering**: Explore n-gram tuning, feature selection, and ensemble methods to improve classifier performance.
- **Automated Webhook Notifications**: Route critical spike alerts to Slack or Discord channels.

---

## 🗺️ Project Roadmap

- [x] **Phase 1**: Initial project setup, classical ML sentiment classification, time-series aggregation, spike alerts, and Streamlit dashboard.
- [x] **Phase 2**: Side-by-side model comparison (Logistic Regression vs. Linear SVM) with leak-free stratified evaluation.
- [x] **Phase 3.1**: Social-media text preprocessing (HTML decoding, URL stripping, handle removal, character compression).
- [x] **Phase 3.2**: Realistic dataset integration (Twitter US Airline Sentiment) and multi-dataset dashboard selection.
- [ ] **Phase 4**: Additional datasets, advanced feature engineering, and webhook alert dispatch.