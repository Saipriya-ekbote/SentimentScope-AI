# SentimentScope AI

## Problem Statement

Organizations need a way to understand how people feel about products, services, or brands across social platforms. SentimentScope AI provides a local-first MVP that loads social-media-style text from a CSV file, classifies sentiment with a transparent machine-learning model, tracks sentiment over time, detects unusual negative spikes, and presents the results in a Streamlit dashboard.

## Features

- Load and validate local CSV datasets
- Clean and preprocess social-media-style text
- Train a sentiment classifier and cache it for performance
- Predict sentiment and confidence for new text input
- Aggregate sentiment statistics by hour or day
- Detect negative sentiment spikes using a statistical baseline
- Generate structured spike alerts with varying severities
- Explore results in an interactive Streamlit dashboard

## Architecture

```text
CSV -> Data Loader -> Preprocessing -> TF-IDF -> Logistic Regression
       -> Sentiment Prediction -> Time-Series Analysis -> Spike Detection
       -> Alerts -> Streamlit Dashboard
```

## Technologies

- Python 3.12
- pandas & numpy (Data manipulation and computation)
- scikit-learn (Machine Learning)
- joblib (Model persistence)
- plotly (Interactive visualizations)
- streamlit (Web dashboard)
- pytest (Testing framework)

## Project Structure

```text
C:\Users\saipr\OneDrive\Documents\GitHub\SentimentScope AI
+---data
|       sample_data.csv
+---models
|       .gitkeep
+---scripts
|       generate_sample_data.py
+---src
|       alerts.py
|       data_loader.py
|       preprocessing.py
|       sentiment.py
|       spike_detection.py
|       __init__.py
+---tests
|       conftest.py
|       test_data_loader.py
|       test_preprocessing.py
|       test_sentiment.py
|       test_spike_detection.py
|   app.py
|   README.md
|   requirements.txt
```

## Installation

1. Create and activate a Python 3.12 virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Regenerate the development dataset:
```bash
python scripts/generate_sample_data.py
```

## Running the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

The app will load the data, process the text, train or load the machine learning model, and launch an interactive web dashboard at `http://localhost:8501`.

## Running Tests

Execute the full test suite with:

```bash
pytest
```

## Dataset

**The included dataset is synthetic development data created for demonstration and testing. It is not collected from real social-media platforms.**

It contains examples of positive, negative, and neutral texts assigned to various platforms (Twitter, Reddit, YouTube, Instagram, Facebook). A negative spike is intentionally injected into the dataset to demonstrate spike detection.

## Machine Learning Approach

The sentiment classifier is built entirely using classical machine learning, avoiding external APIs and generative AI:
- **TF-IDF (Term Frequency-Inverse Document Frequency):** Converts text into numerical vectors based on word frequencies.
- **Logistic Regression:** Classifies the numerical vectors into `positive`, `negative`, or `neutral` categories using a balanced class weight approach.

## Spike Detection

Spikes in negative sentiment are identified using a standard statistical method rather than artificial intelligence. The application calculates a rolling mean and a rolling standard deviation over time-bucketed negative post counts. A period is flagged as a "spike" when its value exceeds the baseline by a configurable z-score threshold (e.g., 2.0 standard deviations above the mean). 

## Results

On the synthetic demonstration dataset (`sample_data.csv`), the model achieves the following metrics (train size = 248, test size = 62):
- **Accuracy:** 1.0000
- **Precision (weighted):** 1.0000
- **Recall (weighted):** 1.0000
- **F1 Score:** 1.0000

*Note: Perfect scores are expected when training on highly distinguishable synthetic data and will not hold on noisy real-world data.*

## Limitations

- Operates solely on local CSV data; no live integration.
- The default model is trained on a small synthetic dataset and will not generalize well to unseen slang or complex contexts.
- Spike detection uses a simplistic rolling z-score approach, which is a baseline statistical method, not an advanced anomaly detection system.
- The dashboard is for single-user demonstration and lacks authentication or database persistence.

## Future Improvements

- **Live APIs:** Integrate with actual platform APIs (Reddit, X, etc.) to ingest real data.
- **Larger Datasets:** Train on massive, real-world datasets for better generalization.
- **Advanced NLP:** Utilize deep learning models (e.g., BERT) or multilingual embeddings for robust sentiment analysis.
- **Real-Time Streaming:** Move from batch CSV processing to a real-time message broker like Kafka.
- **Notification Integrations:** Send alerts to Slack, Discord, or Email automatically.
