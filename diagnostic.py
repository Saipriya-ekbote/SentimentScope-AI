import pandas as pd
from pathlib import Path
import joblib
from src.sentiment import predict_sentiment, load_model
from src.data_loader import load_csv

def main():
    print("=== 1, 2, 5. TESTING SENTENCES DIRECTLY ===")
    model_path = Path("models/sentiment_model.joblib")
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
        
    pipeline = load_model(model_path)
    print(f"Model type: {type(pipeline)}")
    print(f"Model classes: {pipeline.classes_}")
    
    test_sentences = [
        "I am extremely happy with this service.",
        "I absolutely love this product. It is amazing!",
        "The service was excellent and I am very satisfied.",
        "This is a fantastic experience.",
        "I hate this product. It is terrible.",
        "The service was awful and disappointing.",
        "The product arrived yesterday.",
        "The meeting is scheduled for tomorrow."
    ]
    
    for text in test_sentences:
        res = predict_sentiment(text, pipeline)
        print(f"TEXT: {text}")
        print(f"PREDICTION: {res['sentiment']}")
        print(f"PROBABILITIES: {res['probabilities']}")
        print("-")

    print("\n=== 6 & 8. DATASET INSPECTION ===")
    df = load_csv("data/sample_data.csv")
    print(f"Total rows: {len(df)}")
    print("Class distribution:")
    print(df['sentiment'].value_counts())
    
    print("\nDuplicate rows (exact matches in text):")
    dupes = df[df.duplicated(subset=['text'], keep=False)]
    print(f"Found {len(dupes)} duplicate text rows.")
    
    # The generation script added "#1", "#2" etc to the end of texts. 
    # Let's see if the base texts are duplicated across the entire dataset.
    df['base_text'] = df['text'].str.replace(r' #\d+', '', regex=True).str.replace(r' spike-\d+', '', regex=True)
    base_dupes = df['base_text'].value_counts()
    print("\nBase text counts (ignoring #1, spike-1 etc):")
    print(base_dupes.head(10))
    
    print("\n=== 7. OUT-OF-DISTRIBUTION TESTING ===")
    ood_sentences = {
        "POSITIVE": [
            "I am thrilled with how quickly this was resolved.",
            "The experience exceeded my expectations.",
            "Everything worked perfectly.",
            "I couldn't be happier with the service.",
            "The team did an excellent job."
        ],
        "NEGATIVE": [
            "I regret using this service.",
            "This was a frustrating experience.",
            "Nothing worked the way it should.",
            "I am extremely disappointed.",
            "The quality was unacceptable."
        ],
        "NEUTRAL": [
            "The package arrived this afternoon.",
            "The meeting begins at three.",
            "I received an email from the company.",
            "The store closes at eight.",
            "The report was submitted yesterday."
        ]
    }
    
    for true_label, sentences in ood_sentences.items():
        print(f"\nExpected: {true_label}")
        for text in sentences:
            res = predict_sentiment(text, pipeline)
            print(f"  Pred: {res['sentiment']:8} | Prob: {max(res['probabilities'].values()):.2f} | Text: {text}")

if __name__ == '__main__':
    main()
