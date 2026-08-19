import pandas as pd
from pathlib import Path


def validate_data(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False

    df = pd.read_csv(file_path)

    if df.empty:
        print("Dataset is empty.")
        return False

    print("Dataset loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    missing_values = df.isnull().sum()

    print("\nMissing values:")
    print(missing_values)

    return True


if __name__ == "__main__":
    validate_data("data/processed_data.csv")