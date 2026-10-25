import os
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(raw_path="data/data.csv", out_dir="data/processed"):
    df = pd.read_csv(raw_path).dropna().drop_duplicates()
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["target"]
    )
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    return train, test


if __name__ == "__main__":
    preprocess()
