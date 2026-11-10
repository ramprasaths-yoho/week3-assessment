"""Shared helpers: paths, dataset loading and the ONE train/test split used by Tasks 3 and 4
(so the TF-IDF baseline and DistilBERT are compared on exactly the same test texts)."""
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(ROOT, "data", "feedback.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
MODEL_DIR = os.path.join(ROOT, "models", "distilbert-feedback")
os.makedirs(RESULTS_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    """lowercase -> strip punctuation/digits -> collapse spaces"""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str):
    """very simple whitespace tokenisation on cleaned text"""
    return clean_text(text).split()


def load_dataset():
    if not os.path.exists(DATA_CSV):
        from data.make_dataset import build
        build(DATA_CSV)
    return pd.read_csv(DATA_CSV)


def get_split(test_size=0.2, seed=42):
    df = load_dataset()
    return train_test_split(df["text"], df["label"], test_size=test_size,
                            random_state=seed, stratify=df["label"])
