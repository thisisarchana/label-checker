"""Loads the hand-labeled allergen gold set."""
import re
from pathlib import Path

import pandas as pd

GOLD_DIR = Path(__file__).resolve().parent.parent / "data" / "gold"
LABELED = GOLD_DIR / "allergen_gold_labeled.csv"
KEY = GOLD_DIR / "allergen_gold_key.parquet"

ALLERGENS = ["milk", "egg", "fish", "shellfish", "tree_nuts", "peanuts", "wheat", "soy", "sesame"]

# Spelling and plural variants only. Anything else is reported so you can fix it.
ALIASES = {
    "eggs": "egg", "peanut": "peanuts", "tree_nut": "tree_nuts", "treenuts": "tree_nuts",
    "soybeans": "soy", "soybean": "soy", "soya": "soy", "crustacean": "shellfish",
    "crustaceans": "shellfish", "shell_fish": "shellfish", "sesame_seeds": "sesame",
}


def parse_labels(cell: str) -> tuple[set[str], set[str]]:
    """Returns (recognized allergens, unrecognized words)."""
    known, unknown = set(), set()
    for raw in re.split(r"[;,]", cell or ""):
        word = re.sub(r"[\s\-]+", "_", raw.strip().lower())
        if not word:
            continue
        word = ALIASES.get(word, word)
        (known if word in ALLERGENS else unknown).add(word)
    return known, unknown


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding).fillna("")
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str, encoding="latin-1").fillna("")


def load_gold() -> tuple[pd.DataFrame, list[tuple[str, list[str]]]]:
    """Returns the labeled products and a list of (id, unrecognized label words)."""
    sheet = _read_csv(LABELED)
    key = pd.read_parquet(KEY)[["id", "code", "source"]]
    df = sheet.merge(key, on="id", how="left")
    if df.source.isna().any():
        missing = df[df.source.isna()].id.tolist()[:5]
        raise ValueError(f"Rows in the labeled CSV not found in the key: {missing}")

    labels, problems = [], []
    for row in df.itertuples():
        known, unknown = parse_labels(row.allergens_present)
        labels.append(known)
        if unknown:
            problems.append((row.id, sorted(unknown)))
    df["labels"] = labels
    return df, problems