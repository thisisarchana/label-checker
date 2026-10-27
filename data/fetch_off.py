"""Streams Open Food Facts and saves a random sample of English-language US products."""
import random
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

TARGET = 20_000       # how many products to keep
SCAN_LIMIT = None  # quick test run; set to None to scan the whole file
SEED = 42

OUT_PATH = Path(__file__).resolve().parent / "raw" / "off_us_sample.parquet"

COLUMNS = [
    "code", "product_name", "ingredients_text", "allergens_tags",
    "traces_tags", "labels_tags", "countries_tags", "serving_size",
    "nutriments", "lang",
]

TAG_RE = re.compile(r"<[^>]+>")


def open_stream():
    try:
        return load_dataset(
            "openfoodfacts/product-database", split="food",
            streaming=True, columns=COLUMNS,
        )
    except Exception as err:
        print(f"Column selection not supported ({err}); reading all columns instead.")
        return load_dataset("openfoodfacts/product-database", split="food", streaming=True)


def pick_text(field, order=("main", "en")) -> str:
    """Fields are lists of {lang, text}. Pick one entry and clean it."""
    if not field:
        return ""
    if isinstance(field, str):
        text = field
    else:
        by_lang = {}
        for item in field:
            by_lang.setdefault(item.get("lang"), item.get("text") or "")
        text = next((by_lang[lang] for lang in order if by_lang.get(lang)), "")
    text = TAG_RE.sub("", text)
    return " ".join(text.split())


def nutrient(nutriments, name: str, key: str):
    for item in nutriments or []:
        if item.get("name") == name:
            return item.get(key)
    return None


def main() -> None:
    random.seed(SEED)
    reservoir: list[dict] = []
    seen = 0

    for scanned, row in enumerate(open_stream(), start=1):
        if scanned % 100_000 == 0:
            print(f"scanned {scanned:,} rows, {seen:,} qualifying")
        if SCAN_LIMIT and scanned >= SCAN_LIMIT:
            break

        if row.get("lang") != "en":
            continue
        if "en:united-states" not in (row.get("countries_tags") or []):
            continue
        ingredients = pick_text(row.get("ingredients_text"))
        if len(ingredients) < 5:
            continue

        record = {
            "code": row.get("code"),
            "product_name": pick_text(row.get("product_name")),
            "ingredients_text": ingredients,
            "allergens_tags": list(row.get("allergens_tags") or []),
            "traces_tags": list(row.get("traces_tags") or []),
            "labels_tags": list(row.get("labels_tags") or []),
            "serving_size": row.get("serving_size"),
            "protein_100g": nutrient(row.get("nutriments"), "proteins", "100g"),
            "protein_serving_g": nutrient(row.get("nutriments"), "proteins", "serving"),
        }

        # Reservoir sampling: every qualifying product has an equal chance of being kept.
        seen += 1
        if len(reservoir) < TARGET:
            reservoir.append(record)
        else:
            j = random.randint(0, seen - 1)
            if j < TARGET:
                reservoir[j] = record

    if not reservoir:
        print("No qualifying products found. Raise SCAN_LIMIT or set it to None.")
        return

    df = pd.DataFrame(reservoir)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved {len(df):,} products (from {seen:,} qualifying) to {OUT_PATH}")
    print(f"With at least one allergen tag: {(df.allergens_tags.str.len() > 0).mean():.1%}")
    print(f"With protein per serving:       {df.protein_serving_g.notna().mean():.1%}")
    print("\nMost common allergen tags:")
    print(df.allergens_tags.explode().value_counts().head(12).to_string())
    gluten = df.labels_tags.apply(lambda tags: any("gluten" in t for t in tags))
    protein = df.labels_tags.apply(lambda tags: any("protein" in t for t in tags))
    print(f"\nProducts with a gluten-related label tag:  {gluten.sum():,}")
    print(f"Products with a protein-related label tag: {protein.sum():,}")


if __name__ == "__main__":
    main()