"""Builds the allergen training set from the Open Food Facts sample, excluding every gold product."""
import re
from pathlib import Path

import pandas as pd

from app.extraction.clean import strip_advisory
from eval.gold import ALLERGENS, GOLD_DIR

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "raw" / "off_us_sample.parquet"
KEY = GOLD_DIR / "allergen_gold_key.parquet"
OUT_DIR = ROOT / "data" / "processed"

# Open Food Facts allergen tag -> gold allergen. "en:gluten" is the only wheat-related tag,
# and it also covers barley, rye and oats, so the wheat label is noisy.
TAG_TO_ALLERGEN = {
    "en:milk": "milk", "en:eggs": "egg", "en:fish": "fish", "en:crustaceans": "shellfish",
    "en:nuts": "tree_nuts", "en:peanuts": "peanuts", "en:gluten": "wheat",
    "en:soybeans": "soy", "en:sesame-seeds": "sesame",
}
SEED = 13
VALID_FRACTION = 0.1

SOY_OIL = re.compile(r"soy(?:a|bean)?\s+oil", re.I)
PEANUT_OIL = re.compile(r"peanut\s+oil", re.I)
WHEAT_WORD = re.compile(r"\b(?:wheat|semolina|durum|spelt|farina|bulgur|couscous)\b", re.I)


def oil_only(text: str, oil: re.Pattern, word: str) -> bool:
    """True if the allergen word appears in the text only as part of the oil name."""
    return bool(oil.search(text)) and not re.search(rf"\b{word}\w*", oil.sub(" ", text), re.I)


def main() -> None:
    df = pd.read_parquet(SAMPLE)
    df["code"] = df.code.astype(str)
    gold_codes = set(pd.read_parquet(KEY).code.astype(str))

    before = len(df)
    df = df[~df.code.isin(gold_codes)]
    print(f"Gold products removed from training data: {before - len(df)} (gold set has {len(gold_codes)})")

    df = df.assign(text=df.ingredients_text.fillna("").map(strip_advisory))
    df = df[df.text.str.len() >= 10].reset_index(drop=True)

    for tag, allergen in TAG_TO_ALLERGEN.items():
        df[allergen] = df.allergens_tags.apply(lambda tags, t=tag: int(t in list(tags)))

    valid = df.sample(frac=VALID_FRACTION, random_state=SEED)
    train = df.drop(valid.index)
    keep = ["code", "product_name", "text", "allergens_tags"] + ALLERGENS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train[keep].to_parquet(OUT_DIR / "train.parquet", index=False)
    valid[keep].to_parquet(OUT_DIR / "valid.parquet", index=False)
    print(f"Train: {len(train):,}   Validation: {len(valid):,}   (saved to {OUT_DIR})")

    print("\nPositives per allergen in train:")
    for allergen in ALLERGENS:
        n = int(train[allergen].sum())
        print(f"  {allergen:10s} {n:6d}  ({n / len(train):.1%})")
    print(f"No allergen tag at all: {(train[ALLERGENS].sum(axis=1) == 0).mean():.1%}")

    print("\nLabel-policy check (your gold set does NOT label these):")
    for name, oil, word, allergen in (
        ("soybean oil only", SOY_OIL, "soy", "soy"),
        ("peanut oil only", PEANUT_OIL, "peanut", "peanuts"),
    ):
        mask = train.text.apply(lambda t: oil_only(t, oil, word))
        if mask.sum():
            share = train.loc[mask, allergen].mean()
            print(f"  {name}: {int(mask.sum())} products, {share:.0%} carry the '{allergen}' tag anyway")
        else:
            print(f"  {name}: no products found")

    glutenish = train[train.wheat == 1]
    if len(glutenish):
        no_wheat = (~glutenish.text.apply(lambda t: bool(WHEAT_WORD.search(t)))).mean()
        print(f"\nWheat label noise: {no_wheat:.0%} of gluten-tagged products never name wheat "
              f"(barley, rye, oats or malt instead)")


if __name__ == "__main__":
    main()