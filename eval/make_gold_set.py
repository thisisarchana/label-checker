"""Exports a blind labeling sheet of real products for the allergen gold set."""
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "raw" / "off_us_sample.parquet"
OUT_DIR = ROOT / "data" / "gold"

SEED = 7
N_RANDOM = 100   # realistic mix, used for honest overall numbers
PER_RARE = 8     # extra examples of rarer allergens, used for per-allergen recall

# Open Food Facts tags, used ONLY to pick extra examples. They are not shown to the labeler.
RARE_TAGS = ["en:fish", "en:crustaceans", "en:sesame-seeds", "en:nuts", "en:peanuts", "en:eggs"]


def main() -> None:
    df = pd.read_parquet(SAMPLE)
    text = df.ingredients_text.fillna("")
    keep = text.str.len().between(30, 1500) & text.str.contains(",")
    df = df[keep.to_numpy()].reset_index(drop=True)
    print(f"Usable products: {len(df):,}")

    rng = random.Random(SEED)
    chosen = {i: "random" for i in rng.sample(range(len(df)), N_RANDOM)}

    for tag in RARE_TAGS:
        mask = df.allergens_tags.apply(lambda tags: tag in list(tags)).to_numpy()
        candidates = [i for i in df.index[mask] if i not in chosen]
        rng.shuffle(candidates)
        picked = candidates[:PER_RARE]
        for i in picked:
            chosen[i] = "enriched"
        print(f"  {tag:18s} available {len(candidates):5d}, added {len(picked)}")

    order = list(chosen)
    rng.shuffle(order)

    sheet, key = [], []
    for n, i in enumerate(order, start=1):
        row = df.loc[i]
        gid = f"g{n:03d}"
        sheet.append({
            "id": gid,
            "product_name": row.product_name,
            "ingredients_text": row.ingredients_text,
            "allergens_present": "",
            "notes": "",
        })
        key.append({
            "id": gid,
            "code": row.code,
            "source": chosen[i],
            "off_allergens_tags": list(row.allergens_tags),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet_path = OUT_DIR / "allergen_gold_to_label.csv"
    key_path = OUT_DIR / "allergen_gold_key.parquet"
    pd.DataFrame(sheet).to_csv(sheet_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(key).to_parquet(key_path, index=False)

    print(f"\nWrote {len(sheet)} products to label: {sheet_path}")
    print(f"Wrote the hidden key (do not look at it while labeling): {key_path}")


if __name__ == "__main__":
    main()