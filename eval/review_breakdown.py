"""Shows what triggered the review verdicts and the fail rows, for rule tuning."""
import re
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
KEYWORDS = re.compile(r"oats?|wheat starch|brewer'?s yeast", re.I)

audit = pd.read_csv(RAW / "gluten_audit.csv", dtype=str).fillna("")
full = pd.read_parquet(RAW / "off_us_sample.parquet")[["code", "ingredients_text"]]
audit = audit.drop(columns=["ingredients"]).merge(full, on="code", how="left")

print("VERDICT COUNTS")
print(audit.verdict.value_counts().to_string())

review = audit[audit.verdict == "review"]
print("\nREVIEW TRIGGERS")
print(review.review_items.value_counts().head(10).to_string())

print("\nREVIEW EXAMPLES (text around the trigger word)")
for items in review.review_items.value_counts().index[:5]:
    print(f"\n[{items}]")
    for _, row in review[review.review_items == items].head(5).iterrows():
        text = row.ingredients_text or ""
        match = KEYWORDS.search(text)
        start = max(0, match.start() - 50) if match else 0
        print(f"  {row.product_name[:30]:30s} ...{text[start:start + 120]}")

print("\nALL FAIL ROWS")
for _, row in audit[audit.verdict == "fail"].iterrows():
    print(f"  {row.product_name[:45]:45s} -> {row.grains_found}")