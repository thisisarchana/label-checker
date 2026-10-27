"""Recomputes gluten verdicts from the sample and shows what triggered review and fail."""
import re
from pathlib import Path

import pandas as pd

from app.extraction.off_loader import has_gluten_free_claim, to_label_data
from app.rules.gluten_free import check_gluten_free_claim

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "raw" / "off_us_sample.parquet"
KEYWORDS = re.compile(r"oats?|wheat starch|brewer'?s yeast", re.I)
SELF_DECLARED = re.compile(r"gluten[\s-]*free|certified", re.I)


def build_audit() -> pd.DataFrame:
    df = pd.read_parquet(SAMPLE)
    claimed = df[df.labels_tags.apply(has_gluten_free_claim)]
    rows = []
    for record in claimed.to_dict("records"):
        label = to_label_data(record)
        finding = check_gluten_free_claim("Gluten-Free", label.ingredients, None)
        rows.append({
            "product_name": record["product_name"] or "",
            "ingredients_text": record["ingredients_text"] or "",
            "verdict": finding.verdict.value,
            "grains_found": ", ".join(finding.evidence.get("gluten_grains", [])),
            "review_items": ", ".join(finding.evidence.get("review_items", [])),
        })
    return pd.DataFrame(rows)


def main() -> None:
    audit = build_audit()

    print("VERDICT COUNTS")
    print(audit.verdict.value_counts().to_string())

    review = audit[audit.verdict == "review"].copy()
    review["trigger"] = review.review_items.str.lower()
    print("\nREVIEW TRIGGERS (case-insensitive)")
    print(review.trigger.value_counts().head(12).to_string())

    declared = review.ingredients_text.str.contains(SELF_DECLARED)
    print(f"\nReview rows whose own text says 'gluten free' or 'certified': {declared.sum()} of {len(review)}")

    print("\nREVIEW EXAMPLES (text around the trigger word)")
    for trigger in review.trigger.value_counts().index[:5]:
        print(f"\n[{trigger}]")
        for _, row in review[review.trigger == trigger].head(5).iterrows():
            text = row.ingredients_text
            match = KEYWORDS.search(text)
            start = max(0, match.start() - 50) if match else 0
            print(f"  {row.product_name[:30]:30s} ...{text[start:start + 120]}")

    print("\nALL FAIL ROWS")
    for _, row in audit[audit.verdict == "fail"].iterrows():
        print(f"\n  {row.product_name[:50]}")
        print(f"    grains found: {row.grains_found}")
        print(f"    text: {row.ingredients_text[:200]}")


if __name__ == "__main__":
    main()