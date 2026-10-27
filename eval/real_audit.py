"""Runs the gluten-free rule on real products that carry a gluten-free label."""
from collections import Counter
from pathlib import Path

import pandas as pd

from app.extraction.off_loader import has_gluten_free_claim, to_label_data
from app.rules.gluten_free import check_gluten_free_claim

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "raw" / "off_us_sample.parquet"
OUT = SAMPLE.parent / "gluten_audit.csv"


def main() -> None:
    df = pd.read_parquet(SAMPLE)

    tag_counts = Counter(
        tag for tags in df.labels_tags for tag in tags if "gluten" in tag
    )
    print("Gluten-related label tags in the sample:")
    for tag, n in tag_counts.most_common():
        print(f"  {tag:40s} {n}")

    claimed = df[df.labels_tags.apply(has_gluten_free_claim)]
    print(f"\nProducts with a gluten-free claim: {len(claimed):,}")

    rows = []
    for record in claimed.to_dict("records"):
        label = to_label_data(record)
        finding = check_gluten_free_claim("Gluten-Free", label.ingredients, None)
        rows.append({
            "code": record["code"],
            "product_name": record["product_name"],
            "verdict": finding.verdict.value,
            "grains_found": ", ".join(finding.evidence.get("gluten_grains", [])),
            "review_items": ", ".join(finding.evidence.get("review_items", [])),
            "ingredients": record["ingredients_text"][:300],
        })

    audit = pd.DataFrame(rows)
    audit.to_csv(OUT, index=False)

    print("\nVerdicts:")
    print(audit.verdict.value_counts().to_string())

    flagged = audit[audit.verdict == "fail"].head(15)
    print("\nSample of flagged products (verify each by hand):")
    for _, row in flagged.iterrows():
        print(f"  {row.product_name[:45]:45s} -> {row.grains_found}")

    print(f"\nFull audit saved to {OUT}")


if __name__ == "__main__":
    main()