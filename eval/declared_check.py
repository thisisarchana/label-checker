"""Lists products that pass only because the ingredient text says gluten-free."""
import re
from pathlib import Path

import pandas as pd

from app.extraction.off_loader import has_gluten_free_claim, to_label_data
from app.rules.gluten_free import check_gluten_free_claim
from app.rules.models import Verdict

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "raw" / "off_us_sample.parquet"
GF = re.compile(r"gluten[\s-]*free", re.I)


def main() -> None:
    df = pd.read_parquet(SAMPLE)
    claimed = df[df.labels_tags.apply(has_gluten_free_claim)]

    rows = []
    for record in claimed.to_dict("records"):
        label = to_label_data(record)
        finding = check_gluten_free_claim("Gluten-Free", label.ingredients, None)
        accepted = finding.evidence.get("declared_items", [])
        if finding.verdict == Verdict.PASS and accepted:
            text = record["ingredients_text"] or ""
            match = GF.search(text)
            start = max(0, match.start() - 60) if match else 0
            rows.append((record["product_name"][:35], accepted, text[start:start + 130]))

    print(f"Products that pass only on a gluten-free statement: {len(rows)}\n")
    for name, accepted, context in rows:
        print(f"{name}")
        print(f"  accepted: {'; '.join(a[:40] for a in accepted)}")
        print(f"  context:  ...{context}\n")


if __name__ == "__main__":
    main()