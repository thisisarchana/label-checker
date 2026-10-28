"""Finds gluten-claim products where cutting the text at a warning phrase may have hidden ingredients."""
from pathlib import Path

import pandas as pd

from app.extraction.clean import strip_advisory
from app.extraction.off_loader import (
    CONTAINS_RE,
    PREFIX_RE,
    has_gluten_free_claim,
    split_ingredients,
    to_label_data,
)
from app.rules.gluten_free import check_gluten_free_claim

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "raw" / "off_us_sample.parquet"


def uncut_ingredients(raw: str) -> list[str]:
    """Drops only real cross-contact warnings and the 'Contains:' statement, nothing else."""
    text = PREFIX_RE.sub("", strip_advisory(raw))
    match = CONTAINS_RE.search(text)
    if match:
        text = text[:match.start()]
    return split_ingredients(text)


def main() -> None:
    df = pd.read_parquet(SAMPLE)
    claimed = df[df.labels_tags.apply(has_gluten_free_claim)]

    shorter = changed = 0
    examples = []
    for record in claimed.to_dict("records"):
        old = to_label_data(record).ingredients
        new = uncut_ingredients(record["ingredients_text"] or "")
        if len(new) > len(old):
            shorter += 1
        old_verdict = check_gluten_free_claim("Gluten-Free", old, None).verdict.value
        new_verdict = check_gluten_free_claim("Gluten-Free", new, None).verdict.value
        if old_verdict != new_verdict:
            changed += 1
            examples.append((record["product_name"][:35], old_verdict, new_verdict, len(old), len(new)))

    print(f"Gluten-claim products checked: {len(claimed):,}")
    print(f"Ingredient list was cut short by the old parsing: {shorter}")
    print(f"Verdict changes when the list is not cut short: {changed}\n")
    for name, old_v, new_v, n_old, n_new in examples:
        print(f"{name:35s} {old_v} -> {new_v}   (ingredients {n_old} -> {n_new})")


if __name__ == "__main__":
    main()