"""Searches products that PASSED the gluten check for gluten-risk words the engine misses."""
import re
from pathlib import Path

import pandas as pd

from app.extraction.off_loader import has_gluten_free_claim, split_ingredients, to_label_data
from app.rules.gluten_free import check_gluten_free_claim
from app.rules.models import Verdict

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "raw" / "off_us_sample.parquet"

# Flours that are naturally gluten-free, so "rice flour" isn't a hit.
GF_FLOUR = (
    r"\b(rice|corn|maize|almond|coconut|tapioca|potato|buckwheat|sorghum|chickpea|"
    r"cassava|quinoa|millet|teff|amaranth|arrowroot|soy|soya|bean|lentil|pea|banana|"
    r"cashew|nut|seed|flax|sunflower|pumpkin|sesame|peanut|oat|hazelnut|plantain)\b"
)

PATTERNS = {
    "bare flour": lambda t: re.search(r"\bflour\b", t) and not re.search(GF_FLOUR, t),
    "soy sauce": lambda t: re.search(r"\bsoy sauce\b", t),
    "bread crumbs": lambda t: re.search(r"\b(bread ?crumbs?|panko|crouton)", t),
    "pasta / matzo": lambda t: re.search(r"\b(orzo|matzo|matzah)\b", t),
    "beer / malt vinegar": lambda t: re.search(r"\b(beer|ale|lager|malt vinegar)\b", t),
    "gluten word": lambda t: re.search(r"(?<!no )\bgluten\b(?![- ]free)", t),
}


def main() -> None:
    df = pd.read_parquet(SAMPLE)
    claimed = df[df.labels_tags.apply(has_gluten_free_claim)]

    passed = []
    for record in claimed.to_dict("records"):
        label = to_label_data(record)
        finding = check_gluten_free_claim("Gluten-Free", label.ingredients, None)
        if finding.verdict == Verdict.PASS:
            passed.append(record)

    print(f"Products that passed: {len(passed):,}\n")

    hits = []
    for record in passed:
        for item in split_ingredients(record["ingredients_text"]):
            text = item.lower()
            for name, test in PATTERNS.items():
                if test(text):
                    hits.append((name, record["product_name"][:40], item[:60]))

    print(f"Suspicious ingredients in passed products: {len(hits)}\n")
    for name, product, item in sorted(hits)[:50]:
        print(f"[{name}] {product:40s} -> {item}")


if __name__ == "__main__":
    main()