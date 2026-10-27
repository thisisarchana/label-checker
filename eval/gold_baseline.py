"""Scores the keyword allergen detector against the hand-labeled gold set."""
import sys

from app.extraction.clean import strip_advisory
from app.rules.allergen_crosscheck import _detect
from eval.gold import ALLERGENS, load_gold
from eval.metrics import print_report, score

TO_GOLD = {"crustacean shellfish": "shellfish", "tree nuts": "tree_nuts", "soybeans": "soy"}
MAX_EXAMPLES = 4


def keyword_predict(text: str) -> set[str]:
    return {TO_GOLD.get(a, a) for a in _detect(strip_advisory(text))}


def main() -> int:
    df, problems = load_gold()
    if problems:
        print("Fix these labels in the CSV, then rerun (allowed words: " + ", ".join(ALLERGENS) + ")")
        for gid, words in problems:
            print(f"  {gid}: {', '.join(words)}")
        return 1

    df["pred"] = [keyword_predict(t) for t in df.ingredients_text]
    print(f"Gold set: {len(df)} products ({(df.source == 'random').sum()} random, "
          f"{(df.source == 'enriched').sum()} enriched). "
          f"With no allergens: {(df.labels.str.len() == 0).sum()}.")

    rand = df[df.source == "random"]
    print_report("KEYWORD BASELINE, random products only (use for precision)",
                 score(rand.labels.tolist(), rand.pred.tolist(), ALLERGENS))
    print_report("KEYWORD BASELINE, all products (use for per-allergen recall)",
                 score(df.labels.tolist(), df.pred.tolist(), ALLERGENS))
    print("\nNote: the enriched products over-represent positives, so precision on the full set is inflated.")

    print("\nERROR EXAMPLES")
    for allergen in ALLERGENS:
        missed = df[df.apply(lambda r: allergen in r.labels and allergen not in r.pred, axis=1)]
        extra = df[df.apply(lambda r: allergen not in r.labels and allergen in r.pred, axis=1)]
        for kind, rows in (("MISSED", missed), ("FALSE ALARM", extra)):
            for row in rows.head(MAX_EXAMPLES).itertuples():
                text = strip_advisory(row.ingredients_text)[:170]
                print(f"[{allergen}] {kind} {row.id} {row.product_name[:28]}: {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())