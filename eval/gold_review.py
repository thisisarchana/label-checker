"""Writes a file listing every product where the gold labels and the keyword baseline disagree."""
import re
import sys

from app.extraction.clean import split_advisory
from app.rules.allergen_crosscheck import NEUTRAL_PHRASES, TRIGGERS, _detect
from eval.gold import GOLD_DIR, load_gold, version_from_args

TO_GOLD = {"crustacean shellfish": "shellfish", "tree nuts": "tree_nuts", "soybeans": "soy"}
FROM_GOLD = {v: k for k, v in TO_GOLD.items()}


def detect(text: str) -> set[str]:
    return {TO_GOLD.get(a, a) for a in _detect(text)}


def context(text: str, allergen: str) -> str:
    """Shows the first keyword hit for an allergen, with surrounding words."""
    masked = text
    for phrase in NEUTRAL_PHRASES:
        masked = re.sub(re.escape(phrase), lambda m: " " * len(m.group()), masked, flags=re.I)
    for word in TRIGGERS.get(FROM_GOLD.get(allergen, allergen), []):
        match = re.search(rf"\b{re.escape(word)}s?\b", masked, re.I)
        if match:
            start = max(0, match.start() - 45)
            return f"'{match.group()}' in ...{text[start:match.end() + 45]}..."
    return ""


def main() -> int:
    version = version_from_args(sys.argv)
    out_path = GOLD_DIR / f"review_disagreements_{version}.txt"
    df, problems = load_gold(version)
    if problems:
        print("Fix these labels first:", problems)
        return 1

    rows = list(df.itertuples())
    kept_preds = []
    for row in rows:
        kept, removed = split_advisory(row.ingredients_text)
        kept_preds.append((detect(kept), detect(removed), kept, removed))

    out, n_rows, n_missed, n_in_advisory, n_extra = [], 0, 0, 0, 0
    for i, row in enumerate(rows):
        pred, in_removed, kept, removed = kept_preds[i]
        missed = sorted(row.labels - pred)
        extra = sorted(pred - row.labels)
        if not missed and not extra:
            continue
        n_rows += 1

        out.append("=" * 90)
        out.append(f"{row.id}  {row.product_name}  [{row.source}]")
        out.append(f"  your labels : {', '.join(sorted(row.labels)) or '(none)'}")
        out.append(f"  keyword says: {', '.join(sorted(pred)) or '(none)'}")

        if len(missed) + len(extra) >= 3:
            neighbors = [kept_preds[j][0] for j in (i - 1, i + 1) if 0 <= j < len(rows)]
            if any(row.labels == n for n in neighbors):
                out.append("  !! 3+ disagreements, and your labels equal a NEIGHBOR row's keyword result: "
                           "possible paste shift in the spreadsheet")
            else:
                out.append("  !! 3+ disagreements on one product: check the labels or a long allergen list")

        for allergen in missed:
            n_missed += 1
            if allergen in in_removed:
                n_in_advisory += 1
                out.append(f"  MISSED {allergen}: appears only in a cross-contact clause that was removed")
            else:
                out.append(f"  MISSED {allergen}: no keyword for it anywhere in the text "
                           f"(label error, or a word the keyword list lacks)")
        for allergen in extra:
            n_extra += 1
            out.append(f"  EXTRA {allergen}: keyword {context(kept, allergen)}")

        out.append(f"  REMOVED CLAUSES: {removed or '(none)'}")
        out.append(f"  FULL TEXT: {' '.join(row.ingredients_text.split())[:900]}")

    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"Labels: {version}. {n_rows} of {len(rows)} products have at least one disagreement.")
    print(f"Missed allergens: {n_missed}, of which {n_in_advisory} appear only in a removed cross-contact clause.")
    print(f"Extra allergens flagged by the keywords: {n_extra}")
    print(f"\nWrote {out_path}. Open it in VS Code and review each product.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())