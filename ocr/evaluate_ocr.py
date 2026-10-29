"""Runs the allergen systems on real OCR text and on the clean database text for the same products."""
import sys

import joblib
from rapidfuzz import fuzz

from app.extraction.clean import strip_advisory
from eval.gold import ALLERGENS, PATHS, load_gold
from eval.metrics import score
from ml.compare_systems import keyword_with_oil_rule
from ml.robustness import fuzzy_with_oil_rule
from ml.train_tfidf import MODELS, predict_sets
from ocr.run_ocr import OCR_DIR

MAX_LISTED = 15


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def run_systems(texts: list[str], char: dict) -> dict[str, list[set[str]]]:
    cleaned = [strip_advisory(t) for t in texts]
    keyword = [keyword_with_oil_rule(t) for t in texts]
    fuzzy = [fuzzy_with_oil_rule(t) for t in texts]
    chars = predict_sets(char["vectorizer"], char["model"], char["thresholds"], cleaned)
    return {
        "keyword + oil rule": keyword,
        "fuzzy keyword + oil rule": fuzzy,
        "tfidf characters": chars,
        "hybrid (fuzzy kw | chars)": [f | c for f, c in zip(fuzzy, chars)],
    }


def fmt(result: dict) -> str:
    return f"{result['micro_precision']:.2f} / {result['micro_recall']:.2f} / {result['micro_f1']:.2f}"


def main() -> int:
    version = sys.argv[1] if len(sys.argv) > 1 else "adjudicated"
    if version not in PATHS:
        raise SystemExit(f"Unknown version '{version}'. Use: {', '.join(PATHS)}")
    gold, problems = load_gold(version)
    if problems:
        print("Fix unrecognized gold labels first:", problems)
        return 1

    gold = gold[gold.id.apply(lambda i: (OCR_DIR / f"{i}.txt").exists())].reset_index(drop=True)
    if gold.empty:
        print(f"No OCR text found in {OCR_DIR}. Run python -m ocr.run_ocr first.")
        return 1
    ocr_texts = [(OCR_DIR / f"{i}.txt").read_text(encoding="utf-8") for i in gold.id]
    clean_texts = gold.ingredients_text.tolist()
    y = gold.labels.tolist()
    positives = sum(len(s) for s in y)

    coverage = [
        fuzz.partial_ratio(normalize(db), normalize(ocr)) if ocr.strip() else 0.0
        for db, ocr in zip(clean_texts, ocr_texts)
    ]
    print(f"Products with OCR text: {len(gold)}   (gold allergen labels among them: {positives})")
    print(f"How much of the database ingredient text appears in the OCR text (0-100): "
          f"mean {sum(coverage) / len(coverage):.0f}, lowest {min(coverage):.0f}, "
          f"products under 80: {sum(c < 80 for c in coverage)}")

    char = joblib.load(MODELS / "tfidf_char_policy.joblib")
    on_clean = run_systems(clean_texts, char)
    on_ocr = run_systems(ocr_texts, char)

    print(f"\nMICRO precision / recall / F1 on the same {len(gold)} products ({version} labels)")
    print(f"{'system':28s} {'database text':>22s}   {'real OCR text':>22s}")
    for name in on_clean:
        print(f"{name:28s} {fmt(score(y, on_clean[name], ALLERGENS)):>22s}   "
              f"{fmt(score(y, on_ocr[name], ALLERGENS)):>22s}")

    for name in ("fuzzy keyword + oil rule", "hybrid (fuzzy kw | chars)"):
        missed = [(gid, a) for gid, labels, pred in zip(gold.id, y, on_ocr[name])
                  for a in sorted(labels - pred)]
        print(f"\nAllergens missed on real OCR text by {name}: {len(missed)}")
        for gid, allergen in missed[:MAX_LISTED]:
            print(f"  {gid} {allergen}")
    print(f"\nSmall sample ({len(gold)} products): treat these numbers as a sanity check, not a benchmark.")
    return 0


if __name__ == "__main__":
    sys.exit(main())