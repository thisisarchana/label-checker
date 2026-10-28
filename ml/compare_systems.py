"""Compares keyword rules, the TF-IDF model, and a rules+model hybrid on the frozen gold set."""
import sys

import joblib

from app.extraction.clean import strip_advisory
from eval.gold import ALLERGENS, PATHS, load_gold
from eval.gold_baseline import keyword_predict
from eval.metrics import print_report, score
from ml.build_training_data import PEANUT_OIL, SOY_OIL, oil_only
from ml.train_tfidf import MODELS, predict_sets, summarize

MAX_LISTED = 12


def keyword_with_oil_rule(raw_text: str) -> set[str]:
    """The keyword rules plus the same oil policy the gold labels follow."""
    pred = keyword_predict(raw_text)
    text = strip_advisory(raw_text)
    if oil_only(text, SOY_OIL, "soy"):
        pred.discard("soy")
    if oil_only(text, PEANUT_OIL, "peanut"):
        pred.discard("peanuts")
    return pred


def main() -> int:
    version = sys.argv[1] if len(sys.argv) > 1 else "adjudicated"
    if version not in PATHS:
        raise SystemExit(f"Unknown version '{version}'. Use: {', '.join(PATHS)}")
    gold, problems = load_gold(version)
    if problems:
        print("Fix unrecognized gold labels first:", problems)
        return 1
    gold = gold.assign(text=gold.ingredients_text.map(strip_advisory))

    bundle = joblib.load(MODELS / "tfidf_policy.joblib")
    model = predict_sets(bundle["vectorizer"], bundle["model"], bundle["thresholds"], gold.text)
    keyword = [keyword_predict(t) for t in gold.ingredients_text]
    keyword_oil = [keyword_with_oil_rule(t) for t in gold.ingredients_text]
    hybrid = [k | m for k, m in zip(keyword_oil, model)]

    systems = {
        "keyword baseline": keyword,
        "keyword + oil rule": keyword_oil,
        "tfidf (policy labels)": model,
        "hybrid (keyword+oil | tfidf)": hybrid,
    }
    print(f"SUMMARY  (micro precision / recall / F1, gold labels: {version})")
    print(f"{'system':30s} {'random 100':>20s}   {'all 148':>20s}")
    for name, preds in systems.items():
        print(f"{name:30s}" + summarize("", gold, preds)[26:])
    print_report(f"HYBRID [{version} labels], all gold products",
                 score(gold.labels.tolist(), hybrid, ALLERGENS))

    helped, missed = [], []
    for row, k, m in zip(gold.itertuples(), keyword_oil, model):
        for allergen in sorted(row.labels):
            if allergen not in k and allergen in m:
                helped.append(f"{row.id} {row.product_name[:30]:30s} {allergen}")
            if allergen in k and allergen not in m:
                missed.append(f"{row.id} {row.product_name[:30]:30s} {allergen}")
    print(f"\nGold allergens the TF-IDF model finds but the keyword rules miss: {len(helped)}")
    for line in helped[:MAX_LISTED]:
        print("  " + line)
    print(f"\nGold allergens the keyword rules find but the TF-IDF model misses: {len(missed)}")
    for line in missed[:MAX_LISTED]:
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())