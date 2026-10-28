"""TF-IDF + logistic regression allergen models, scored on the frozen gold set."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from app.extraction.clean import strip_advisory
from eval.gold import ALLERGENS, PATHS, load_gold
from eval.gold_baseline import keyword_predict
from eval.metrics import print_report, score
from ml.policy import apply_policy_labels

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
THRESHOLDS = np.arange(0.10, 0.91, 0.05)


def f1_at(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> float:
    pred = prob >= threshold
    tp = int((pred & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    fn = int((~pred & (y_true == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def best_thresholds(y_valid: np.ndarray, prob_valid: np.ndarray) -> list[float]:
    """Per-allergen threshold that maximizes F1 on the validation split (never on gold)."""
    return [
        float(THRESHOLDS[int(np.argmax([f1_at(y_valid[:, j], prob_valid[:, j], t) for t in THRESHOLDS]))])
        for j in range(y_valid.shape[1])
    ]


def train(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    x_train = vectorizer.fit_transform(train_df.text)
    model = OneVsRestClassifier(LogisticRegression(C=10, solver="liblinear", max_iter=1000))
    model.fit(x_train, train_df[ALLERGENS].values)
    prob_valid = model.predict_proba(vectorizer.transform(valid_df.text))
    return vectorizer, model, best_thresholds(valid_df[ALLERGENS].values, prob_valid)


def predict_sets(vectorizer, model, thresholds, texts) -> list[set[str]]:
    prob = model.predict_proba(vectorizer.transform(texts))
    return [{a for a, t, p in zip(ALLERGENS, thresholds, row) if p >= t} for row in prob]


def summarize(name: str, gold: pd.DataFrame, preds: list[set[str]]) -> str:
    rand = (gold.source == "random").to_numpy()
    y = gold.labels.tolist()
    rows = []
    for mask in (rand, np.ones(len(gold), dtype=bool)):
        idx = np.where(mask)[0]
        r = score([y[i] for i in idx], [preds[i] for i in idx], ALLERGENS)
        rows.append(f"{r['micro_precision']:.2f} / {r['micro_recall']:.2f} / {r['micro_f1']:.2f}")
    return f"{name:26s} {rows[0]:>20s}   {rows[1]:>20s}"


def main() -> int:
    version = sys.argv[1] if len(sys.argv) > 1 else "adjudicated"
    if version not in PATHS:
        raise SystemExit(f"Unknown version '{version}'. Use: {', '.join(PATHS)}")
    gold, problems = load_gold(version)
    if problems:
        print("Fix unrecognized gold labels first:", problems)
        return 1
    gold = gold.assign(text=gold.ingredients_text.map(strip_advisory))

    train_df = pd.read_parquet(DATA / "train.parquet")
    valid_df = pd.read_parquet(DATA / "valid.parquet")
    variants = {
        "tfidf (OFF tags)": (train_df, valid_df),
        "tfidf (policy labels)": (apply_policy_labels(train_df), apply_policy_labels(valid_df)),
    }

    results = {"keyword baseline": [keyword_predict(t) for t in gold.ingredients_text]}
    for name, (tr, va) in variants.items():
        print(f"\nTraining {name} on {len(tr):,} products...")
        vectorizer, model, thresholds = train(tr, va)
        print("  thresholds chosen on validation:",
              ", ".join(f"{a}={t:.2f}" for a, t in zip(ALLERGENS, thresholds)))
        results[name] = predict_sets(vectorizer, model, thresholds, gold.text)
        print_report(f"{name.upper()} [{version} labels], all gold products",
                     score(gold.labels.tolist(), results[name], ALLERGENS))
        if "policy" in name:
            MODELS.mkdir(exist_ok=True)
            joblib.dump({"vectorizer": vectorizer, "model": model, "thresholds": thresholds},
                        MODELS / "tfidf_policy.joblib")

    print(f"\nSUMMARY  (micro precision / recall / F1, gold labels: {version})")
    print(f"{'system':26s} {'random 100':>20s}   {'all 148':>20s}")
    for name, preds in results.items():
        print(summarize(name, gold, preds))
    return 0


if __name__ == "__main__":
    sys.exit(main())