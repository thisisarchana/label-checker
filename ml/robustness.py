"""How do the systems hold up when label text is corrupted the way OCR corrupts it?"""
import random
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from app.extraction.clean import strip_advisory
from eval.gold import ALLERGENS, PATHS, load_gold
from app.rules.fuzzy_allergens import fuzzy_detect
from eval.metrics import score
from ml.build_training_data import PEANUT_OIL, SOY_OIL, oil_only
from ml.compare_systems import keyword_with_oil_rule
from ml.policy import apply_policy_labels
from ml.train_tfidf import DATA, MODELS, best_thresholds, predict_sets

NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10]   # chance that each letter is corrupted
N_SEEDS = 5                               # independent noise draws per level
SEED = "ocr-noise-v2"
TO_GOLD = {"crustacean shellfish": "shellfish", "tree nuts": "tree_nuts", "soybeans": "soy"}
CONFUSIONS = {"l": "1", "i": "l", "o": "0", "e": "c", "s": "5", "b": "6",
              "m": "rn", "u": "v", "a": "o", "t": "f"}


def ocr_noise(text: str, rate: float, rng: random.Random) -> str:
    """Simulated OCR errors: look-alike swaps, dropped letters, random letters."""
    out = []
    for ch in text:
        if ch.isalpha() and rng.random() < rate:
            action = rng.random()
            if action < 0.5 and ch.lower() in CONFUSIONS:
                out.append(CONFUSIONS[ch.lower()])
            elif action < 0.8:
                continue
            else:
                out.append(rng.choice("abcdefghijklmnopqrstuvwxyz"))
        else:
            out.append(ch)
    return "".join(out)


def fuzzy_with_oil_rule(raw_text: str) -> set[str]:
    """Fuzzy keyword matching plus the same oil policy as the exact version."""
    text = strip_advisory(raw_text)
    pred = {TO_GOLD.get(a, a) for a in fuzzy_detect(text)}
    if oil_only(text, SOY_OIL, "soy"):
        pred.discard("soy")
    if oil_only(text, PEANUT_OIL, "peanut"):
        pred.discard("peanuts")
    return pred


def char_model(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict:
    path = MODELS / "tfidf_char_policy.joblib"
    if path.exists():
        return joblib.load(path)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 sublinear_tf=True, max_features=300_000)
    x_train = vectorizer.fit_transform(train_df.text)
    model = OneVsRestClassifier(LogisticRegression(C=10, solver="liblinear", max_iter=1000))
    model.fit(x_train, train_df[ALLERGENS].values)
    prob = model.predict_proba(vectorizer.transform(valid_df.text))
    bundle = {"vectorizer": vectorizer, "model": model,
              "thresholds": best_thresholds(valid_df[ALLERGENS].values, prob)}
    MODELS.mkdir(exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


def main() -> int:
    version = sys.argv[1] if len(sys.argv) > 1 else "adjudicated"
    if version not in PATHS:
        raise SystemExit(f"Unknown version '{version}'. Use: {', '.join(PATHS)}")
    gold, problems = load_gold(version)
    if problems:
        print("Fix unrecognized gold labels first:", problems)
        return 1
    y = gold.labels.tolist()

    word = joblib.load(MODELS / "tfidf_policy.joblib")
    print("Preparing the character n-gram model (first run trains it, a few minutes)...")
    char = char_model(apply_policy_labels(pd.read_parquet(DATA / "train.parquet")),
                      apply_policy_labels(pd.read_parquet(DATA / "valid.parquet")))

    names = ["keyword + oil rule", "fuzzy keyword + oil rule", "tfidf characters",
             "hybrid (keyword | chars)", "hybrid (fuzzy kw | chars)"]
    metrics = ("precision", "recall", "f1")
    results = {m: {n: [[] for _ in NOISE_LEVELS] for n in names} for m in metrics}
    for seed in range(N_SEEDS):
        for i, rate in enumerate(NOISE_LEVELS):
            noisy = [ocr_noise(t, rate, random.Random(f"{SEED}-{seed}-{rate}-{gid}"))
                     for gid, t in zip(gold.id, gold.ingredients_text)]
            texts = [strip_advisory(t) for t in noisy]
            keyword = [keyword_with_oil_rule(t) for t in noisy]
            fuzzy = [fuzzy_with_oil_rule(t) for t in noisy]
            chars = predict_sets(char["vectorizer"], char["model"], char["thresholds"], texts)
            systems = {
                "keyword + oil rule": keyword,
                "fuzzy keyword + oil rule": fuzzy,
                "tfidf characters": chars,
                "hybrid (keyword | chars)": [k | c for k, c in zip(keyword, chars)],
                "hybrid (fuzzy kw | chars)": [k | c for k, c in zip(fuzzy, chars)],
            }
            for name, preds in systems.items():
                r = score(y, preds, ALLERGENS)
                results["precision"][name][i].append(r["micro_precision"])
                results["recall"][name][i].append(r["micro_recall"])
                results["f1"][name][i].append(r["micro_f1"])

    header = f"{'system':28s}" + "".join(f"{f'{r:.0%} noise':>13s}" for r in NOISE_LEVELS)
    for metric in ("f1", "recall", "precision"):
        print(f"\nMICRO {metric.upper()} on {len(gold)} gold products ({version} labels), "
              f"mean ± sd over {N_SEEDS} noise draws")
        print(header)
        for name in names:
            cells = "".join(
                f"{f'{np.mean(v):.2f}±{np.std(v):.2f}':>13s}" for v in results[metric][name]
            )
            print(f"{name:28s}{cells}")
    print("\nNoise is simulated (look-alike swaps, dropped and random letters), not real OCR output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())