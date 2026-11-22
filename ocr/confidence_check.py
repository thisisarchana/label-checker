"""Does OCR confidence predict how much of the ingredient text was read? Tests a read-quality gate."""
import numpy as np
import pandas as pd
from rapidocr import RapidOCR

from eval.gold import load_gold
from ocr.compare_preprocessing import word_recall
from ocr.run_ocr import IMAGE_DIR, OCR_DIR

CUTOFFS = [0.80, 0.85, 0.90, 0.95]
MIN_CHARS = 80          # a very short read is suspicious whatever the confidence
GOOD_RECALL = 0.70      # below this, too much of the ingredient list was missed to trust a pass verdict


def read_photo(engine: RapidOCR, path) -> dict:
    result = engine(str(path))
    texts = list(result.txts or [])
    scores = list(result.scores or [])
    return {
        "text": " ".join(texts),
        "lines": len(texts),
        "chars": sum(len(t) for t in texts),
        "mean_conf": float(np.mean(scores)) if scores else 0.0,
        "min_conf": float(np.min(scores)) if scores else 0.0,
    }


def gate_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cutoff in CUTOFFS:
        accepted = frame[(frame.mean_conf >= cutoff) & (frame.chars >= MIN_CHARS)]
        rejected = frame.drop(accepted.index)
        rows.append({
            "cutoff": cutoff,
            "accepted": len(accepted),
            "recall of accepted": accepted.recall.mean() if len(accepted) else np.nan,
            "accepted but poorly read": int((accepted.recall < GOOD_RECALL).sum()),
            "recall of rejected": rejected.recall.mean() if len(rejected) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    gold, _ = load_gold("adjudicated")
    gold = gold[gold.id.apply(lambda i: (IMAGE_DIR / f"{i}.jpg").exists())]
    if gold.empty:
        print(f"No images in {IMAGE_DIR}. Run python -m ocr.fetch_gold_images first.")
        return

    engine = RapidOCR()
    rows = []
    for row in gold.itertuples():
        read = read_photo(engine, IMAGE_DIR / f"{row.id}.jpg")
        read["id"] = row.id
        read["recall"] = word_recall(row.ingredients_text, read["text"])
        rows.append(read)
        print(f"{row.id}: recall {read['recall']:.2f}, confidence {read['mean_conf']:.2f}")
    frame = pd.DataFrame(rows).drop(columns="text")

    OCR_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OCR_DIR.parent / "confidence_table.csv", index=False)

    print(f"\nPHOTOS, worst read first ({len(frame)} photos)")
    print(frame.sort_values("recall")[["id", "recall", "mean_conf", "min_conf", "lines", "chars"]]
          .round(2).to_string(index=False))

    if len(frame) >= 5:
        print("\nSpearman correlation with the share of ingredient words recovered:")
        for column in ("mean_conf", "min_conf", "chars", "lines"):
            print(f"  {column:10s} {frame[column].corr(frame.recall, method='spearman'):+.2f}")

    print(f"\nGATE: accept a photo if mean confidence >= cutoff and at least {MIN_CHARS} characters were read")
    print(f"      'poorly read' means under {GOOD_RECALL:.0%} of the database ingredient words were recovered")
    print(gate_table(frame).round(2).to_string(index=False))
    print("\nSmall sample: use this to pick a cautious cutoff, not to claim a precise accuracy.")


if __name__ == "__main__":
    main()