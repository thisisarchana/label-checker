"""Measures which image preprocessing recovers the most ingredient text from the gold photos."""
import re
import time

import cv2
import numpy as np
from rapidfuzz.distance import Levenshtein
from rapidocr import RapidOCR

from eval.gold import load_gold
from ocr.run_ocr import IMAGE_DIR

WORD_RE = re.compile(r"[a-z]{3,}")
MAX_SIDE = 4000   # keeps upscaled photos from using too much memory


def upscale(img: np.ndarray, factor: float = 2.0) -> np.ndarray:
    scale = min(factor, MAX_SIDE / max(img.shape[:2]))
    if scale <= 1.05:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def contrast(img: np.ndarray) -> np.ndarray:
    """Local contrast equalization on the lightness channel, which helps with glare and faint print."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
    return cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)


def tiles(img: np.ndarray) -> list[np.ndarray]:
    """Four overlapping quarters, so tiny print is read at a larger relative size."""
    h, w = img.shape[:2]
    th, tw = int(h * 0.55), int(w * 0.55)
    return [img[y:y + th, x:x + tw] for y in (0, h - th) for x in (0, w - tw)]


VARIANTS = {
    "original": lambda img: [img],
    "upscale 2x": lambda img: [upscale(img)],
    "contrast (CLAHE)": lambda img: [contrast(img)],
    "2x2 tiles": tiles,
    "upscale + contrast": lambda img: [contrast(upscale(img))],
}


def ocr_text(engine: RapidOCR, pieces: list[np.ndarray]) -> str:
    return " ".join(" ".join(engine(piece).txts or []) for piece in pieces)


def word_recall(database_text: str, ocr: str) -> float:
    """Share of the database ingredient words that the OCR text contains, allowing one typo."""
    wanted = set(WORD_RE.findall(database_text.lower()))
    found = set(WORD_RE.findall(ocr.lower()))
    if not wanted:
        return 1.0
    hits = 0
    for word in wanted:
        tolerance = 0 if len(word) < 5 else 1
        if word in found or (tolerance and any(
            abs(len(other) - len(word)) <= tolerance and Levenshtein.distance(word, other) <= tolerance
            for other in found
        )):
            hits += 1
    return hits / len(wanted)


def main() -> None:
    gold, _ = load_gold("adjudicated")
    gold = gold[gold.id.apply(lambda i: (IMAGE_DIR / f"{i}.jpg").exists())]
    if gold.empty:
        print(f"No images in {IMAGE_DIR}. Run python -m ocr.fetch_gold_images first.")
        return

    engine = RapidOCR()
    images = {gid: cv2.imread(str(IMAGE_DIR / f"{gid}.jpg")) for gid in gold.id}
    print(f"{len(images)} photos, {len(VARIANTS)} variants (this takes several minutes)\n")
    print(f"{'variant':22s} {'word recall':>12s} {'photos under 0.70':>18s} {'seconds':>9s}")
    for name, make in VARIANTS.items():
        start = time.time()
        scores = [word_recall(text, ocr_text(engine, make(images[gid])))
                  for gid, text in zip(gold.id, gold.ingredients_text) if images[gid] is not None]
        print(f"{name:22s} {np.mean(scores):12.2f} {sum(s < 0.70 for s in scores):18d} "
              f"{time.time() - start:9.0f}")


if __name__ == "__main__":
    main()