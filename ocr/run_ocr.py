"""Runs OCR on the downloaded ingredient photos and saves the text."""
from pathlib import Path

from rapidocr import RapidOCR

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "images" / "gold"
OCR_DIR = ROOT / "data" / "ocr" / "gold"


def read_text(engine: RapidOCR, path: Path) -> str:
    result = engine(str(path))
    return " ".join(result.txts) if result.txts else ""


def main() -> None:
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(IMAGE_DIR.glob("*.jpg"))
    if not images:
        print(f"No images found in {IMAGE_DIR}. Run python -m ocr.fetch_gold_images first.")
        return

    engine = RapidOCR()
    empty = 0
    for path in images:
        target = OCR_DIR / f"{path.stem}.txt"
        if target.exists():
            continue
        text = read_text(engine, path)
        target.write_text(text, encoding="utf-8")
        empty += int(not text)
        print(f"{path.stem}: {len(text)} characters")
    print(f"\nOCR text for {len(images)} images saved to {OCR_DIR} ({empty} came back empty)")


if __name__ == "__main__":
    main()