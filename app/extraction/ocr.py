"""OCR for ingredient photos, returning the text and the engine's average confidence."""
from functools import lru_cache


@lru_cache(maxsize=1)
def _engine():
    from rapidocr import RapidOCR

    return RapidOCR()


def read_photo(image) -> tuple[str, float]:
    """`image` is a file path or a numpy array. Returns (text, mean confidence)."""
    result = _engine()(image)
    texts = list(result.txts or [])
    scores = list(result.scores or [])
    return " ".join(texts), (sum(scores) / len(scores) if scores else 0.0)