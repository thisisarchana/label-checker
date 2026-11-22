"""Decides whether a photo was read well enough to trust a 'pass' verdict."""
from dataclasses import dataclass

# Chosen on 27 photos (OCR confidence correlated +0.80 with the share of ingredient words recovered).
# At 0.90 only 1 of 13 accepted photos was poorly read. A small sample: treat as a cautious starting point.
CONF_CUTOFF = 0.90
MIN_CHARS = 80   # confidence cannot see text the OCR never found, so a very short read is also suspect


@dataclass(frozen=True)
class ReadQuality:
    source: str                       # "typed" or "photo"
    passed: bool
    mean_confidence: float | None = None
    characters: int = 0
    reasons: tuple[str, ...] = ()


def assess_photo(mean_confidence: float, characters: int) -> ReadQuality:
    reasons = []
    if mean_confidence < CONF_CUTOFF:
        reasons.append(f"OCR confidence {mean_confidence:.2f} is below {CONF_CUTOFF:.2f}")
    if characters < MIN_CHARS:
        reasons.append(f"only {characters} characters were read (minimum {MIN_CHARS})")
    return ReadQuality("photo", not reasons, mean_confidence, characters, tuple(reasons))


def assess_typed(text: str) -> ReadQuality:
    return ReadQuality("typed", True, None, len(text))