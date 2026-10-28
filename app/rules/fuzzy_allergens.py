"""Keyword allergen detection that tolerates small spelling and OCR errors."""
import re

from rapidfuzz.distance import Levenshtein

from app.rules.allergen_crosscheck import NEUTRAL_PHRASES, TRIGGERS, _detect

TOKEN_RE = re.compile(r"[a-z0-9]+")   # OCR often swaps letters for digits


def tolerance(word: str) -> int:
    """Short words must match exactly: one wrong letter would turn 'milk' into 'silk'."""
    if len(word) < 5:
        return 0
    return 1 if len(word) < 9 else 2


def _close(token: str, word: str) -> bool:
    tol = tolerance(word)
    if abs(len(token) - len(word)) > tol + 1:
        return False
    for target in (word, word + "s"):
        if Levenshtein.distance(token, target) <= tol:
            return True
    return False


def fuzzy_detect(text: str) -> set[str]:
    """Exact keyword matches plus single-word matches within the edit-distance tolerance."""
    found = set(_detect(text))
    lowered = text.lower()
    for phrase in NEUTRAL_PHRASES:
        lowered = lowered.replace(phrase, " ")
    tokens = set(TOKEN_RE.findall(lowered))

    for allergen, words in TRIGGERS.items():
        if allergen in found:
            continue
        for word in words:
            if " " in word:
                continue   # multi-word triggers are matched exactly by _detect
            if any(_close(token, word) for token in tokens):
                found.add(allergen)
                break
    return found