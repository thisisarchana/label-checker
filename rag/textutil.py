"""Small text helpers for building and checking retrieval test questions."""
import re

STOP = {
    "the", "and", "for", "that", "with", "from", "this", "what", "when", "which", "must", "may", "can",
    "does", "about", "would", "should", "there", "their", "have", "been", "were", "are", "not", "any",
    "such", "other", "than", "into", "only", "also", "each", "used", "use", "shall", "how", "who",
    "whose", "these", "those", "they", "them", "then", "will", "while", "where", "under", "over",
    "after", "before", "between", "being", "because", "more", "most", "some", "same",
}
WORD_RE = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def content_words(text: str) -> set[str]:
    return {w for w in words(text) if len(w) >= 4 and w not in STOP}


def longest_shared_run(a: list[str], b: list[str]) -> int:
    """Length of the longest run of consecutive words that appears in both lists."""
    best = 0
    previous = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best