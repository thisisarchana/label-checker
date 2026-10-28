"""Text cleanup shared by the allergen baselines and models."""
import html
import re

# Cross-contact warnings, not ingredients. The gold labels ignore them.
# "may contain up to 2% water" is a processing note, so it is not matched.
ADVISORY_CLAUSE_RE = re.compile(
    r"\b(?:may contain(?!\s+(?:up to|\d|less))|might contain|produced in|"
    r"manufactured in|made in|processed in|packaged in|made on|"
    r"manufactured on|processed on|shared equipment|shared facility)\b",
    re.I,
)
MAX_CLAUSE_CHARS = 250   # safety net: never delete a huge span because a period is missing


def _clause_end(text: str, start: int) -> int:
    """End of the warning: the sentence end, the enclosing bracket, or a length cap."""
    limit = min(len(text), start + MAX_CLAUSE_CHARS)
    depth = 0
    for i in range(start, limit):
        ch = text[i]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            if depth == 0:
                return i                 # closes the bracket the warning sits in
            depth -= 1
        elif ch == "." and depth == 0:
            return i + 1
    return limit


def split_advisory(text: str | None) -> tuple[str, str]:
    """Returns (text without cross-contact clauses, the clauses that were removed)."""
    text = " ".join(html.unescape(text or "").split())
    kept, removed, pos = [], [], 0
    for match in ADVISORY_CLAUSE_RE.finditer(text):
        if match.start() < pos:
            continue
        kept.append(text[pos:match.start()])
        stop = _clause_end(text, match.end())
        removed.append(text[match.start():stop])
        pos = stop
    kept.append(text[pos:])
    return " ".join("".join(kept).split()), " ".join(removed)


def strip_advisory(text: str | None) -> str:
    """Removes cross-contact warnings; keeps ingredients and 'Contains:' lines."""
    return split_advisory(text)[0]