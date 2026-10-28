"""Text cleanup shared by the allergen baselines and models."""
import html
import re

# Clauses that describe cross-contact, not ingredients. The gold labels ignore them.
ADVISORY_CLAUSE_RE = re.compile(
    r"\b(?:may contain|might contain|produced in|manufactured in|made in|"
    r"processed in|packaged in|made on|manufactured on|processed on|"
    r"shared equipment|shared facility)\b",
    re.I,
)


def split_advisory(text: str | None) -> tuple[str, str]:
    """Returns (text without cross-contact clauses, the clauses that were removed)."""
    text = " ".join(html.unescape(text or "").split())
    kept, removed, pos = [], [], 0
    for match in ADVISORY_CLAUSE_RE.finditer(text):
        if match.start() < pos:
            continue
        kept.append(text[pos:match.start()])
        end = text.find(".", match.end())
        stop = len(text) if end == -1 else end + 1
        removed.append(text[match.start():stop])
        pos = stop
    kept.append(text[pos:])
    return " ".join("".join(kept).split()), " ".join(removed)


def strip_advisory(text: str | None) -> str:
    """Removes 'may contain ...' style sentences; keeps ingredients and 'Contains:' lines."""
    return split_advisory(text)[0]