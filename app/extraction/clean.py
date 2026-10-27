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


def strip_advisory(text: str | None) -> str:
    """Removes 'may contain ...' style sentences; keeps ingredients and 'Contains:' lines."""
    text = " ".join(html.unescape(text or "").split())
    kept, pos = [], 0
    for match in ADVISORY_CLAUSE_RE.finditer(text):
        if match.start() < pos:
            continue
        kept.append(text[pos:match.start()])
        end = text.find(".", match.end())
        pos = len(text) if end == -1 else end + 1
    kept.append(text[pos:])
    return " ".join("".join(kept).split())