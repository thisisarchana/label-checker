"""The labeling policy for refined oils: soy or peanut named only as 'soybean oil' or 'peanut oil' is not flagged."""
import re

SOY_OIL = re.compile(r"soy(?:a|bean)?\s+oil", re.I)
PEANUT_OIL = re.compile(r"peanut\s+oil", re.I)


def oil_only(text: str, oil: re.Pattern, word: str) -> bool:
    """True if the allergen word appears in the text only as part of the oil name."""
    return bool(oil.search(text)) and not re.search(rf"\b{word}\w*", oil.sub(" ", text), re.I)