"""Splits label text into an ingredient list and a 'Contains' statement."""
import re

from app.extraction.clean import strip_advisory
from app.extraction.off_loader import CONTAINS_RE, PREFIX_RE, split_ingredients

# OCR misspells the heading ("INGREDTENTS:"), so match any word that starts like "ingredients" and ends in a colon.
LIST_START_RE = re.compile(r"\bingr\w{3,9}\s*[:;]", re.I)


def extract_fields(raw_text: str) -> tuple[list[str], str | None]:
    """Cross-contact warnings are dropped, so a 'may contain' phrase cannot hide later ingredients.

    Text before the 'Ingredients:' heading (a product name, nutrition fragments) is ignored.
    """
    text = strip_advisory(raw_text)

    contains = CONTAINS_RE.search(text)
    statement = text[contains.start():].split(".")[0].strip() if contains else None

    heading = LIST_START_RE.search(text)
    prefix = PREFIX_RE.match(text)
    start = heading.end() if heading else (prefix.end() if prefix else 0)
    end = contains.start() if contains and contains.start() >= start else len(text)
    return split_ingredients(text[start:end]), statement