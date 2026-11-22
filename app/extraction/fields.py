"""Splits label text into an ingredient list and a 'Contains' statement."""
from app.extraction.clean import strip_advisory
from app.extraction.off_loader import CONTAINS_RE, PREFIX_RE, split_ingredients


def extract_fields(raw_text: str) -> tuple[list[str], str | None]:
    """Cross-contact warnings are dropped, so a 'may contain' phrase cannot hide later ingredients."""
    text = PREFIX_RE.sub("", strip_advisory(raw_text))
    match = CONTAINS_RE.search(text)
    if not match:
        return split_ingredients(text), None
    statement = text[match.start():].split(".")[0].strip()
    return split_ingredients(text[:match.start()]), statement