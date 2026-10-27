"""Turns an Open Food Facts product record into a LabelData for the rule engine."""
import html
import math
import re

from app.rules.label import LabelData

# "Contains: milk" is an allergen statement. "contains 2% or less of..." is not.
CONTAINS_RE = re.compile(
    r"(?<!may )\bcontains?\b(?!\s*(?:\d|<|less|one or more))", re.I
)
ADVISORY_RE = re.compile(
    r"\b(may contain|produced in|manufactured in|made in|processed in|"
    r"packaged in|allergen information|allergy advice)\b",
    re.I,
)
PREFIX_RE = re.compile(r"^\s*ingredients?\s*:?\s*", re.I)


def _as_list(value) -> list:
    return [] if value is None else list(value)


def split_ingredients(text: str) -> list[str]:
    """Splits on commas and semicolons that are not inside parentheses or brackets."""
    items, current, depth = [], [], 0
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch in ",;" and depth == 0:
            items.append("".join(current))
            current = []
        else:
            current.append(ch)
    items.append("".join(current))
    cleaned = [item.strip(" .\r\n\t") for item in items]
    return [item for item in cleaned if item]


def parse_label_text(text: str) -> tuple[list[str], str | None]:
    """Returns (ingredient list, 'Contains: ...' statement or None)."""
    text = html.unescape(text)
    text = PREFIX_RE.sub("", " ".join(text.split()))

    cut = len(text)
    for pattern in (CONTAINS_RE, ADVISORY_RE):
        match = pattern.search(text)
        if match:
            cut = min(cut, match.start())
    body, tail = text[:cut], text[cut:]

    statement = None
    match = CONTAINS_RE.search(tail)
    if match:
        statement = tail[match.start():].split(".")[0].strip()

    return split_ingredients(body), statement


def has_gluten_free_claim(labels_tags) -> bool:
    return any(
        ("gluten-free" in tag or tag.startswith("en:no-gluten")) and "contain" not in tag
        for tag in _as_list(labels_tags)
    )


def to_label_data(record: dict) -> LabelData:
    ingredients, statement = parse_label_text(record.get("ingredients_text") or "")

    protein = record.get("protein_serving_g")
    if protein is None or (isinstance(protein, float) and math.isnan(protein)):
        protein = None

    claims = []
    if has_gluten_free_claim(record.get("labels_tags")):
        claims.append("Gluten-Free")

    return LabelData(
        claims=claims,
        ingredients=ingredients,
        allergen_statement=statement,
        protein_g=protein,
        gluten_ppm=None,   # Open Food Facts has no lab test results
    )