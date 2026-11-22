"""The allergen detector the app uses: fuzzy keyword matching plus the refined-oil policy."""
from app.rules.fuzzy_allergens import fuzzy_detect
from app.rules.oil_policy import PEANUT_OIL, SOY_OIL, oil_only


def detect_item(text: str) -> set[str]:
    """Allergens named in a piece of label text (an ingredient or a 'Contains' statement)."""
    found = set(fuzzy_detect(text))
    if oil_only(text, SOY_OIL, "soy"):
        found.discard("soybeans")
    if oil_only(text, PEANUT_OIL, "peanut"):
        found.discard("peanuts")
    return found