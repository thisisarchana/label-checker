"""Checks that each allergen in the ingredient list is named on the label."""
import re

from app.extraction.off_loader import split_ingredients
from app.rules.detection import detect_item
from app.rules.models import Finding, Verdict

CITATION = "FD&C Act sec. 403(w) (FALCPA, as amended by the FASTER Act)"
CLAIM = "Allergen declaration"

# Words that are the food-source name itself. An ingredient called "wheat flour" or "soy lecithin" already
# names its allergen. Derivatives such as whey or albumin do not, so they need "(milk)" or a Contains line.
SOURCE_NAMES = {
    "milk": ["milk"],
    "egg": ["egg"],
    "fish": ["fish", "anchovy", "anchovies", "salmon", "tuna", "cod", "tilapia", "sardine"],
    "crustacean shellfish": ["shellfish", "shrimp", "crab", "lobster", "crawfish", "prawn"],
    "tree nuts": ["tree nut", "almond", "cashew", "walnut", "pecan", "pistachio",
                  "hazelnut", "macadamia", "brazil nut"],
    "peanuts": ["peanut"],
    "wheat": ["wheat"],
    "soybeans": ["soy", "soya", "soybean"],
    "sesame": ["sesame"],
}
AMBIGUOUS = ["lecithin"]   # the source is unclear unless the ingredient says soy or sunflower


def _names_source(allergen: str, text: str) -> bool:
    return any(re.search(rf"\b{re.escape(name)}s?\b", text) for name in SOURCE_NAMES[allergen])


def _split_head_groups(item: str) -> tuple[str, list[str]]:
    """Separates an ingredient's own name from the parenthetical lists that follow it.

    OCR often drops a closing bracket; an unclosed group simply runs to the end of the text.
    """
    head: list[str] = []
    groups: list[str] = []
    current: list[str] = []
    depth = 0
    for char in item:
        if char in "([":
            if depth > 0:
                current.append(char)
            depth += 1
        elif char in ")]" and depth > 0:
            depth -= 1
            if depth == 0:
                groups.append("".join(current))
                current = []
            else:
                current.append(char)
        elif depth > 0:
            current.append(char)
        else:
            head.append(char)
    if depth > 0:
        groups.append("".join(current))
    return " ".join("".join(head).split()), groups


def _walk(item: str, declared: set[str], inherited: str, missing: dict, ambiguous: list) -> None:
    """Checks one ingredient, then each sub-ingredient on its own.

    An allergen counts as named if the ingredient, its own bracketed list, or an ancestor's name says
    its source. Judging sub-ingredients separately stops one long, badly bracketed line from excusing
    every allergen in it.
    """
    head, groups = _split_head_groups(item)
    context = " ".join([inherited, head, *groups]).lower()
    found = detect_item(head)
    for allergen in sorted(found):
        if allergen in declared or _names_source(allergen, context):
            continue
        missing.setdefault(allergen, []).append(head)
    if not found and any(word in head.lower() for word in AMBIGUOUS):
        ambiguous.append(head)
    for group in groups:
        for sub_item in split_ingredients(group):
            _walk(sub_item, declared, f"{inherited} {head}", missing, ambiguous)


def check_declared_allergens(ingredients: list[str], contains_statement: str | None) -> Finding:
    if not ingredients:
        return Finding(
            claim=CLAIM, verdict=Verdict.REVIEW, rule_citation=CITATION,
            explanation="No ingredient list was found, so allergens cannot be checked.", confidence=0.0,
        )

    declared = detect_item(contains_statement or "")
    missing: dict[str, list[str]] = {}
    ambiguous: list[str] = []
    for item in ingredients:
        _walk(item, declared, "", missing, ambiguous)

    if missing:
        parts = [f"{allergen} (from {', '.join(dict.fromkeys(items))})" for allergen, items in missing.items()]
        return Finding(
            claim=CLAIM, verdict=Verdict.FAIL, rule_citation=CITATION,
            explanation="The ingredient list contains allergens the label does not name: " + "; ".join(parts) + ".",
            evidence={"missing": missing, "declared": sorted(declared)},
        )
    if ambiguous:
        return Finding(
            claim=CLAIM, verdict=Verdict.REVIEW, rule_citation=CITATION,
            explanation="An ingredient may contain an allergen, but its source is unclear: "
            + ", ".join(dict.fromkeys(ambiguous)) + ".",
            evidence={"ambiguous": ambiguous}, confidence=0.5,
        )
    return Finding(
        claim=CLAIM, verdict=Verdict.PASS, rule_citation=CITATION,
        explanation="Every allergen found in the ingredient list is named on the label.",
        evidence={"declared": sorted(declared)},
    )