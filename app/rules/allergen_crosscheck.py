import re

from app.rules.models import Finding, Verdict

CITATION = "FD&C Act sec. 403(w) (FALCPA, as amended by the FASTER Act)"

# Words in an ingredient list that signal a major allergen.
TRIGGERS = {
    "milk": ["milk", "whey", "casein", "caseinate", "butter", "cheese",
             "cream", "lactose", "yogurt", "ghee", "buttermilk"],
    "egg": ["egg", "albumin", "mayonnaise", "meringue"],
    "fish": ["fish", "anchovy", "anchovies", "salmon", "tuna", "cod",
             "tilapia", "sardine"],
    "crustacean shellfish": ["shrimp", "crab", "lobster", "crawfish", "prawn"],
    "tree nuts": ["almond", "cashew", "walnut", "pecan", "pistachio",
                  "hazelnut", "macadamia", "brazil nut"],
    "peanuts": ["peanut"],
    "wheat": ["wheat", "semolina", "durum", "spelt", "farina"],
    "soybeans": ["soy", "soya", "soybean", "tofu", "edamame", "miso"],
    "sesame": ["sesame", "tahini"],
}

# How each allergen may legitimately be named in a "Contains" statement.
ALIASES = {
    "milk": ["milk"],
    "egg": ["egg"],
    "fish": ["fish"],
    "crustacean shellfish": ["shellfish", "crustacean"],
    "tree nuts": ["tree nut"] + TRIGGERS["tree nuts"],
    "peanuts": ["peanut"],
    "wheat": ["wheat"],
    "soybeans": ["soy"],
    "sesame": ["sesame"],
}

# Phrases that contain a trigger word but are not the allergen.
NEUTRAL_PHRASES = ["cocoa butter", "shea butter", "coconut milk",
                   "coconut cream", "cream of tartar", "butternut"]

# Ingredients whose allergen source is unclear without more information.
AMBIGUOUS = ["lecithin"]


def _detect(text: str) -> set[str]:
    t = text.lower()
    for phrase in NEUTRAL_PHRASES:
        t = t.replace(phrase, " ")
    found = set()
    for allergen, words in TRIGGERS.items():
        if any(re.search(rf"\b{re.escape(w)}s?\b", t) for w in words):
            found.add(allergen)
    return found


def _is_named(allergen: str, text: str) -> bool:
    t = text.lower()
    return any(re.search(rf"\b{re.escape(a)}", t) for a in ALIASES[allergen])


def check_allergen_declaration(
    ingredients: list[str], contains_statement: str | None
) -> Finding:
    if not ingredients:
        return Finding(
            claim="Allergen declaration",
            verdict=Verdict.REVIEW,
            rule_citation=CITATION,
            explanation="No ingredients were extracted, so allergens cannot be checked.",
            confidence=0.0,
        )

    contains_text = contains_statement or ""
    missing: dict[str, list[str]] = {}
    ambiguous: list[str] = []

    for ingredient in ingredients:
        parenthetical = " ".join(re.findall(r"\(([^)]*)\)", ingredient))
        found = _detect(ingredient)
        for allergen in found:
            declared = _is_named(allergen, contains_text) or _is_named(
                allergen, parenthetical
            )
            if not declared:
                missing.setdefault(allergen, []).append(ingredient)
        if not found and any(a in ingredient.lower() for a in AMBIGUOUS):
            ambiguous.append(ingredient)

    if missing:
        names = ", ".join(sorted(missing))
        return Finding(
            claim="Allergen declaration",
            verdict=Verdict.FAIL,
            rule_citation=CITATION,
            explanation=f"Ingredients contain {names}, but the allergen is not declared.",
            evidence={"missing": sorted(missing), "triggers": missing},
        )

    if ambiguous:
        return Finding(
            claim="Allergen declaration",
            verdict=Verdict.REVIEW,
            rule_citation=CITATION,
            explanation="An ingredient may contain an allergen, but its source is unclear.",
            evidence={"ambiguous": ambiguous},
            confidence=0.5,
        )

    return Finding(
        claim="Allergen declaration",
        verdict=Verdict.PASS,
        rule_citation=CITATION,
        explanation="All allergens found in the ingredients are declared.",
        evidence={"contains_statement": contains_text},
    )
