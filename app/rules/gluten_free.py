import re

from app.rules.models import Finding, Verdict

CITATION = "21 CFR 101.91"
PPM_LIMIT = 20.0

GLUTEN_FREE_TERMS = {"gluten-free", "no gluten", "free of gluten", "without gluten"}

# Gluten-containing grains and grain-derived ingredients.
FAIL_TRIGGERS = [
    "wheat", "rye", "barley", "triticale", "spelt", "kamut", "khorasan",
    "farro", "semolina", "durum", "farina", "einkorn", "emmer", "bulgur",
    "couscous", "seitan", "graham", "malt", "malted",
]

# Ingredients that may or may not be a problem; a human should look.
REVIEW_PATTERNS = [
    r"\bwheat starch\b",       # allowed only if processed to under 20 ppm
    r"\bbrewer'?s yeast\b",    # often grown on barley
    r"\boats?\b",              # frequent cross-contact with wheat
]


def _normalize(claim_text: str) -> str:
    c = " ".join(claim_text.lower().split())
    c = re.sub(r"gluten[\s-]+free", "gluten-free", c)
    return c.strip(" .!")


def check_gluten_free_claim(
    claim_text: str,
    ingredients: list[str],
    gluten_ppm: float | None = None,
) -> Finding:
    claim = _normalize(claim_text)

    if claim not in GLUTEN_FREE_TERMS:
        return Finding(
            claim=claim_text,
            verdict=Verdict.REVIEW,
            rule_citation=CITATION,
            explanation="Unrecognized gluten claim wording.",
            confidence=0.0,
        )

    if not ingredients:
        return Finding(
            claim=claim_text,
            verdict=Verdict.REVIEW,
            rule_citation=CITATION,
            explanation="No ingredients were extracted, so the claim cannot be checked.",
            confidence=0.0,
        )

    grains: list[str] = []
    review_items: list[str] = []

    for ingredient in ingredients:
        text = ingredient.lower()
        for pattern in REVIEW_PATTERNS:
            if re.search(pattern, text):
                review_items.append(ingredient)
                text = re.sub(pattern, " ", text)
        for trigger in FAIL_TRIGGERS:
            if re.search(rf"\b{re.escape(trigger)}\b", text):
                grains.append(ingredient)
                break

    evidence = {
        "gluten_grains": grains,
        "review_items": review_items,
        "gluten_ppm": gluten_ppm,
    }

    if grains or (gluten_ppm is not None and gluten_ppm >= PPM_LIMIT):
        reasons = []
        if grains:
            reasons.append(f"contains gluten-source ingredients: {', '.join(grains)}")
        if gluten_ppm is not None and gluten_ppm >= PPM_LIMIT:
            reasons.append(f"tested at {gluten_ppm} ppm (limit is under {PPM_LIMIT:.0f})")
        return Finding(
            claim=claim_text,
            verdict=Verdict.FAIL,
            rule_citation=CITATION,
            explanation="A gluten-free claim is not supported: " + "; ".join(reasons) + ".",
            evidence=evidence,
        )

    if review_items:
        return Finding(
            claim=claim_text,
            verdict=Verdict.REVIEW,
            rule_citation=CITATION,
            explanation=(
                "Ingredients may carry gluten depending on sourcing or processing: "
                + ", ".join(review_items) + "."
            ),
            evidence=evidence,
            confidence=0.5,
        )

    if gluten_ppm is None:
        return Finding(
            claim=claim_text,
            verdict=Verdict.PASS,
            rule_citation=CITATION,
            explanation=(
                "No gluten-source ingredients found. This is an ingredient screen only; "
                "no test result confirms the under-20-ppm limit."
            ),
            evidence=evidence,
            confidence=0.8,
        )

    return Finding(
        claim=claim_text,
        verdict=Verdict.PASS,
        rule_citation=CITATION,
        explanation=f"No gluten-source ingredients, and testing shows {gluten_ppm} ppm.",
        evidence=evidence,
    )
