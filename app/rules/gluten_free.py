import html
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

# Wording that looks like a trigger but means the opposite or something else.
NEGATION_RE = re.compile(r"\b(?:wheat|barley|rye|gluten)[\s-]*free\b")
NEUTRAL_RE = re.compile(r"\brice\s+malt\b")

# Wheat-derived ingredients: treated as containing gluten unless the label
# states gluten-free or a test result is under the limit.
WHEAT_DEFAULT_PATTERNS = [
    re.compile(r"\bwheat starch\b"),
    re.compile(r"\bwheat\s+(?:dextrins?|maltodextrins?)\b"),
]
# In the US, plain "flour" means wheat flour; bread products are wheat-based.
BARE_FLOUR_RE = re.compile(
    r"^(?:(?:organic|unbleached|bleached|enriched|unenriched|non-gmo|white|"
    r"all[\s-]*purpose|bread|cake|pastry)\s+)*flour\s*(?:[(\[].*)?$"
)
BREAD_RE = re.compile(r"\b(?:bread\s*crumbs?|panko|croutons?)\b")
SOY_SAUCE_RE = re.compile(r"\bsoy sauce\b")

# Ingredients that may or may not carry gluten: a human should look.
REVIEW_PATTERNS = [
    re.compile(r"\bbrewer'?s yeast\b"),   # often grown on barley
    re.compile(r"\boats?\b"),             # frequent cross-contact with wheat
    # wheat/barley grass or leaf: usually gluten-free, but seed contamination is a known risk
    re.compile(r"\b(?:wheat|barley)\s*grass\b|\b(?:wheat|barley)\b(?:\s*\([^)]*\))?\s*leaf\b"),
]

DECLARED_RE = re.compile(r"gluten[\s-]*free")


def _normalize(claim_text: str) -> str:
    c = " ".join(claim_text.lower().split())
    c = re.sub(r"gluten[\s-]+free", "gluten-free", c)
    return c.strip(" .!")


def _clean(ingredient: str) -> str:
    text = html.unescape(ingredient).lower()
    text = re.sub(r"[*\u2020\u2021\u00ae\u2122]", "", text)
    return " ".join(text.split())


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

    # "Stated otherwise": the ingredient text says gluten-free, or a test is under the limit.
    tested_clean = gluten_ppm is not None and gluten_ppm < PPM_LIMIT
    stated_otherwise = tested_clean or any(
        DECLARED_RE.search(_clean(i)) for i in ingredients
    )

    grains: list[str] = []
    unsupported_items: list[str] = []
    review_items: list[str] = []
    declared_items: list[str] = []

    for ingredient in ingredients:
        text = NEGATION_RE.sub(" ", _clean(ingredient))
        text = NEUTRAL_RE.sub(" ", text)

        wheat_default = False
        for pattern in WHEAT_DEFAULT_PATTERNS:
            if pattern.search(text):
                wheat_default = True
                text = pattern.sub(" ", text)

        uncertain = False
        for pattern in REVIEW_PATTERNS:
            if pattern.search(text):
                uncertain = True
                text = pattern.sub(" ", text)

        if any(re.search(rf"\b{re.escape(t)}\b", text) for t in FAIL_TRIGGERS):
            grains.append(ingredient)
            continue

        if BARE_FLOUR_RE.search(text) or BREAD_RE.search(text):
            wheat_default = True
        elif (
            SOY_SAUCE_RE.search(text)
            and "(" not in ingredient
            and "[" not in ingredient
            and "tamari" not in text
        ):
            wheat_default = True

        if wheat_default or uncertain:
            if stated_otherwise:
                declared_items.append(ingredient)
            elif wheat_default:
                unsupported_items.append(ingredient)
            else:
                review_items.append(ingredient)

    evidence = {
        "gluten_grains": grains,
        "unsupported_items": unsupported_items,
        "review_items": review_items,
        "declared_items": declared_items,
        "gluten_ppm": gluten_ppm,
    }

    over_limit = gluten_ppm is not None and gluten_ppm >= PPM_LIMIT
    if grains or unsupported_items or over_limit:
        reasons = []
        if grains:
            reasons.append(f"contains gluten-source ingredients: {', '.join(grains)}")
        if unsupported_items:
            reasons.append(
                "wheat-derived ingredients with no gluten-free statement or test: "
                + ", ".join(unsupported_items)
            )
        if over_limit:
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

    note = ""
    if declared_items:
        note = (
            " Accepted on the strength of a gluten-free statement or test for: "
            + ", ".join(declared_items) + "."
        )

    if gluten_ppm is None:
        return Finding(
            claim=claim_text,
            verdict=Verdict.PASS,
            rule_citation=CITATION,
            explanation=(
                "No gluten-source ingredients found. This is an ingredient screen only; "
                "no test result confirms the under-20-ppm limit." + note
            ),
            evidence=evidence,
            confidence=0.8,
        )

    return Finding(
        claim=claim_text,
        verdict=Verdict.PASS,
        rule_citation=CITATION,
        explanation=f"No gluten-source ingredients, and testing shows {gluten_ppm} ppm." + note,
        evidence=evidence,
    )