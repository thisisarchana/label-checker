from app.rules.models import Finding, Verdict

CITATION = "21 CFR 101.54"
PROTEIN_DV_G = 50.0
HIGH_THRESHOLD = 20.0
GOOD_THRESHOLD = 10.0

HIGH_TERMS = {"high protein", "rich in protein", "excellent source of protein"}
GOOD_TERMS = {"good source of protein", "contains protein", "provides protein"}


def percent_dv(protein_g: float) -> float:
    return protein_g * 100 / PROTEIN_DV_G


def check_protein_claim(claim_text: str, protein_g: float | None) -> Finding:
    claim = " ".join(claim_text.lower().split())

    if protein_g is None or protein_g < 0:
        return Finding(
            claim=claim_text,
            verdict=Verdict.REVIEW,
            rule_citation=CITATION,
            explanation="Protein amount missing or invalid, so the claim cannot be checked.",
            evidence={"protein_g": protein_g},
            confidence=0.0,
        )

    if claim in HIGH_TERMS:
        threshold, level = HIGH_THRESHOLD, "high"
    elif claim in GOOD_TERMS:
        threshold, level = GOOD_THRESHOLD, "good source"
    else:
        return Finding(
            claim=claim_text,
            verdict=Verdict.REVIEW,
            rule_citation=CITATION,
            explanation="Unrecognized protein claim wording.",
            evidence={"protein_g": protein_g},
            confidence=0.0,
        )

    pct = percent_dv(protein_g)
    passed = pct >= threshold
    return Finding(
        claim=claim_text,
        verdict=Verdict.PASS if passed else Verdict.FAIL,
        rule_citation=CITATION,
        explanation=(
            f"A '{level}' claim needs at least {threshold:.0f}% of the Daily Value. "
            f"{protein_g}g of protein is {pct:.1f}% DV."
        ),
        evidence={"protein_g": protein_g, "percent_dv": round(pct, 1), "threshold": threshold},
    )
