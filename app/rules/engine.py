"""Runs every automated check for one label and combines the verdicts."""
from app.rules.declaration import check_declared_allergens
from app.rules.gluten_free import check_gluten_free_claim
from app.rules.models import Finding, Verdict
from app.rules.protein_claim import check_protein_claim


def check_label(
    claims: list[str],
    ingredients: list[str],
    contains_statement: str | None,
    protein_g: float | None = None,
    gluten_ppm: float | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    for claim in claims:
        lowered = claim.lower()
        if "protein" in lowered:
            findings.append(check_protein_claim(claim, protein_g))
        elif "gluten" in lowered:
            findings.append(check_gluten_free_claim(claim, ingredients, gluten_ppm))
        elif "natural" in lowered:
            findings.append(Finding(
                claim=claim, verdict=Verdict.REVIEW,
                rule_citation="FDA informal policy on 'natural' (no formal regulation)",
                explanation="FDA has no formal definition of 'natural'. Flagged as a litigation and "
                            "retailer risk, not a pass or fail.",
                confidence=0.5,
            ))
        else:
            findings.append(Finding(
                claim=claim, verdict=Verdict.REVIEW, rule_citation="n/a",
                explanation="No automated rule exists for this claim yet.", confidence=0.0,
            ))
    findings.append(check_declared_allergens(ingredients, contains_statement))
    return findings


def overall_verdict(findings: list[Finding]) -> Verdict:
    verdicts = {f.verdict for f in findings}
    if Verdict.FAIL in verdicts:
        return Verdict.FAIL
    if Verdict.REVIEW in verdicts:
        return Verdict.REVIEW
    return Verdict.PASS