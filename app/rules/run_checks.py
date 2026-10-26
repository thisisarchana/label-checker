from app.rules.allergen_crosscheck import check_allergen_declaration
from app.rules.gluten_free import check_gluten_free_claim
from app.rules.label import LabelData
from app.rules.models import Finding, Verdict
from app.rules.protein_claim import check_protein_claim


def run_all_checks(label: LabelData) -> list[Finding]:
    findings: list[Finding] = []

    for claim in label.claims:
        lowered = claim.lower()
        if "protein" in lowered:
            findings.append(check_protein_claim(claim, label.protein_g))
        elif "gluten" in lowered:
            findings.append(
                check_gluten_free_claim(claim, label.ingredients, label.gluten_ppm)
            )
        elif "natural" in lowered:
            findings.append(Finding(
                claim=claim,
                verdict=Verdict.REVIEW,
                rule_citation="FDA informal policy on 'natural' (no formal regulation)",
                explanation=(
                    "FDA has no formal definition of 'natural'. Flagged as a "
                    "litigation and retailer risk, not a pass or fail."
                ),
                confidence=0.5,
            ))
        else:
            findings.append(Finding(
                claim=claim,
                verdict=Verdict.REVIEW,
                rule_citation="n/a",
                explanation="No automated rule exists for this claim yet.",
                confidence=0.0,
            ))

    findings.append(
        check_allergen_declaration(label.ingredients, label.allergen_statement)
    )
    return findings


def overall_verdict(findings: list[Finding]) -> Verdict:
    verdicts = {f.verdict for f in findings}
    if Verdict.FAIL in verdicts:
        return Verdict.FAIL
    if Verdict.REVIEW in verdicts:
        return Verdict.REVIEW
    return Verdict.PASS
