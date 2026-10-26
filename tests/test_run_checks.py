from app.rules.label import LabelData
from app.rules.models import Verdict
from app.rules.run_checks import overall_verdict, run_all_checks


def test_clean_label_passes():
    label = LabelData(
        claims=["High Protein"],
        ingredients=["oats", "honey", "whey protein"],
        allergen_statement="Contains: Milk",
        protein_g=12,
    )
    findings = run_all_checks(label)
    assert overall_verdict(findings) == Verdict.PASS
    assert len(findings) == 2


def test_false_protein_claim_fails_overall():
    label = LabelData(
        claims=["High Protein"],
        ingredients=["oats", "honey"],
        protein_g=3,
    )
    findings = run_all_checks(label)
    assert overall_verdict(findings) == Verdict.FAIL
    assert len(findings) == 2


def test_natural_claim_is_flagged_for_review():
    label = LabelData(claims=["All Natural"], ingredients=["oats", "honey"])
    findings = run_all_checks(label)
    assert "no formal definition" in findings[0].explanation
    assert overall_verdict(findings) == Verdict.REVIEW


def test_allergen_violation_found_without_any_claims():
    label = LabelData(claims=[], ingredients=["wheat flour", "sugar"])
    findings = run_all_checks(label)
    assert len(findings) == 1
    assert overall_verdict(findings) == Verdict.FAIL


def test_gluten_claim_fails_even_when_wheat_is_declared():
    label = LabelData(
        claims=["Gluten-Free"],
        ingredients=["wheat flour", "whey"],
        allergen_statement="Contains: Wheat",
    )
    findings = run_all_checks(label)
    assert len(findings) == 2
    assert findings[0].verdict == Verdict.FAIL   # gluten claim
    assert findings[1].verdict == Verdict.FAIL   # milk not declared
