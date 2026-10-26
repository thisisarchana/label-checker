from app.rules.models import Verdict
from app.rules.protein_claim import check_protein_claim


def test_high_protein_passes_at_exactly_20_percent():
    assert check_protein_claim("High Protein", 10).verdict == Verdict.PASS


def test_high_protein_fails_with_low_protein():
    f = check_protein_claim("High Protein", 3)
    assert f.verdict == Verdict.FAIL
    assert f.evidence["percent_dv"] == 6.0


def test_good_source_passes_at_12_percent():
    assert check_protein_claim("good source of protein", 6).verdict == Verdict.PASS


def test_good_source_fails_at_8_percent():
    assert check_protein_claim("Good source of protein", 4).verdict == Verdict.FAIL


def test_missing_protein_goes_to_review():
    assert check_protein_claim("High Protein", None).verdict == Verdict.REVIEW


def test_unknown_wording_goes_to_review():
    assert check_protein_claim("Protein packed", 12).verdict == Verdict.REVIEW