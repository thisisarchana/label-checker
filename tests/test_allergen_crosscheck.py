from app.rules.allergen_crosscheck import check_allergen_declaration
from app.rules.models import Verdict


def test_whey_without_milk_declared_fails():
    f = check_allergen_declaration(["sugar", "whey protein"], "Contains: Soy")
    assert f.verdict == Verdict.FAIL
    assert "milk" in f.evidence["missing"]


def test_whey_with_milk_declared_passes():
    f = check_allergen_declaration(["sugar", "whey protein"], "Contains: Milk")
    assert f.verdict == Verdict.PASS


def test_parenthetical_declaration_passes():
    f = check_allergen_declaration(["sugar", "whey (milk)"], None)
    assert f.verdict == Verdict.PASS


def test_no_allergens_passes():
    f = check_allergen_declaration(["sugar", "cocoa butter", "salt"], None)
    assert f.verdict == Verdict.PASS


def test_soy_lecithin_undeclared_fails():
    f = check_allergen_declaration(["sugar", "soy lecithin"], None)
    assert f.verdict == Verdict.FAIL
    assert "soybeans" in f.evidence["missing"]


def test_plain_lecithin_goes_to_review():
    f = check_allergen_declaration(["sugar", "lecithin"], None)
    assert f.verdict == Verdict.REVIEW


def test_multiple_missing_allergens():
    f = check_allergen_declaration(["enriched wheat flour", "egg", "salt"], "Contains: Milk")
    assert f.verdict == Verdict.FAIL
    assert f.evidence["missing"] == ["egg", "wheat"]


def test_named_tree_nut_counts_as_declared():
    f = check_allergen_declaration(["almonds", "sugar"], "Contains: Almonds")
    assert f.verdict == Verdict.PASS


def test_empty_ingredients_goes_to_review():
    f = check_allergen_declaration([], "Contains: Milk")
    assert f.verdict == Verdict.REVIEW
