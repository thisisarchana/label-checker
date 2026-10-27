from app.rules.gluten_free import check_gluten_free_claim
from app.rules.models import Verdict


def test_wheat_flour_fails():
    f = check_gluten_free_claim("Gluten-Free", ["wheat flour", "sugar"])
    assert f.verdict == Verdict.FAIL
    assert "wheat flour" in f.evidence["gluten_grains"]


def test_clean_ingredients_pass_with_reduced_confidence():
    f = check_gluten_free_claim("Gluten Free", ["rice flour", "sugar", "salt"])
    assert f.verdict == Verdict.PASS
    assert f.confidence == 0.8


def test_barley_malt_fails():
    f = check_gluten_free_claim("gluten-free", ["rice", "barley malt"])
    assert f.verdict == Verdict.FAIL


def test_buckwheat_is_not_wheat():
    f = check_gluten_free_claim("Gluten-Free", ["buckwheat flour", "salt"])
    assert f.verdict == Verdict.PASS


def test_oats_go_to_review():
    f = check_gluten_free_claim("Gluten-Free", ["rolled oats", "honey"])
    assert f.verdict == Verdict.REVIEW


def test_undeclared_wheat_starch_fails():
    f = check_gluten_free_claim("Gluten-Free", ["wheat starch", "sugar"])
    assert f.verdict == Verdict.FAIL
    assert f.evidence["unsupported_items"] == ["wheat starch"]


def test_high_ppm_fails_even_with_clean_ingredients():
    f = check_gluten_free_claim("Gluten-Free", ["rice flour"], gluten_ppm=25)
    assert f.verdict == Verdict.FAIL


def test_low_ppm_passes_with_full_confidence():
    f = check_gluten_free_claim("Gluten-Free", ["rice flour"], gluten_ppm=10)
    assert f.verdict == Verdict.PASS
    assert f.confidence == 1.0


def test_no_gluten_wording_is_recognized():
    f = check_gluten_free_claim("No Gluten", ["rice flour"])
    assert f.verdict == Verdict.PASS


def test_unrecognized_wording_goes_to_review():
    f = check_gluten_free_claim("Gluten friendly", ["rice flour"])
    assert f.verdict == Verdict.REVIEW


def test_empty_ingredients_go_to_review():
    f = check_gluten_free_claim("Gluten-Free", [])
    assert f.verdict == Verdict.REVIEW