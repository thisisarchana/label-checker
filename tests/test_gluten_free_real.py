"""Cases taken from the real-data audit of Open Food Facts."""
from app.rules.gluten_free import check_gluten_free_claim
from app.rules.models import Verdict


def check(ingredients):
    return check_gluten_free_claim("Gluten-Free", ingredients)


def test_declared_gluten_free_wheat_starch_with_asterisk_passes():
    f = check(["Certified gluten-free wheat* starch", "water", "egg"])
    assert f.verdict == Verdict.PASS
    assert f.evidence["declared_items"]


def test_undeclared_wheat_starch_with_asterisk_fails():
    f = check(["wheat* starch", "water"])
    assert f.verdict == Verdict.FAIL


def test_declared_gluten_free_oats_pass():
    f = check(["certified gluten free whole grain rolled oats", "maple syrup"])
    assert f.verdict == Verdict.PASS
    assert f.confidence == 0.8


def test_plain_oats_still_go_to_review():
    assert check(["whole grain oats", "sugar"]).verdict == Verdict.REVIEW


def test_wheat_grass_goes_to_review_not_fail():
    f = check(["organic wheat grass juice powder (leaf)", "spirulina"])
    assert f.verdict == Verdict.REVIEW


def test_barley_leaf_with_latin_name_goes_to_review():
    f = check(["Superfood Complex (spirulina, organic barley (Hordeum vulgare) leaf powder)"])
    assert f.verdict == Verdict.REVIEW


def test_wheat_dextrin_fails_unless_stated_otherwise():
    assert check(["Wheat dextrins", "cellulose"]).verdict == Verdict.FAIL
    assert check(["Wheat dextrins", "cellulose", "gluten free"]).verdict == Verdict.PASS


def test_wheat_dextrin_passes_with_a_clean_lab_result():
    f = check_gluten_free_claim("Gluten-Free", ["wheat dextrin", "cellulose"], gluten_ppm=5)
    assert f.verdict == Verdict.PASS
    assert f.confidence == 1.0


def test_powdered_wheat_fails_even_when_gluten_free_is_stated():
    f = check(["rice flour", "powdered wheat", "gluten free"])
    assert f.verdict == Verdict.FAIL


def test_wheat_free_wording_is_not_a_trigger():
    f = check(["wheat free tamari soy sauce (water, soybeans, salt)", "rice"])
    assert f.verdict == Verdict.PASS


def test_rice_malt_syrup_is_not_malt():
    assert check(["rice malt syrup", "rice"]).verdict == Verdict.PASS


def test_bare_soy_sauce_fails_unless_stated_otherwise():
    assert check(["soy sauce", "sugar"]).verdict == Verdict.FAIL
    assert check(["gluten free soy sauce", "sugar"]).verdict == Verdict.PASS


def test_soy_sauce_with_sublist_passes():
    f = check(["soy sauce (water, soybeans, rice, salt)", "sugar"])
    assert f.verdict == Verdict.PASS


def test_bare_flour_fails():
    assert check(["enriched flour", "sugar"]).verdict == Verdict.FAIL
    assert check(["flour", "sugar"]).verdict == Verdict.FAIL


def test_specialty_flours_pass():
    f = check(["mustard flour", "chia flour", "konjac flour", "lupini flour"])
    assert f.verdict == Verdict.PASS


def test_bare_breadcrumbs_fail_but_declared_ones_pass():
    assert check(["breadcrumbs", "salt"]).verdict == Verdict.FAIL
    f = check(["Gluten Free Bread Crumb (Rice Flour, Salt, Water)", "salt"])
    assert f.verdict == Verdict.PASS

def test_gluten_free_flour_blend_with_sublist_passes():
    f = check(["Flour blend (sorghum flour, tapioca flour, rice flour)", "sugar"])
    assert f.verdict == Verdict.PASS

def test_enriched_flour_with_vitamin_sublist_fails():
    f = check(["enriched flour (niacin, iron, thiamin mononitrate)", "sugar"])
    assert f.verdict == Verdict.FAIL