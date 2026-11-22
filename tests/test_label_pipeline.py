from app.extraction.fields import extract_fields
from app.extraction.read_quality import CONF_CUTOFF, MIN_CHARS, assess_photo, assess_typed
from app.rules.declaration import check_declared_allergens
from app.rules.detection import detect_item
from app.rules.engine import check_label, overall_verdict
from app.rules.models import Verdict


# ---- extraction -------------------------------------------------------------------------------

def test_extract_splits_ingredients_statement_and_drops_cross_contact_warning():
    ingredients, statement = extract_fields(
        "Ingredients: wheat flour, sugar. Contains: Wheat. May contain milk."
    )
    assert ingredients == ["wheat flour", "sugar"]
    assert statement == "Contains: Wheat"


def test_extract_keeps_ingredients_after_a_processing_note():
    ingredients, _ = extract_fields(
        "Ingredients: rice, chicken (may contain up to 2% retained water), wheat flour, salt."
    )
    assert ingredients == ["rice", "chicken (may contain up to 2% retained water)", "wheat flour", "salt"]


def test_extract_without_a_statement():
    ingredients, statement = extract_fields("sugar, cocoa butter, salt")
    assert ingredients == ["sugar", "cocoa butter", "salt"] and statement is None


# ---- detection --------------------------------------------------------------------------------

def test_oil_only_mentions_follow_the_policy():
    assert "soybeans" not in detect_item("soybean oil")
    assert "soybeans" in detect_item("soy lecithin")
    assert "peanuts" not in detect_item("peanut oil")
    assert "peanuts" in detect_item("peanut oil, peanut flour")


# ---- allergen declaration -----------------------------------------------------------------------

def test_a_derivative_without_a_declaration_fails():
    f = check_declared_allergens(["sugar", "whey protein"], None)
    assert f.verdict == Verdict.FAIL
    assert "milk" in f.evidence["missing"]


def test_a_contains_statement_declares_the_allergen():
    assert check_declared_allergens(["sugar", "whey protein"], "Contains: Milk").verdict == Verdict.PASS


def test_a_parenthetical_names_the_source():
    assert check_declared_allergens(["sugar", "whey (milk)"], None).verdict == Verdict.PASS


def test_an_ingredient_that_names_its_own_source_needs_no_extra_declaration():
    f = check_declared_allergens(["wheat flour", "egg", "soy lecithin", "almonds"], None)
    assert f.verdict == Verdict.PASS


def test_refined_oil_alone_is_not_flagged():
    assert check_declared_allergens(["water", "soybean oil", "salt"], None).verdict == Verdict.PASS


def test_plain_lecithin_goes_to_review_and_empty_list_goes_to_review():
    assert check_declared_allergens(["sugar", "lecithin"], None).verdict == Verdict.REVIEW
    assert check_declared_allergens([], None).verdict == Verdict.REVIEW


def test_cocoa_butter_is_not_dairy():
    assert check_declared_allergens(["cocoa mass", "cocoa butter", "sugar"], None).verdict == Verdict.PASS


# ---- engine -----------------------------------------------------------------------------------

def test_gluten_claim_with_wheat_fails_while_declaration_passes():
    findings = check_label(["Gluten-Free"], ["wheat flour", "sugar"], None)
    assert [f.verdict for f in findings] == [Verdict.FAIL, Verdict.PASS]
    assert overall_verdict(findings) == Verdict.FAIL


def test_protein_claim_is_checked_against_the_grams():
    findings = check_label(["High Protein"], ["oats", "whey protein"], "Contains: Milk", protein_g=3)
    assert findings[0].verdict == Verdict.FAIL and findings[1].verdict == Verdict.PASS


def test_natural_and_unknown_claims_go_to_review():
    findings = check_label(["All Natural", "Heart healthy"], ["oats"], None)
    assert findings[0].verdict == Verdict.REVIEW and findings[1].verdict == Verdict.REVIEW
    assert overall_verdict(findings) == Verdict.REVIEW


def test_a_clean_label_passes():
    findings = check_label([], ["oats", "honey"], None)
    assert overall_verdict(findings) == Verdict.PASS


# ---- read quality -------------------------------------------------------------------------------

def test_photo_read_gate():
    assert assess_photo(CONF_CUTOFF, MIN_CHARS).passed
    low = assess_photo(0.85, 300)
    assert not low.passed and "confidence" in low.reasons[0]
    short = assess_photo(0.98, 14)
    assert not short.passed and "characters" in short.reasons[0]


def test_typed_text_always_passes_the_read_gate():
    assert assess_typed("sugar, salt").passed