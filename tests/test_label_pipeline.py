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
    
# ---- real OCR text (photo g025, a pesto pasta salad) and badly bracketed lists --------------------

G025_OCR = (
    "PESTO PASTA SALAD #14110 RON THIAMIN MONONITRATE. RIBOFLAVIN. FOLIC ACIDI. INGREDTENTS: "
    "FULLY COOKED TROTTOLE PASTA (WATER. NRICHED WHEAT CURUM SEMOLINA (SEMOLINA, NIACIN. SEA SALT. "
    "ENZYMES), PINE NUTS, EXTRA VIRGIN OLIVE CHLESE (PASTEURIZED PART-SKIM MILK, CHEESE CULTURES. "
    "SASIL PESTO IGENOVESE BASIL. SUNFLOWER CIL. PARMESAN OIL, PECORINO ROMANO CHEESE (SHEEP'S MILK. "
    "SALT. RENNET). GARLIC. SEA SALT. ASCORBIC ACIC, CITRIC ACID. BLACK PEPPER1, MARINATED SLOW ROASTED "
    "CHERRY TOMATOES (OVEN ROASTED CHERRY TOMATOES, CANOLA OIL, WHITE BALSAMIC VINEGAR, GARLIC. SEA SAL1, "
    "OREGANO). FRESH MOZZARELLA CHEESE PEARLS (PASTEURIZED MILK, VINEGAR, ENZYMES. SALT). BABY ARUGULA "
    "CONTAINS: EGG, MILK, TREE NUTS (PINE NUTS), WHEAT"
)


def test_real_ocr_text_is_anchored_at_the_misspelled_ingredients_heading():
    ingredients, statement = extract_fields(G025_OCR)
    assert statement == "CONTAINS: EGG, MILK, TREE NUTS (PINE NUTS), WHEAT"
    assert ingredients[0].startswith("FULLY COOKED TROTTOLE PASTA")
    assert "PESTO PASTA SALAD" not in ingredients[0]


def test_real_ocr_text_with_its_contains_statement_passes_the_declaration_check():
    ingredients, statement = extract_fields(G025_OCR)
    assert check_declared_allergens(ingredients, statement).verdict == Verdict.PASS


def test_a_contains_statement_before_the_ingredients_is_still_found():
    ingredients, statement = extract_fields("Contains: Milk. Ingredients: sugar, whey protein, salt")
    assert statement == "Contains: Milk" and ingredients == ["sugar", "whey protein", "salt"]


def test_an_unclosed_bracket_cannot_excuse_an_undeclared_derivative():
    text = "Ingredients: pasta (water, whey powder, salt, sauce (pasteurized milk, cream"
    ingredients, statement = extract_fields(text)
    assert len(ingredients) == 1   # the unclosed bracket swallows the rest of the list
    finding = check_declared_allergens(ingredients, statement)
    assert finding.verdict == Verdict.FAIL and "milk" in finding.evidence["missing"]


def test_sub_ingredients_inherit_the_source_named_by_their_parent():
    finding = check_declared_allergens(["enriched wheat durum semolina (semolina, niacin, iron)"], None)
    assert finding.verdict == Verdict.PASS


def test_a_cheese_with_milk_in_its_own_bracket_is_named():
    assert check_declared_allergens(["parmesan cheese (pasteurized milk, salt, enzymes)"], None).verdict == Verdict.PASS