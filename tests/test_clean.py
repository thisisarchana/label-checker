from app.extraction.clean import split_advisory, strip_advisory
from app.rules.allergen_crosscheck import _detect

G083 = (
    "Cooked pasta ([semolina, niacin, ferrous sulfate (iron), thiamin mononitrate, riboflavin, "
    "folic acid], water), white meat chicken ([chicken breasts with rib meat may contain up to 2% "
    "retained water], canola oil, back pepper), caesar, dressing (soybean oil, water, egg yolk, "
    "lemon juice concentrate, salt, parmesan cheese [part skim milk, cheese culture, salt, enzymes], "
    "natural flavors, anchovies dehydrated), parmesan cheese (pasteurized part, skimmed cow a milk), "
    "red pepper green onion, parsley, salt, black pepper."
)


def test_may_contain_sentence_is_removed():
    kept, removed = split_advisory("Sugar, cocoa. May contain traces of milk and soy. Contains: wheat")
    assert "milk" not in kept
    assert "Contains: wheat" in kept
    assert "May contain traces of milk and soy." in removed


def test_may_contain_up_to_percent_is_not_a_warning():
    kept, removed = split_advisory("chicken (may contain up to 2% retained water), egg yolk, anchovies")
    assert removed == ""
    assert "egg yolk" in kept


def test_real_caesar_salad_text_keeps_its_allergens():
    found = _detect(strip_advisory(G083))
    assert {"milk", "egg", "fish", "wheat"} <= found


def test_warning_inside_brackets_stops_at_the_closing_bracket():
    kept, removed = split_advisory("flour (may contain traces of nuts), eggs, butter")
    assert "eggs" in kept and "butter" in kept
    assert "nuts" not in kept


def test_facility_statement_is_removed_but_later_sentence_survives():
    kept, _ = split_advisory("Rice, salt. Made in a facility that processes peanuts. Contains: soy.")
    assert "peanuts" not in kept
    assert "Contains: soy." in kept


def test_missing_period_does_not_delete_the_rest_of_a_long_list():
    tail = ", ".join(f"ingredient{i}" for i in range(60)) + ", egg yolk"
    kept, removed = split_advisory("sugar, may contain traces of milk " + tail)
    assert "egg yolk" in kept
    assert len(removed) <= 270