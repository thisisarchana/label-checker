from app.rules.fuzzy_allergens import fuzzy_detect, tolerance


def test_exact_matches_still_work():
    assert fuzzy_detect("sugar, whey, almonds") == {"milk", "tree nuts"}


def test_one_wrong_letter_in_a_long_word_is_tolerated():
    assert "peanuts" in fuzzy_detect("roasted peanuf butter")
    assert "tree nuts" in fuzzy_detect("alm0nds, sugar")


def test_two_errors_in_a_very_long_word_are_tolerated():
    assert "fish" in fuzzy_detect("anch0v1es, water")


def test_short_words_must_match_exactly():
    assert tolerance("milk") == 0
    assert "milk" not in fuzzy_detect("m1lk chocolate")
    assert "milk" not in fuzzy_detect("silk, sugar")


def test_neutral_phrases_are_still_ignored():
    assert "milk" not in fuzzy_detect("cocoa butter, sugar")


def test_unrelated_text_matches_nothing():
    assert fuzzy_detect("water, salt, citric acid") == set()