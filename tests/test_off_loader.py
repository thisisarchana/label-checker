import math

from app.extraction.off_loader import (
    has_gluten_free_claim,
    parse_label_text,
    split_ingredients,
    to_label_data,
)


def test_split_respects_parentheses():
    result = split_ingredients("sugar, chocolate (cocoa mass, milk), salt")
    assert result == ["sugar", "chocolate (cocoa mass, milk)", "salt"]


def test_two_percent_or_less_is_not_a_contains_statement():
    ingredients, statement = parse_label_text(
        "Flour, sugar, contains 2% or less of salt, yeast"
    )
    assert statement is None
    assert "salt" in " ".join(ingredients)


def test_may_contain_advisory_is_dropped():
    ingredients, statement = parse_label_text(
        "Hazelnuts, sugar. May contain traces of milk and soy."
    )
    assert ingredients == ["Hazelnuts", "sugar"]
    assert statement is None


def test_contains_statement_is_captured():
    ingredients, statement = parse_label_text(
        "Wheat flour, sugar. Contains: Wheat, Milk. May contain nuts."
    )
    assert ingredients == ["Wheat flour", "sugar"]
    assert statement == "Contains: Wheat, Milk"


def test_gluten_free_tag_detection():
    assert has_gluten_free_claim(["en:no-gluten", "en:vegan"])
    assert has_gluten_free_claim(["en:gluten-free"])
    assert not has_gluten_free_claim(["en:vegan"])
    assert not has_gluten_free_claim([])


def test_to_label_data_handles_nan_protein():
    record = {
        "ingredients_text": "rice, salt",
        "labels_tags": ["en:gluten-free"],
        "protein_serving_g": math.nan,
    }
    label = to_label_data(record)
    assert label.protein_g is None
    assert label.claims == ["Gluten-Free"]