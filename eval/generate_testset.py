"""Builds a labeled test set by injecting known violations into clean labels."""
import copy
import json
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "testset" / "cases.json"

# Synthetic clean products. Every one should pass all checks.
CLEAN_PRODUCTS = [
    {
        "id": "protein-bar",
        "name": "Almond protein bar",
        "label": {
            "claims": ["High Protein"],
            "ingredients": ["whey protein isolate", "almonds", "dates", "cocoa butter", "salt"],
            "allergen_statement": "Contains: Milk, Almonds",
            "protein_g": 15.0,
            "gluten_ppm": None,
        },
    },
    {
        "id": "granola",
        "name": "Peanut granola",
        "label": {
            "claims": ["Good Source of Protein"],
            "ingredients": ["rolled oats", "honey", "canola oil", "sea salt", "peanuts"],
            "allergen_statement": "Contains: Peanuts",
            "protein_g": 7.0,
            "gluten_ppm": None,
        },
    },
    {
        "id": "gf-crackers",
        "name": "Sesame rice crackers",
        "label": {
            "claims": ["Gluten-Free"],
            "ingredients": ["rice flour", "tapioca starch", "sunflower oil", "sea salt", "sesame seeds"],
            "allergen_statement": "Contains: Sesame",
            "protein_g": None,
            "gluten_ppm": 8.0,
        },
    },
    {
        "id": "dark-chocolate",
        "name": "Dark chocolate bar",
        "label": {
            "claims": [],
            "ingredients": ["cocoa mass", "sugar", "cocoa butter", "vanilla"],
            "allergen_statement": None,
            "protein_g": None,
            "gluten_ppm": None,
        },
    },
    {
        "id": "yogurt",
        "name": "Plain yogurt",
        "label": {
            "claims": ["Good Source of Protein"],
            "ingredients": ["cultured pasteurized milk", "live active cultures"],
            "allergen_statement": "Contains: Milk",
            "protein_g": 6.0,
            "gluten_ppm": None,
        },
    },
    {
        "id": "pasta",
        "name": "Egg pasta",
        "label": {
            "claims": ["High Protein"],
            "ingredients": ["durum wheat semolina", "egg", "salt"],
            "allergen_statement": "Contains: Wheat, Egg",
            "protein_g": 12.0,
            "gluten_ppm": None,
        },
    },
    {
        "id": "trail-mix",
        "name": "Nut trail mix",
        "label": {
            "claims": [],
            "ingredients": ["peanuts", "raisins", "sunflower seeds", "cashews"],
            "allergen_statement": "Contains: Peanuts, Cashews",
            "protein_g": None,
            "gluten_ppm": None,
        },
    },
    {
        "id": "pancake-mix",
        "name": "Gluten-free pancake mix",
        "label": {
            "claims": ["Gluten-Free", "Good Source of Protein"],
            "ingredients": ["brown rice flour", "buttermilk powder", "egg white powder", "baking soda"],
            "allergen_statement": "Contains: Milk, Egg",
            "protein_g": 6.0,
            "gluten_ppm": None,
        },
    },
    {
        "id": "tofu-snack",
        "name": "Baked tofu snack",
        "label": {
            "claims": ["High Protein"],
            "ingredients": ["organic soybeans", "water", "calcium sulfate"],
            "allergen_statement": "Contains: Soy",
            "protein_g": 12.0,
            "gluten_ppm": None,
        },
    },
    {
        "id": "fruit-snack",
        "name": "Apple fruit snack",
        "label": {
            "claims": [],
            "ingredients": ["apple puree", "sugar", "pectin", "citric acid"],
            "allergen_statement": None,
            "protein_g": None,
            "gluten_ppm": None,
        },
    },
    {
        "id": "tuna-pouch",
        "name": "Tuna pouch",
        "label": {
            "claims": ["High Protein"],
            "ingredients": ["tuna", "water", "salt"],
            "allergen_statement": "Contains: Fish",
            "protein_g": 17.0,
            "gluten_ppm": None,
        },
    },
]


def _has_claim(label: dict, word: str) -> bool:
    return any(word in c.lower() for c in label["claims"])


def _drop_first_allergen(statement: str | None) -> str | None:
    if not statement:
        return None
    body = statement.split(":", 1)[-1]
    items = [i.strip() for i in body.split(",") if i.strip()]
    rest = items[1:]
    return "Contains: " + ", ".join(rest) if rest else None


# Each mutation returns a modified copy of the label, or None if it doesn't apply.

def remove_allergen_declaration(label: dict) -> dict | None:
    if not label["allergen_statement"]:
        return None
    new = copy.deepcopy(label)
    new["allergen_statement"] = _drop_first_allergen(label["allergen_statement"])
    return new


def inflate_protein_claim(label: dict) -> dict | None:
    if label["protein_g"] is None or not _has_claim(label, "protein"):
        return None
    new = copy.deepcopy(label)
    new["protein_g"] = 3.0
    return new


def add_false_high_protein(label: dict) -> dict | None:
    if _has_claim(label, "protein"):
        return None
    new = copy.deepcopy(label)
    new["claims"].append("High Protein")
    new["protein_g"] = 2.0
    return new


def add_wheat_to_gluten_free(label: dict) -> dict | None:
    if not _has_claim(label, "gluten"):
        return None
    new = copy.deepcopy(label)
    new["ingredients"].append("wheat flour")
    statement = new["allergen_statement"]
    if statement is None:
        new["allergen_statement"] = "Contains: Wheat"
    elif "wheat" not in statement.lower():
        new["allergen_statement"] = statement + ", Wheat"
    return new


def add_high_gluten_ppm(label: dict) -> dict | None:
    if not _has_claim(label, "gluten"):
        return None
    new = copy.deepcopy(label)
    new["gluten_ppm"] = 35.0
    return new


HIDDEN_ALLERGENS = [
    ("soy lecithin", "soy"),
    ("whey powder", "milk"),
    ("egg white", "egg"),
    ("sesame oil", "sesame"),
]


def add_hidden_allergen(label: dict) -> dict | None:
    statement = (label["allergen_statement"] or "").lower()
    for ingredient, keyword in HIDDEN_ALLERGENS:
        if keyword not in statement:
            new = copy.deepcopy(label)
            new["ingredients"].append(ingredient)
            return new
    return None


def add_natural_claim(label: dict) -> dict | None:
    new = copy.deepcopy(label)
    new["claims"].append("All Natural")
    return new


# (violation type, function, expected overall verdict)
MUTATIONS = [
    ("allergen_not_declared", remove_allergen_declaration, "fail"),
    ("protein_claim_unsupported", inflate_protein_claim, "fail"),
    ("false_high_protein_added", add_false_high_protein, "fail"),
    ("gluten_grain_in_gf_product", add_wheat_to_gluten_free, "fail"),
    ("gluten_ppm_too_high", add_high_gluten_ppm, "fail"),
    ("hidden_allergen_ingredient", add_hidden_allergen, "fail"),
    ("natural_claim_no_rule", add_natural_claim, "review"),
]


def build_cases() -> list[dict]:
    cases = []
    for product in CLEAN_PRODUCTS:
        cases.append({
            "id": f"{product['id']}--clean",
            "product": product["name"],
            "label": product["label"],
            "expected_overall": "pass",
            "violation_type": "none",
        })
        for violation_type, mutate, expected in MUTATIONS:
            mutated = mutate(product["label"])
            if mutated is None:
                continue
            cases.append({
                "id": f"{product['id']}--{violation_type}",
                "product": product["name"],
                "label": mutated,
                "expected_overall": expected,
                "violation_type": violation_type,
            })
    return cases


def main() -> None:
    cases = build_cases()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(cases, indent=2))
    print(f"Wrote {len(cases)} cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()