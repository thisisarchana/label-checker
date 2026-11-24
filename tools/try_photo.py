"""Runs one label photo (or typed text) through the whole rules pipeline and prints what it finds.

Examples:
    python -m tools.try_photo data/images/gold/g025.jpg "Gluten-Free"
    python -m tools.try_photo "Ingredients: oats, whey. Contains: Milk." "High Protein" protein=3
"""
import sys

from app.extraction.fields import extract_fields
from app.extraction.ocr import read_photo
from app.extraction.read_quality import assess_photo, assess_typed
from app.rules.engine import check_label, overall_verdict

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def shorten(text: str, limit: int = 70) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    source, extras = argv[1], argv[2:]
    protein = next((float(a.split("=")[1]) for a in extras if a.startswith("protein=")), None)
    claims = [a for a in extras if not a.startswith("protein=")]

    if source.lower().endswith(IMAGE_SUFFIXES):
        text, confidence = read_photo(source)
        quality = assess_photo(confidence, len(text))
    else:
        text = source
        quality = assess_typed(text)
    ingredients, statement = extract_fields(text)

    status = "PASSED" if quality.passed else "FAILED: " + "; ".join(quality.reasons)
    print(f"Read quality: {status}")
    print(f"Statement:    {statement}")
    print(f"Ingredients ({len(ingredients)}):")
    for item in ingredients[:12]:
        print("   -", shorten(item))
    if len(ingredients) > 12:
        print(f"   ... and {len(ingredients) - 12} more")

    findings = check_label(claims, ingredients, statement, protein_g=protein)
    print("\nFindings:")
    for f in findings:
        print(f"   [{f.verdict.value.upper():6s}] {f.claim}: {shorten(f.explanation, 110)}")
    print(f"\nOverall by the rules alone: {overall_verdict(findings).value.upper()}")
    if not quality.passed:
        print("The read gate failed, so a PASS above cannot be trusted; the agent will route this to review.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))