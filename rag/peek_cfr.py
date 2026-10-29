"""Lists every section in the downloaded Part 101 XML and shows how one section is structured."""
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

CFR_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "cfr"
KEY_SECTIONS = ["101.9", "101.13", "101.22", "101.54", "101.62", "101.65", "101.91"]
SHOW = "101.54"


def latest_file() -> Path:
    files = sorted(CFR_DIR.glob("title21_part101_*.xml"))
    if not files:
        raise SystemExit(f"No XML found in {CFR_DIR}. Run python -m rag.fetch_cfr first.")
    return files[-1]


def main() -> None:
    root = ET.parse(latest_file()).getroot()
    sections = [el for el in root.iter() if el.get("TYPE") == "SECTION"]
    print(f"Root element: {root.tag}   sections: {len(sections)}\n")

    tags = Counter(child.tag for section in sections for child in section.iter())
    print("Element types inside sections:", dict(tags.most_common()), "\n")

    print("SECTIONS (number, characters, heading):")
    numbers = set()
    for section in sections:
        number = section.get("N")
        numbers.add(number)
        head = section.find("HEAD")
        title = " ".join("".join(head.itertext()).split()) if head is not None else ""
        size = len("".join(section.itertext()))
        print(f"  {number:9s} {size:7d}  {title[:70]}")

    print("\nKey sections present:", {s: s in numbers for s in KEY_SECTIONS})

    target = next((s for s in sections if s.get("N") == SHOW), None)
    if target is not None:
        raw = ET.tostring(target, encoding="unicode")
        print(f"\nRaw XML of {SHOW} (first 2500 characters):\n{raw[:2500]}")


if __name__ == "__main__":
    main()