"""Splits 21 CFR Part 101 into paragraph-level chunks that each carry a precise citation."""
import json
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFR_DIR = ROOT / "data" / "raw" / "cfr"
OUT_PATH = ROOT / "data" / "processed" / "cfr_chunks.jsonl"

MAX_CHARS = 1500   # keeps each chunk within a small embedding model's input size
MAX_LEAD_CHARS = 400
SKIP_TAGS = {"HEAD", "CITA", "EFFDNOT", "EDNOTE", "img", "br"}
BLOCK_TAGS = {"P", "FP", "FP-1", "LI", "PSPACE"}
MARKER_RE = re.compile(r"^\(([A-Za-z]+|\d+)\)")
ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
         "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx"}


def text_of(element: ET.Element) -> str:
    return " ".join("".join(element.itertext()).split())


def table_text(table: ET.Element) -> str:
    lines = []
    caption = table.find("CAPTION")
    if caption is not None and text_of(caption):
        lines.append(text_of(caption))
    for row in table.iter("TR"):
        cells = [text_of(cell) for cell in row if cell.tag in ("TH", "TD")]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def blocks(element: ET.Element):
    """Yields the text blocks of a section in reading order: paragraphs and flattened tables."""
    for child in element:
        if child.tag in SKIP_TAGS:
            continue
        if child.tag == "TABLE":
            text = table_text(child)
        elif child.tag in BLOCK_TAGS:
            text = text_of(child)
        else:
            yield from blocks(child)   # containers such as EXTRACT or DIV
            continue
        if text:
            yield text


def level_of(marker: str, last_level1: str | None) -> int:
    """1 = (a), 2 = (1), 3 = (i), 4 = (A). A lone 'i', 'v' or 'x' is level 1 if it follows 'h', 'u' or 'w'."""
    if marker.isdigit():
        return 2
    if marker.isupper():
        return 4
    if marker in ROMAN:
        follows = (last_level1 is not None and len(marker) == 1 and len(last_level1) == 1
                   and ord(marker) == ord(last_level1) + 1)
        return 1 if follows else 3
    return 1


def label_blocks(texts: list[str]) -> list[tuple[str, int, str | None]]:
    labelled, last_level1 = [], None
    for text in texts:
        match = MARKER_RE.match(text)
        if match:
            marker = match.group(1)
            level = level_of(marker, last_level1)
            if level == 1:
                last_level1 = marker
        else:
            marker, level = None, 0
        labelled.append((text, level, marker))
    return labelled


def split_long(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Breaks one oversized block at line ends, then sentence ends, then by length."""
    if len(text) <= limit:
        return [text]
    pieces = []
    for line in text.split("\n"):
        if len(line) <= limit:
            pieces.append(line)
            continue
        for sentence in re.split(r"(?<=[.;]) ", line):
            while len(sentence) > limit:
                pieces.append(sentence[:limit])
                sentence = sentence[limit:]
            pieces.append(sentence)
    return pieces


def pack(texts: list[str], limit: int = MAX_CHARS) -> list[str]:
    parts, current = [], ""
    for text in texts:
        for piece in split_long(text, limit):
            if current and len(current) + 1 + len(piece) > limit:
                parts.append(current)
                current = piece
            else:
                current = f"{current}\n{piece}" if current else piece
    if current:
        parts.append(current)
    return parts


def build_chunks(number: str, labelled: list[tuple[str, int, str | None]]) -> list[tuple[str, str]]:
    """Returns (citation, text) pairs: one per lettered paragraph, split by number when too long."""
    groups: list[list] = []
    for item in labelled:
        if item[1] == 1 or not groups:
            groups.append([item[2] if item[1] == 1 else None, []])
        groups[-1][1].append(item)

    chunks = []
    for level1, items in groups:
        base = number + (f"({level1})" if level1 else "")
        body = "\n".join(text for text, _, _ in items)
        if len(body) <= MAX_CHARS:
            chunks.append((base, body))
            continue
        subs: list[list] = []
        for text, level, marker in items:
            if level == 2 or not subs:
                subs.append([marker if level == 2 else None, []])
            subs[-1][1].append(text)
        # A short lead-in such as "(b) The following apply:" is repeated in each numbered chunk,
        # so no chunk is just a heading.
        lead = ""
        if subs and subs[0][0] is None:
            lead_texts = subs.pop(0)[1]
            lead = "\n".join(lead_texts)
            if len(lead) > MAX_LEAD_CHARS:
                chunks.extend((base, part) for part in pack(lead_texts))
                lead = ""
        for level2, texts in subs:
            citation = base + (f"({level2})" if level2 else "")
            room = MAX_CHARS - (len(lead) + 1 if lead else 0)
            for part in pack(texts, room):
                chunks.append((citation, f"{lead}\n{part}" if lead else part))
    return chunks


def latest_file() -> Path:
    files = sorted(CFR_DIR.glob("title21_part101_*.xml"))
    if not files:
        raise SystemExit(f"No XML found in {CFR_DIR}. Run python -m rag.fetch_cfr first.")
    return files[-1]


def chunk_file(path: Path) -> list[dict]:
    date = re.search(r"(\d{4}-\d{2}-\d{2})", path.name).group(1)
    root = ET.parse(path).getroot()
    records = []
    for section in (el for el in root.iter() if el.get("TYPE") == "SECTION"):
        number = section.get("N")
        head = text_of(section.find("HEAD")) if section.find("HEAD") is not None else number
        heading = re.sub(r"^§\s*[\d.]+\s*", "", head)
        pieces = build_chunks(number, label_blocks(list(blocks(section))))
        for index, (citation, text) in enumerate(pieces, start=1):
            records.append({
                "id": f"{number}-{index}",
                "section": number,
                "heading": heading,
                "citation": f"21 CFR {citation}",
                "text": text,
                "url": f"https://www.ecfr.gov/current/title-21/section-{number}",
                "regulation_date": date,
            })
    return records


def main() -> None:
    path = latest_file()
    records = chunk_file(path)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    sizes = [len(r["text"]) for r in records]
    print(f"Source: {path.name}")
    print(f"Sections: {len({r['section'] for r in records})}   Chunks: {len(records)}")
    print(f"Chunk length in characters: min {min(sizes)}, median {int(statistics.median(sizes))}, "
          f"max {max(sizes)}; over the {MAX_CHARS} limit: {sum(s > MAX_CHARS for s in sizes)}")
    print(f"Wrote {OUT_PATH}")
    for section in ("101.54", "101.91"):
        print(f"\nChunks in {section}:")
        for r in (r for r in records if r["section"] == section):
            print(f"  {r['citation']:28s} {len(r['text']):5d}  {r['text'][:70]}")


if __name__ == "__main__":
    main()