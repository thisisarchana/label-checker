"""Builds a synthetic retrieval test set: Gemini writes one question per sampled regulation chunk."""
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from rag.textutil import content_words, longest_shared_run, words

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "processed" / "cfr_chunks.jsonl"
EVAL_DIR = ROOT / "data" / "rag_eval"
RAW = EVAL_DIR / "synthetic_raw.jsonl"       # cache: one line per generated question
FINAL = EVAL_DIR / "synthetic_questions.csv"

N_QUESTIONS = 120
MAX_PER_SECTION = 4
MIN_CHARS = 250
SEED = 21
DELAY_SECONDS = 7        # free-tier requests per minute are limited
MAX_COPIED_RUN = 6       # a question may not copy 6+ consecutive words from its source
MIN_OVERLAP = 0.35       # share of the question's content words that appear in the source
BANNED = ("section", "§", "paragraph", "excerpt", "passage", "regulation text", "cfr")

PROMPT = """You are writing test questions for a search system over US food labeling regulation (21 CFR Part 101).

Below is one excerpt. Write ONE question that a food-company label reviewer might ask and that this excerpt answers.

Rules:
- Paraphrase in everyday language. Do not copy any run of 4 or more words from the excerpt.
- Do not mention section numbers, paragraph letters, "the excerpt", or "the regulation".
- The question must be answerable from the excerpt alone, and specific enough that other parts of the regulation would not answer it equally well.
- Return JSON only: {{"question": "..."}}

Topic: {heading}

Excerpt:
{text}
"""


def load_chunks() -> list[dict]:
    with CHUNKS.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def table_heavy(text: str) -> bool:
    lines = text.split("\n")
    return sum(" | " in line for line in lines) > len(lines) / 2


def sample_chunks(chunks: list[dict]) -> list[dict]:
    """A spread across sections: up to MAX_PER_SECTION chunks each, no tables, no tiny chunks."""
    rng = random.Random(SEED)
    by_section: dict[str, list[dict]] = {}
    for chunk in chunks:
        if len(chunk["text"]) >= MIN_CHARS and not table_heavy(chunk["text"]):
            by_section.setdefault(chunk["section"], []).append(chunk)
    picked = []
    for section in sorted(by_section):
        pool = by_section[section]
        rng.shuffle(pool)
        picked.extend(pool[:MAX_PER_SECTION])
    rng.shuffle(picked)
    return picked[:N_QUESTIONS]


def ask(client, model: str, chunk: dict):
    """Returns a question, None to skip this chunk, or False when the daily quota is gone."""
    from google.genai import types

    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.4)
    prompt = PROMPT.format(heading=chunk["heading"], text=chunk["text"])
    for attempt in range(3):
        try:
            reply = client.models.generate_content(model=model, contents=prompt, config=config)
            data = json.loads(reply.text)
            if isinstance(data, list):
                data = data[0]
            return str(data["question"]).strip()
        except Exception as error:
            message = str(error)
            if "429" in message or "RESOURCE_EXHAUSTED" in message:
                if "perday" in message.lower().replace(" ", "").replace("_", ""):
                    print("Daily free-tier quota reached. Run this script again later; it resumes.")
                    return False
                print("Rate limit; waiting a minute...")
                time.sleep(65)
                continue
            print(f"  skipped {chunk['id']}: {message[:120]}")
            return None
    return None


def generate(client, model: str, chunks: list[dict]) -> bool:
    """Fills the cache. Returns False if it stopped early because of quota."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if RAW.exists():
        done = {json.loads(line)["chunk_id"] for line in RAW.read_text(encoding="utf-8").splitlines()}
    todo = [c for c in chunks if c["id"] not in done]
    print(f"{len(done)} questions already generated, {len(todo)} to go")
    with RAW.open("a", encoding="utf-8") as handle:
        for number, chunk in enumerate(todo, start=1):
            question = ask(client, model, chunk)
            if question is False:
                return False
            if question:
                handle.write(json.dumps({
                    "chunk_id": chunk["id"], "section": chunk["section"],
                    "citation": chunk["citation"], "question": question,
                }, ensure_ascii=False) + "\n")
                handle.flush()
            if number % 10 == 0:
                print(f"  {number}/{len(todo)}")
            time.sleep(DELAY_SECONDS)
    return True


def validate(rows: list[dict], chunk_text: dict[str, str]) -> tuple[list[dict], Counter]:
    kept, dropped, seen = [], Counter(), []
    for row in rows:
        question = row["question"]
        q_words = words(question)
        source = chunk_text[row["chunk_id"]]
        q_content = content_words(question)
        lowered = question.lower()
        if len(q_words) < 8:
            dropped["too short"] += 1
        elif any(term in lowered for term in BANNED):
            dropped["mentions the source"] += 1
        elif longest_shared_run(q_words, words(source)) >= MAX_COPIED_RUN:
            dropped["copies the source text"] += 1
        elif q_content and len(q_content & content_words(source)) / len(q_content) < MIN_OVERLAP:
            dropped["not grounded in the source"] += 1
        elif any(len(q_content & other) / max(1, len(q_content | other)) > 0.7 for other in seen):
            dropped["near-duplicate"] += 1
        else:
            seen.append(q_content)
            kept.append(row)
    return kept, dropped


def main() -> int:
    load_dotenv()
    key, model = os.environ.get("GEMINI_API_KEY"), os.environ.get("GEMINI_MODEL")
    if not key or not model:
        print("Set GEMINI_API_KEY and GEMINI_MODEL in .env first (see llm/check_gemini.py).")
        return 1
    if not CHUNKS.exists():
        print(f"{CHUNKS} not found. Run python -m rag.chunk_cfr first.")
        return 1

    from google import genai

    chunks = sample_chunks(load_chunks())
    finished = generate(genai.Client(api_key=key), model, chunks)

    rows = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines()]
    kept, dropped = validate(rows, {c["id"]: c["text"] for c in chunks})
    with FINAL.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "question", "expected_sections", "source_chunk", "source_citation"])
        for number, row in enumerate(kept, start=1):
            writer.writerow([f"s{number:03d}", row["question"], row["section"], row["chunk_id"], row["citation"]])

    print(f"\nGenerated {len(rows)} questions, kept {len(kept)}, dropped {sum(dropped.values())}: {dict(dropped)}")
    print(f"Sections covered: {len({r['section'] for r in kept})}   Wrote {FINAL}")
    if not finished or len(rows) < len(chunks) * 0.9:
        print("The set is incomplete. Run this script again later, and only freeze the set when it finishes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())