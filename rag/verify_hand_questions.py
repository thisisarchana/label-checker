"""Sanity-checks the hand-written questions: is the expected section a plausible answer to each?"""
import csv
import json
import math
from pathlib import Path

from rag.textutil import content_words

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "processed" / "cfr_chunks.jsonl"
HAND = ROOT / "data" / "rag_eval" / "hand_questions.csv"
FLAG_RANK = 3   # flag a question if its expected section is not in the top 3 by weighted word overlap


def main() -> None:
    texts: dict[str, str] = {}
    for line in CHUNKS.read_text(encoding="utf-8").splitlines():
        chunk = json.loads(line)
        texts[chunk["section"]] = texts.get(chunk["section"], "") + " " + chunk["heading"] + " " + chunk["text"]
    vocab = {section: content_words(text) for section, text in texts.items()}
    doc_freq: dict[str, int] = {}
    for words in vocab.values():
        for word in words:
            doc_freq[word] = doc_freq.get(word, 0) + 1

    def weight(word: str) -> float:
        return math.log(len(vocab) / doc_freq[word]) + 0.1 if word in doc_freq else 0.0

    flagged = 0
    with HAND.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            q_words = content_words(row["question"])
            total = sum(weight(w) for w in q_words) or 1.0
            scores = {s: sum(weight(w) for w in q_words if w in v) / total for s, v in vocab.items()}
            ranking = sorted(scores, key=scores.get, reverse=True)
            expected = row["expected_sections"].split("|")
            best_rank = min((ranking.index(s) + 1 for s in expected if s in ranking), default=None)
            missing = [s for s in expected if s not in vocab]
            problem = missing or best_rank is None or best_rank > FLAG_RANK
            flagged += bool(problem)
            mark = "CHECK" if problem else "ok   "
            print(f"{mark} {row['id']}  expected {row['expected_sections']:12s} rank {best_rank}  "
                  f"top3 {ranking[:3]}" + (f"  UNKNOWN SECTION {missing}" if missing else ""))
    print(f"\n{flagged} question(s) flagged. Read those against the regulation text; the check is only lexical.")


if __name__ == "__main__":
    main()