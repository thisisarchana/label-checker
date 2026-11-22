"""Scores BM25, dense, and hybrid retrieval on the frozen question sets."""
import csv
import sys
from pathlib import Path

from eval.metrics import wilson
from rag.retrieval import Retriever, load_chunks

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "rag_eval"
KS = (1, 3, 5, 10)
METHODS = ("bm25", "dense", "hybrid")
MAX_LISTED = 8


def read_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def first_rank(chunks: list[dict], ranking: list[int], expected: set[str]) -> int | None:
    """1-based position of the first retrieved chunk from an expected section."""
    for position, index in enumerate(ranking, start=1):
        if chunks[index]["section"] in expected:
            return position
    return None


def hit_rate(ranks: list[int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks) / len(ranks)


def mrr(ranks: list[int | None], cutoff: int = 10) -> float:
    return sum(1 / r for r in ranks if r is not None and r <= cutoff) / len(ranks)


def evaluate(name: str, questions: list[dict], retriever: Retriever) -> None:
    chunks = retriever.chunks
    ranks = {m: [] for m in METHODS}
    exact = {m: [] for m in METHODS}   # synthetic set only: was the very chunk the question came from found?
    missed = []
    for q in questions:
        expected = set(q["expected_sections"].split("|"))
        rankings = {m: getattr(retriever, f"{m}_ranking")(q["question"], 10) for m in METHODS}
        for m in METHODS:
            ranks[m].append(first_rank(chunks, rankings[m], expected))
            if q.get("source_chunk"):
                ids = [chunks[i]["id"] for i in rankings[m]]
                exact[m].append(ids.index(q["source_chunk"]) + 1 if q["source_chunk"] in ids else None)
        if ranks["hybrid"][-1] is None or ranks["hybrid"][-1] > 5:
            top = chunks[rankings["hybrid"][0]]["citation"]
            missed.append(f"  {q['id']}  expected {q['expected_sections']:10s} got {top:22s} {q['question'][:70]}")

    print(f"\n{name}  (n={len(questions)})   hit = a chunk from an expected section is in the top k")
    print(f"{'method':8s}" + "".join(f"{f'hit@{k}':>9s}" for k in KS) + f"{'MRR@10':>9s}   hit@5 95% CI")
    for m in METHODS:
        low, high = wilson(sum(r is not None and r <= 5 for r in ranks[m]), len(questions))
        print(f"{m:8s}" + "".join(f"{hit_rate(ranks[m], k):9.2f}" for k in KS)
              + f"{mrr(ranks[m]):9.2f}   [{low:.2f}, {high:.2f}]")
    if exact["bm25"]:
        print("\nStricter: the exact source chunk is in the top k")
        for m in METHODS:
            print(f"{m:8s}" + "".join(f"{hit_rate(exact[m], k):9.2f}" for k in KS))
    print(f"\nQuestions where the hybrid missed the top 5: {len(missed)}")
    for line in missed[:MAX_LISTED]:
        print(line)


def main() -> int:
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks. Building the index (the first run downloads the embedding model).")
    retriever = Retriever(chunks)
    for name, file in (("HAND-WRITTEN QUESTIONS", "hand_questions.csv"),
                       ("SYNTHETIC QUESTIONS (Gemini)", "synthetic_questions.csv")):
        evaluate(name, read_questions(EVAL_DIR / file), retriever)
    return 0


if __name__ == "__main__":
    sys.exit(main())