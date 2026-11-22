import hashlib

import numpy as np

from rag.evaluate_retrieval import first_rank, hit_rate, mrr
from rag.retrieval import BM25, Retriever, rrf_fuse, tokenize


class FakeEmbedder:
    """Hashes words into a small vector so the dense path can be tested without a model download."""
    name = "fake"

    def _encode(self, text: str) -> np.ndarray:
        vector = np.zeros(64)
        for word in tokenize(text):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def encode_documents(self, texts):
        return np.array([self._encode(t) for t in texts])

    def encode_query(self, query):
        return self._encode(query)


CHUNKS = [
    {"id": "a", "section": "101.91", "heading": "Gluten-free labeling", "citation": "21 CFR 101.91",
     "text": "A food bearing a gluten-free claim must contain less than 20 parts per million gluten."},
    {"id": "b", "section": "101.61", "heading": "Sodium claims", "citation": "21 CFR 101.61",
     "text": "A low sodium claim requires 140 milligrams or less of sodium per serving."},
    {"id": "c", "section": "101.62", "heading": "Fat claims", "citation": "21 CFR 101.62",
     "text": "A low fat claim requires 3 grams or less of fat per serving."},
]


def test_bm25_ranks_the_document_with_the_rare_term_first():
    docs = [tokenize(c["text"]) for c in CHUNKS]
    scores = BM25(docs).scores(tokenize("gluten"))
    assert int(np.argmax(scores)) == 0


def test_tokenize_stems_word_forms_and_drops_stop_words():
    assert tokenize("claims") == tokenize("claim")
    assert tokenize("labeled labeling") == tokenize("label label")
    assert tokenize("the claim") == tokenize("claim")


def test_rrf_rewards_agreement_between_rankings():
    assert rrf_fuse([[1, 2, 3], [3, 1, 2]])[0] == 1


def test_retriever_finds_the_right_chunk_with_every_method():
    retriever = Retriever(CHUNKS, embedder=FakeEmbedder(), cache_dir=None)
    for method in ("bm25", "dense", "hybrid"):
        assert retriever.search("how much sodium for a low sodium claim", k=1, method=method)[0]["id"] == "b"


def test_metrics():
    chunks = CHUNKS
    assert first_rank(chunks, [2, 0, 1], {"101.91"}) == 2
    assert first_rank(chunks, [1, 2], {"101.91"}) is None
    ranks = [1, 2, None, 6]
    assert hit_rate(ranks, 1) == 0.25 and hit_rate(ranks, 5) == 0.5
    assert abs(mrr(ranks) - (1 + 0.5 + 1 / 6) / 4) < 1e-9