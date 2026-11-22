"""Hybrid retrieval over the regulation chunks: BM25 keyword search, dense embeddings, and fusion."""
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import snowballstemmer

from rag.textutil import STOP, words

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "cfr_chunks.jsonl"
CACHE_DIR = ROOT / "data" / "processed"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # bge models expect this on queries
RRF_K = 60
POOL = 100   # how deep each ranking goes before fusion

_stemmer = snowballstemmer.stemmer("english")


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def tokenize(text: str) -> list[str]:
    return _stemmer.stemWords([w for w in words(text) if w not in STOP])


def chunk_text(chunk: dict) -> str:
    return f"{chunk['heading']}\n{chunk['text']}"


class BM25:
    """Okapi BM25 with an inverted index."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.size = len(documents)
        self.lengths = np.array([len(d) for d in documents], dtype=float)
        self.average = float(self.lengths.mean()) or 1.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, document in enumerate(documents):
            for word, count in Counter(document).items():
                self.postings[word].append((index, count))
        self.idf = {
            word: math.log(1 + (self.size - len(posts) + 0.5) / (len(posts) + 0.5))
            for word, posts in self.postings.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        result = np.zeros(self.size)
        for word in set(query_tokens):
            for index, count in self.postings.get(word, ()):
                norm = count + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average)
                result[index] += self.idf[word] * count * (self.k1 + 1) / norm
        return result


class HFEmbedder:
    """Sentence embeddings from a Hugging Face model, run locally with PyTorch."""

    def __init__(self, model_name: str = MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal rank fusion: documents ranked high in several lists rise to the top."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            scores[index] += 1.0 / (k + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))


class Retriever:
    def __init__(self, chunks: list[dict], embedder=None, cache_dir: Path | None = CACHE_DIR):
        self.chunks = chunks
        self.bm25 = BM25([tokenize(chunk_text(c)) for c in chunks])
        self.embedder = embedder or HFEmbedder()
        self.vectors = self._vectors(cache_dir)

    def _vectors(self, cache_dir: Path | None) -> np.ndarray:
        texts = [chunk_text(c) for c in self.chunks]
        fingerprint = hashlib.sha1(("\n".join(texts) + self.embedder.name).encode("utf-8")).hexdigest()[:12]
        path = None
        if cache_dir is not None:
            slug = re.sub(r"[^a-z0-9]+", "-", self.embedder.name.lower())
            path = cache_dir / f"cfr_vectors_{slug}_{fingerprint}.npy"
            if path.exists():
                return np.load(path)
        vectors = self.embedder.encode_documents(texts)
        if path is not None:
            np.save(path, vectors)
        return vectors

    def bm25_ranking(self, query: str, top: int = POOL) -> list[int]:
        return np.argsort(-self.bm25.scores(tokenize(query)), kind="stable")[:top].tolist()

    def dense_ranking(self, query: str, top: int = POOL) -> list[int]:
        return np.argsort(-(self.vectors @ self.embedder.encode_query(query)), kind="stable")[:top].tolist()

    def hybrid_ranking(self, query: str, top: int = POOL) -> list[int]:
        return rrf_fuse([self.bm25_ranking(query, top), self.dense_ranking(query, top)])[:top]

    def search(self, query: str, k: int = 5, method: str = "hybrid") -> list[dict]:
        ranking = {"bm25": self.bm25_ranking, "dense": self.dense_ranking, "hybrid": self.hybrid_ranking}[method](query)
        return [self.chunks[i] for i in ranking[:k]]