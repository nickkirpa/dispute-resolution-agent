"""Text embeddings for dense policy retrieval, with a content-addressed cache.

The cache key is sha256(model + text), so publishing a new policy version re-embeds only clauses whose text is new
or changed (incremental re-indexing); unchanged clauses are reused across versions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class OpenAIEmbedder:
    """OpenAI embeddings, directly or via a LiteLLM proxy (OPENAI_BASE_URL / OPENAI_API_KEY)."""

    def __init__(self, model: str | None = None, client=None, batch: int = 64):
        import openai

        self.name = model or os.getenv("DISPUTE_AGENT_EMBED_MODEL", "openai/text-embedding-3-small")
        self.client = client or openai.OpenAI()
        self.batch = batch

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            resp = self.client.embeddings.create(model=self.name, input=texts[i : i + self.batch])
            out += [_normalize(d.embedding) for d in sorted(resp.data, key=lambda d: d.index)]
        return out


class HashEmbedder:
    """Deterministic offline embedder (hashed bag of words). For tests, not for quality."""

    name = "hash-256"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            v = [0.0] * 256
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1.0
            vecs.append(_normalize(v))
        return vecs


class CachedEmbedder:
    """Wraps an embedder with a JSON cache on disk. Thread-safe; `stats` reports hits/misses for the last calls."""

    def __init__(self, inner: Embedder, path: Path):
        self.inner, self.path, self.name = inner, path, inner.name
        self._lock = threading.Lock()
        self._cache: dict[str, list[float]] = json.loads(path.read_text()) if path.exists() else {}
        self.stats = {"hits": 0, "misses": 0}

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.inner.name}\n{text}".encode()).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        with self._lock:
            todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self._cache]
        if todo:
            vecs = self.inner.embed([t for _, t in todo])
            with self._lock:
                self._cache.update({k: v for (k, _), v in zip(todo, vecs)})
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self._cache))
        self.stats["misses"] += len(todo)
        self.stats["hits"] += len(keys) - len(todo)
        return [self._cache[k] for k in keys]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vectors are L2-normalised
