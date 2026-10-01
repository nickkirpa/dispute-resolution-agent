"""Policy knowledge base: versioned markdown clauses + BM25 retrieval.

Clause format inside kb/policies/<version>/*.md:

    ## POL-DUP-01: Duplicate charge refunds
    Applies to: duplicate_charge
    Parameters: duplicate_window_days=3        (optional; machine-readable values the guards enforce)
    <clause text...>

Parameters make the policy document the single source of truth: publishing a new version with a different value
changes the agent's behaviour without a code change.
"""

from __future__ import annotations

import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from ..state import PolicyClause
from .embeddings import Embedder, cosine

RRF_K = 60  # standard Reciprocal Rank Fusion constant: score = sum over rankings of 1 / (RRF_K + rank)

CLAUSE_RE = re.compile(r"^## (?P<id>POL-[A-Z]+-\d+):\s*(?P<title>.+)$", re.M)
TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class KnowledgeBase:
    """Versioned policy clauses with keyword (BM25), dense (embedding) and hybrid (RRF) retrieval.

    `mode` is the default retrieval mode for search(); dense and hybrid need an embedder. Clause vectors are computed
    once per KB load, and a CachedEmbedder makes that incremental across policy versions.
    """

    def __init__(self, kb_dir: Path, version: str = "latest", embedder: Embedder | None = None, mode: str = "bm25"):
        if mode not in ("bm25", "dense", "hybrid"):
            raise ValueError(f"unknown retrieval mode {mode!r}")
        if mode != "bm25" and embedder is None:
            raise ValueError(f"retrieval mode {mode!r} needs an embedder")
        versions = sorted(p.name for p in kb_dir.iterdir() if p.is_dir() and p.name.startswith("v"))
        if not versions:
            raise FileNotFoundError(f"No policy versions in {kb_dir}")
        self.version = versions[-1] if version == "latest" else version
        self.params: dict[str, float] = {}
        self.param_source: dict[str, str] = {}  # parameter -> clause id that defines it
        self.clauses = self._load(kb_dir / self.version)
        self._by_id = {c.clause_id: c for c in self.clauses}
        self._bm25 = BM25Okapi([_tokens(f"{c.title} {c.text} {' '.join(c.applies_to)}") for c in self.clauses])
        self.mode, self.embedder = mode, embedder
        self._vectors = embedder.embed([f"{c.title}. {c.text}" for c in self.clauses]) if embedder else None

    def _load(self, folder: Path) -> list[PolicyClause]:
        clauses = []
        for md in sorted(folder.glob("*.md")):
            text = md.read_text()
            matches = list(CLAUSE_RE.finditer(text))
            for i, m in enumerate(matches):
                body = text[m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()
                applies: list[str] = []
                if body.lower().startswith("applies to:"):
                    first, _, body = body.partition("\n")
                    applies = [a.strip() for a in first.split(":", 1)[1].split(",") if a.strip()]
                    body = body.strip()
                if body.lower().startswith("parameters:"):
                    first, _, body = body.partition("\n")
                    for kv in first.split(":", 1)[1].split(","):
                        key, _, value = kv.strip().partition("=")
                        if key in self.params:
                            raise ValueError(f"parameter {key} defined twice ({self.param_source[key]}, {m['id']})")
                        self.params[key], self.param_source[key] = float(value), m["id"]
                clauses.append(
                    PolicyClause(clause_id=m["id"], version=self.version, title=m["title"].strip(), text=body.strip(), applies_to=applies)
                )
        return clauses

    def param(self, name: str) -> float:
        """A policy parameter of this version. Missing parameters are an error: guards must never run on a guess."""
        if name not in self.params:
            raise KeyError(f"policy {self.version} defines no parameter {name!r}")
        return self.params[name]

    def get(self, clause_id: str) -> PolicyClause | None:
        return self._by_id.get(clause_id)

    def search(self, query: str, k: int = 4, dispute_type: str | None = None, mode: str | None = None,
               include_general: bool = True) -> list[PolicyClause]:
        """Top-k clauses for `query`.

        dispute_type: keep only clauses for that type (plus general `all` clauses unless include_general=False).
        mode: bm25 | dense | hybrid (default: the KB's mode). `score` on the returned clauses is the mode's own score.
        """
        mode = mode or self.mode
        candidates = [i for i, c in enumerate(self.clauses)
                      if dispute_type is None or dispute_type in c.applies_to or (include_general and "all" in c.applies_to)]
        if not include_general and dispute_type is None:
            candidates = [i for i in candidates if "all" not in self.clauses[i].applies_to]

        def bm25_rank() -> list[tuple[int, float]]:
            scores = self._bm25.get_scores(_tokens(query))
            return sorted(((i, float(scores[i])) for i in candidates), key=lambda x: -x[1])

        def dense_rank() -> list[tuple[int, float]]:
            if self._vectors is None:
                raise ValueError("dense retrieval needs an embedder")
            q = self.embedder.embed([query])[0]
            return sorted(((i, cosine(q, self._vectors[i])) for i in candidates), key=lambda x: -x[1])

        if mode == "bm25":
            ranked = bm25_rank()
        elif mode == "dense":
            ranked = dense_rank()
        else:
            fused: dict[int, float] = {}
            for ranking in (bm25_rank(), dense_rank()):
                for rank, (i, _) in enumerate(ranking, 1):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
            ranked = sorted(fused.items(), key=lambda x: -x[1])
        return [self.clauses[i].model_copy(update={"score": score}) for i, score in ranked[:k]]
