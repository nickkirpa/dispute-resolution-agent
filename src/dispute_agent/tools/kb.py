"""Policy knowledge base: versioned markdown clauses + BM25 retrieval.

Clause format inside kb/policies/<version>/*.md:

    ## POL-DUP-01: Duplicate charge refunds
    Applies to: duplicate_charge
    <clause text...>
"""

from __future__ import annotations

import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from ..state import PolicyClause

CLAUSE_RE = re.compile(r"^## (?P<id>POL-[A-Z]+-\d+):\s*(?P<title>.+)$", re.M)
TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class KnowledgeBase:
    def __init__(self, kb_dir: Path, version: str = "latest"):
        versions = sorted(p.name for p in kb_dir.iterdir() if p.is_dir() and p.name.startswith("v"))
        if not versions:
            raise FileNotFoundError(f"No policy versions in {kb_dir}")
        self.version = versions[-1] if version == "latest" else version
        self.clauses = self._load(kb_dir / self.version)
        self._by_id = {c.clause_id: c for c in self.clauses}
        self._bm25 = BM25Okapi([_tokens(f"{c.title} {c.text} {' '.join(c.applies_to)}") for c in self.clauses])

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
                clauses.append(
                    PolicyClause(clause_id=m["id"], version=self.version, title=m["title"].strip(), text=body.strip(), applies_to=applies)
                )
        return clauses

    def get(self, clause_id: str) -> PolicyClause | None:
        return self._by_id.get(clause_id)

    def search(self, query: str, k: int = 4, dispute_type: str | None = None) -> list[PolicyClause]:
        """BM25 over clauses; clauses that apply to `dispute_type` (or to `all`) are kept, others filtered out."""
        scores = self._bm25.get_scores(_tokens(query))
        ranked = sorted(zip(self.clauses, scores), key=lambda x: -x[1])
        out = []
        for clause, score in ranked:
            if dispute_type and not ({dispute_type, "all"} & set(clause.applies_to)):
                continue
            out.append(clause.model_copy(update={"score": float(score)}))
            if len(out) == k:
                break
        return out
