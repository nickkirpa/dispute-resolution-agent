"""Policy retrieval eval: does search find the clause that decides the case?

Queries are the golden complaints; relevant clauses are each case's expected type-specific clauses (general POL-GEN-*
clauses are always attached by the agent, so they are excluded). Searched against a policy version that includes the
v2 servicing clauses as distractors.

    uv run python eval/run_retrieval_eval.py                     # bm25 + hash embedder (offline)
    uv run python eval/run_retrieval_eval.py --embedder openai   # real embeddings (cached in data/embeddings_cache.json)

Two settings:
  filtered   - candidates restricted to the case's dispute type (what the agent does today; easy)
  unfiltered - all clauses compete (what you get if the type is wrong or a query spans topics; the real test)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
load_dotenv(ROOT / ".env")

from dispute_agent.config import Settings  # noqa: E402
from dispute_agent.tools import KnowledgeBase  # noqa: E402
from dispute_agent.tools.embeddings import CachedEmbedder, HashEmbedder, OpenAIEmbedder  # noqa: E402
from eval.run_eval import load_golden  # noqa: E402


def metrics(ranked: list[list[str]], relevant: list[set[str]], ks=(1, 3, 5)) -> dict:
    out = {}
    for k in ks:
        out[f"recall@{k}"] = sum(len(set(r[:k]) & rel) / len(rel) for r, rel in zip(ranked, relevant)) / len(ranked)
    rr = []
    for r, rel in zip(ranked, relevant):
        first = next((i for i, c in enumerate(r, 1) if c in rel), None)
        rr.append(1 / first if first else 0.0)
    out["mrr"] = sum(rr) / len(rr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, default=ROOT / "eval" / "golden_v1.jsonl")
    ap.add_argument("--kb-version", default="v2")
    ap.add_argument("--embedder", choices=["hash", "openai"], default="hash")
    args = ap.parse_args()

    inner = HashEmbedder() if args.embedder == "hash" else OpenAIEmbedder()
    embedder = CachedEmbedder(inner, ROOT / "data" / "embeddings_cache.json")
    kb = KnowledgeBase(Settings().kb_dir, args.kb_version, embedder=embedder)
    index_stats = dict(embedder.stats)

    cases = [g for g in load_golden(args.golden) if [c for c in g["expected_clauses"] if not c.startswith("POL-GEN-")]]
    queries = [g["narrative"] for g in cases]
    relevant = [{c for c in g["expected_clauses"] if not c.startswith("POL-GEN-")} for g in cases]
    types = [g["expected_type"] for g in cases]

    print(f"{len(cases)} queries | KB {kb.version}: {len(kb.clauses)} clauses | embedder {embedder.name} "
          f"(index: {index_stats['misses']} embedded, {index_stats['hits']} reused from cache)")
    print(f"{'setting':<11} {'mode':<7} {'recall@1':>9} {'recall@3':>9} {'recall@5':>9} {'MRR':>6}")
    results = {}
    for setting in ("filtered", "unfiltered"):
        for mode in ("bm25", "dense", "hybrid"):
            ranked = [[c.clause_id for c in kb.search(q, k=10, dispute_type=t if setting == "filtered" else None,
                                                       mode=mode, include_general=False)]
                      for q, t in zip(queries, types)]
            m = metrics(ranked, relevant)
            results[f"{setting}/{mode}"] = m
            print(f"{setting:<11} {mode:<7} {m['recall@1']:>9.3f} {m['recall@3']:>9.3f} {m['recall@5']:>9.3f} {m['mrr']:>6.3f}")
    out = ROOT / "eval" / "results" / f"retrieval-{kb.version}-{embedder.name.replace('/', '_')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"n_queries": len(cases), "kb_version": kb.version, "embedder": embedder.name, "results": results}, indent=2))
    print(f"\nReport: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
