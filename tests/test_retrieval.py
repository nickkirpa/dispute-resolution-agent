"""Dense / hybrid retrieval and incremental re-indexing, offline (HashEmbedder)."""

import pytest

from dispute_agent.config import Settings
from dispute_agent.tools import KnowledgeBase
from dispute_agent.tools.embeddings import CachedEmbedder, HashEmbedder


def test_modes_return_type_filtered_clauses():
    kb = KnowledgeBase(Settings().kb_dir, "v2", embedder=HashEmbedder())
    for mode in ("bm25", "dense", "hybrid"):
        hits = kb.search("charged twice for the same purchase, duplicate payment", k=3, dispute_type="duplicate_charge",
                         mode=mode, include_general=False)
        assert hits and all("duplicate_charge" in h.applies_to for h in hits), mode


def test_unfiltered_search_spans_all_topics_but_can_exclude_general():
    kb = KnowledgeBase(Settings().kb_dir, "v2")
    ids = [c.clause_id for c in kb.search("ATM did not give me cash", k=37, include_general=False)]
    assert "POL-ATM-01" in ids[:3] and not any(i.startswith("POL-GEN-") for i in ids)


def test_dense_mode_requires_embedder():
    with pytest.raises(ValueError):
        KnowledgeBase(Settings().kb_dir, "v2", mode="hybrid")


def test_publishing_v2_reembeds_only_new_or_changed_clauses(tmp_path):
    cache = CachedEmbedder(HashEmbedder(), tmp_path / "emb.json")
    KnowledgeBase(Settings().kb_dir, "v1", embedder=cache)
    assert cache.stats == {"hits": 0, "misses": 13}  # first index: everything is new
    cache.stats = {"hits": 0, "misses": 0}
    KnowledgeBase(Settings().kb_dir, "v2", embedder=cache)
    # v2 = 13 v1 clauses (2 with changed text: GEN-02, GEN-04) + 24 new servicing clauses
    assert cache.stats == {"hits": 11, "misses": 26}
