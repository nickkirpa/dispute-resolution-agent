from .kb import KnowledgeBase
from .ledger import Ledger
from .refund import compute_refund

__all__ = ["KnowledgeBase", "Ledger", "build_kb", "compute_refund"]


def build_kb(settings, version: str | None = None) -> KnowledgeBase:
    """KB for the configured retrieval mode; dense/hybrid get the OpenAI embedder behind the on-disk cache."""
    from ..config import ROOT

    embedder = None
    if settings.retrieval_mode != "bm25":
        from .embeddings import CachedEmbedder, OpenAIEmbedder

        embedder = CachedEmbedder(OpenAIEmbedder(), ROOT / "data" / "embeddings_cache.json")
    return KnowledgeBase(settings.kb_dir, version or settings.kb_version, embedder=embedder, mode=settings.retrieval_mode)
