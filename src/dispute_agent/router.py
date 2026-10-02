"""Fine-tuned transformer router (scripts/train_router.py) used as a cheap first classifier.

The graph asks the router first; if its confidence is at or above the threshold its answer is used and the LLM
classification call is skipped, otherwise the LLM classifies. torch/transformers are optional dependencies
(`uv sync --extra router`), imported only when a router is configured.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from .state import DisputeType

# All GPU work (model loading, tensor transfer, inference) runs on ONE dedicated thread. Metal/MPS aborts the
# process when GPU commands are encoded from several threads, even with a lock (seen with 8 eval worker threads).
_GPU_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="router-gpu")


class Router:
    def __init__(self, path: Path, max_len: int = 128):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        self.intent_to_type = json.loads((Path(path) / "intent_to_type.json").read_text())
        self.max_len = max_len

    def _predict_on_gpu_thread(self, text: str) -> tuple[DisputeType, float, str]:
        with self._torch.no_grad():
            batch = self.tok(text, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
            probs = self._torch.softmax(self.model(**batch).logits.float(), dim=-1)[0].cpu()
        idx = int(probs.argmax())
        intent = self.model.config.id2label[idx]
        return DisputeType(self.intent_to_type[intent]), float(probs[idx]), intent

    def predict(self, text: str) -> tuple[DisputeType, float, str]:
        """(dispute type, confidence of the top intent, intent name). Safe to call from any thread."""
        return _GPU_THREAD.submit(self._predict_on_gpu_thread, text).result()


@lru_cache(maxsize=4)
def _load(path: str) -> Router:
    return Router(Path(path))


def load_router(path: str) -> Router:
    """Loaded on the GPU thread too; the single-worker executor also serialises concurrent first calls."""
    return _GPU_THREAD.submit(_load, path).result()
