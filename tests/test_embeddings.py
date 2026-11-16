"""
tests/test_embeddings.py — all-MiniLM-L6-v2 on onnxruntime.

The model moved from PyTorch to onnxruntime to fit a 512 MB instance. The
reference numbers below were produced by sentence-transformers (PyTorch)
before the switch; matching them proves vectors already stored in Qdrant
stay comparable with new queries.

Skipped when the model files are not in the local Hugging Face cache
(the Dockerfile downloads them; locally, run embeddings.download_model()).
"""

import math

import pytest

from agents.langgraph_agent.embeddings import embeddings as E


def _model_cached() -> bool:
    try:
        from huggingface_hub import hf_hub_download
        for name in E.MODEL_FILES:
            hf_hub_download(E.EMBED_MODEL_NAME, name, revision=E.EMBED_MODEL_REVISION, local_files_only=True)
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _model_cached(), reason="embedding model not in the local cache")

# sentence-transformers 3.x + torch, all-MiniLM-L6-v2: dims 0-5, 100, 383.
TORCH_REFERENCE = {
    "leave policy": ([0.054054, 0.061716, 0.016689, -0.001863, 0.158608, 0.09337], 0.035561, -0.015023),
    "8. Annual Leave\n\nPermanent employees receive 20 paid annual leave days per year.":
        ([0.06251, 0.038214, 0.091125, 0.081903, 0.014416, 0.111356], 0.066788, -0.005835),
}


def test_matches_the_pytorch_vectors():
    for text, (head, d100, d383) in TORCH_REFERENCE.items():
        v = E.embed_text(text)
        assert len(v) == E.EMBED_DIMENSIONS
        for got, want in zip(v[:6] + [v[100], v[383]], head + [d100, d383]):
            assert got == pytest.approx(want, abs=1e-5)


def test_vectors_are_normalised_and_batches_match_singles():
    texts = ["Annual leave", "Passwords must be 12 characters", "word " * 500, ""]
    batch = E.embed_texts(texts)
    for text, v in zip(texts, batch):
        assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-5)
        single = E.embed_text(text)
        assert max(abs(a - b) for a, b in zip(single, v)) < 1e-5


def test_related_texts_are_closer():
    q, leave, password = E.embed_texts(["How many leave days do I get?", "Annual leave: 20 days a year.",
                                        "Passwords must be at least 12 characters."])
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert dot(q, leave) > dot(q, password)


def test_empty_batch():
    assert E.embed_texts([]) == []
