"""
agents/langgraph_agent/embeddings/embeddings.py
------------------------------------------------
Single place for all embedding logic (ingestion and retrieval).

Model: sentence-transformers/all-MiniLM-L6-v2 — 384 dimensions, unchanged.

It runs on onnxruntime instead of PyTorch: the same model weights,
exported to ONNX by the model's authors (onnx/model.onnx in the model
repository), with the same steps sentence-transformers applies:
tokenize (max 256 tokens) → model → mean pooling over real tokens →
L2 normalisation. Vectors match the PyTorch ones to about 1e-6, so points
already in Qdrant stay valid (tests/test_embeddings.py checks this).

Why: PyTorch alone needs ~1 GB of RAM; onnxruntime keeps the whole API small
enough for a 512 MB free instance (see docs/DECISIONS.md).

Files come from the Hugging Face cache, downloaded on first use (the
Dockerfile downloads them at build time so a container starts offline).
"""

import threading

import numpy as np

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Pinned so every machine runs exactly the same weights and tokenizer.
EMBED_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBED_DIMENSIONS = 384
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 32
MODEL_FILES = ("onnx/model.onnx", "tokenizer.json")

_lock = threading.Lock()
_session = None
_tokenizer = None


def _model_file(name: str) -> str:
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(EMBED_MODEL_NAME, name, revision=EMBED_MODEL_REVISION, local_files_only=True)
    except Exception:
        # First run on a new machine: download into the cache.
        return hf_hub_download(EMBED_MODEL_NAME, name, revision=EMBED_MODEL_REVISION)


def download_model() -> None:
    """Fetches the model files into the cache (used by the Dockerfile)."""
    for name in MODEL_FILES:
        _model_file(name)


def _load():
    """The shared ONNX session and tokenizer, loaded once per process."""
    global _session, _tokenizer
    with _lock:
        if _session is None:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(_model_file("tokenizer.json"))
            tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
            tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

            options = ort.SessionOptions()
            # One worker process on a small machine: keep threads modest.
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            _session = ort.InferenceSession(
                _model_file("onnx/model.onnx"), sess_options=options, providers=["CPUExecutionProvider"],
            )
            _tokenizer = tokenizer
    return _session, _tokenizer


def _encode(texts: list[str]) -> np.ndarray:
    session, tokenizer = _load()
    input_names = {i.name for i in session.get_inputs()}
    out = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = tokenizer.encode_batch(texts[start:start + BATCH_SIZE])
        ids = np.array([e.ids for e in batch], dtype=np.int64)
        mask = np.array([e.attention_mask for e in batch], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        tokens = session.run(None, feed)[0]                       # (batch, seq, 384)
        weights = mask[:, :, None].astype(np.float32)
        pooled = (tokens * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        out.append(pooled / np.clip(norms, 1e-12, None))
    return np.vstack(out) if out else np.zeros((0, EMBED_DIMENSIONS), dtype=np.float32)


def embed_text(text: str) -> list[float]:
    """One string → a 384-float list (query embedding at retrieval time)."""
    return _encode([text])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Many strings → 384-float lists (ingestion), embedded in batches."""
    return _encode(list(texts)).tolist()
