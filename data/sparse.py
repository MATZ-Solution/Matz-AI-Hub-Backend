"""
data/sparse.py
---------------
BM25-style sparse vectors for hybrid retrieval (task C2).

A document vector holds, per term, BM25's saturated term frequency
    tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len))
and a query vector holds 1.0 per distinct term. The Qdrant sparse vector is
configured with the IDF modifier, so Qdrant multiplies in each term's IDF
over the whole collection at query time — together that is BM25.

Terms are lowercased words with English stop words removed and a light
plural stemmer ("policies" → "policy"), hashed to stable 31-bit indices.
"""

import re
import zlib
from collections import Counter

K1 = 1.2
B = 0.75
# Average chunk length in terms. C1 chunks are 400-700 tokens, about 300-500
# words before stop words are dropped; a fixed value keeps vectors stable.
AVG_LEN = 250.0

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_STOP_WORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most
my myself no nor not now of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these they this those
through to too under until up very was we were what when where which while who whom why will with
would you your yours yourself yourselves
""".split())


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def terms(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower()):
        w = w.split("'")[0]
        if w and w not in _STOP_WORDS:
            out.append(_stem(w))
    return out


def term_index(term: str) -> int:
    return zlib.crc32(term.encode("utf-8")) & 0x7FFFFFFF


def _vector(weights: dict[int, float]) -> tuple[list[int], list[float]]:
    indices = sorted(weights)
    return indices, [weights[i] for i in indices]


def encode_document(text: str) -> tuple[list[int], list[float]]:
    """(indices, values) of a chunk's sparse vector."""
    words = terms(text)
    if not words:
        return [], []
    norm = K1 * (1 - B + B * len(words) / AVG_LEN)
    weights: dict[int, float] = {}
    for term, tf in Counter(words).items():
        idx = term_index(term)
        weights[idx] = weights.get(idx, 0.0) + tf * (K1 + 1) / (tf + norm)
    return _vector(weights)


def encode_query(text: str) -> tuple[list[int], list[float]]:
    """(indices, values) of a query's sparse vector: 1.0 per distinct term."""
    return _vector({term_index(t): 1.0 for t in set(terms(text))})
