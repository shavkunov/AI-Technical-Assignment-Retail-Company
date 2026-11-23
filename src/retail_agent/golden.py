"""Golden trios most similar to a question: hybrid search, TF-IDF + embeddings.

A toy of the production retrieval (technical-design.md, R1): the same scoring over a few
YAML files in memory instead of a vector index.
"""

import logging
import math
import re
from collections import Counter

from langchain_core.embeddings import Embeddings

WEIGHT = 0.5  # TF-IDF share of the score, the rest is embedding cosine
# ponytail: calibrated by hand on 10 questions with gemini-embedding-001 (unrelated text
# peaks at 0.31, relevant starts at 0.36); re-measure on a model change or many trios.
MIN_SCORE = 0.35
TOP_K = 3

log = logging.getLogger(__name__)


def _tokens(text: str) -> list[str]:
    # 5+ letters drop what/is/our/the
    return [w for w in re.findall(r"\w+", text.lower()) if len(w) > 4]


def _cosine(a: dict, b: dict) -> float:
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    norm = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


def _tfidf_scores(question: str, texts: list[str]) -> list[float]:
    docs = [Counter(_tokens(t)) for t in texts]
    df = Counter(w for d in docs for w in d)

    def vector(counts: Counter) -> dict:
        # smoothed IDF: a word in every trio still counts a little
        n = len(docs)
        return {
            w: tf * (math.log((1 + n) / (1 + df[w])) + 1) for w, tf in counts.items()
        }

    query = vector(Counter(_tokens(question)))
    return [_cosine(query, vector(d)) for d in docs]


def _embedding_scores(
    question: str, texts: list[str], embeddings: Embeddings, cache: dict
) -> list[float] | None:
    # The prompt is rebuilt before every model call of a turn: embed the question once,
    # and after a failure do not retry until the next question.
    asked, query = cache.get("question", (None, None))
    if asked == question and query is None:
        return None
    docs = cache.setdefault("docs", {})  # trio text -> vector: edited file re-embedded
    try:
        if asked != question:
            query = dict(enumerate(embeddings.embed_query(question)))
            cache["question"] = (question, query)
        missing = [t for t in texts if t not in docs]
        if missing:
            docs.update(zip(missing, embeddings.embed_documents(missing)))
    except Exception as error:  # quota, network: a lookup must not break the turn
        log.warning("golden: embeddings failed, TF-IDF only: %s", error)
        cache["question"] = (question, None)
        return None
    return [_cosine(query, dict(enumerate(docs[t]))) for t in texts]


def similar(
    question: str,
    trios: list[dict],
    embeddings: Embeddings | None,
    cache: dict,
) -> list[dict]:
    """Top trios by WEIGHT * TF-IDF + (1 - WEIGHT) * embedding cosine, at least
    MIN_SCORE. Without embeddings (none configured or the call failed): TF-IDF only,
    any shared word. `cache` keeps vectors between calls; pass the same dict."""
    # message.text is a str subclass that google-genai sends as empty content (500)
    question = str(question)
    texts = [f"{t['question']}\n{t.get('report', '')}" for t in trios]
    scores = _tfidf_scores(question, texts)
    cosines = None
    if embeddings is not None:
        cosines = _embedding_scores(question, texts, embeddings, cache)
    if cosines:
        scores = [WEIGHT * s + (1 - WEIGHT) * c for s, c in zip(scores, cosines)]
    floor = MIN_SCORE if cosines else 1e-9
    order = sorted(range(len(trios)), key=lambda i: -scores[i])
    return [trios[i] for i in order if scores[i] >= floor][:TOP_K]
