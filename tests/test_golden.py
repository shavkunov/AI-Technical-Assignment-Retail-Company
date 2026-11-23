from langchain_core.embeddings import Embeddings

from retail_agent.golden import similar

REVENUE = {"question": "What is our monthly revenue trend?", "report": "net revenue"}
CHURN = {"question": "Why did churn spike?", "report": "no order for 90 days"}
STATE = {"question": "Why is state X underspending?", "report": "order value"}
TRIOS = [REVENUE, CHURN, STATE]
TOPICS = (("revenue", "sales"), ("churn", "leaving"), ("state", "texas"))


class TopicEmbeddings(Embeddings):
    """One axis per topic plus a shared one, so unrelated texts are not orthogonal
    (real models put unrelated text at cosine ~0.6, too)."""

    def __init__(self):
        self.embedded = []

    def _vector(self, text):
        low = text.lower()
        return [float(any(w in low for w in topic)) for topic in TOPICS] + [0.5]

    def embed_documents(self, texts):
        self.embedded += texts
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


class BrokenEmbeddings(Embeddings):
    def embed_documents(self, texts):
        raise RuntimeError("429 quota")

    def embed_query(self, text):
        raise RuntimeError("429 quota")


def test_tfidf_only_keeps_trios_sharing_a_word():
    assert similar("revenue by month", TRIOS, None, {}) == [REVENUE]


def test_embeddings_find_synonyms_tfidf_misses():
    question = "How are sales doing over time?"
    assert similar(question, TRIOS, None, {}) == []
    assert similar(question, TRIOS, TopicEmbeddings(), {}) == [REVENUE]


def test_below_threshold_is_dropped_and_top_3_kept():
    assert similar("I prefer tables", TRIOS, TopicEmbeddings(), {}) == []
    many = [{"question": f"revenue {i}", "report": "r"} for i in range(5)]
    assert len(similar("revenue", many, TopicEmbeddings(), {})) == 3


def test_embedding_error_falls_back_to_tfidf():
    assert similar("churn spike?", TRIOS, BrokenEmbeddings(), {}) == [CHURN]


def test_trio_vectors_are_cached_by_text():
    emb, cache = TopicEmbeddings(), {}
    similar("revenue", TRIOS, emb, cache)
    similar("churn", TRIOS, emb, cache)
    assert len(emb.embedded) == 3  # once per trio
    edited = [REVENUE, CHURN, {**STATE, "report": "category mix"}]
    similar("state", edited, emb, cache)
    assert len(emb.embedded) == 4  # only the edited trio again


def test_question_without_keywords_gets_nothing():
    assert similar("hi", TRIOS, None, {}) == []


def test_question_is_embedded_once_per_turn():
    # the prompt is rebuilt before every model call of a turn (up to 12)
    class Counting(TopicEmbeddings):
        queries = 0

        def embed_query(self, text):
            Counting.queries += 1
            return super().embed_query(text)

    class Failing(BrokenEmbeddings):
        queries = 0

        def embed_query(self, text):
            Failing.queries += 1
            raise RuntimeError("timeout")

    for emb in (Counting(), Failing()):
        cache = {}
        similar("churn spike?", TRIOS, emb, cache)
        assert similar("churn spike?", TRIOS, emb, cache) == [CHURN]
        assert type(emb).queries == 1


def test_question_reaches_embeddings_as_plain_str():
    # HumanMessage.text is a str subclass (TextAccessor); google-genai sends it as an
    # empty content and Gemini answers 500 INTERNAL (live: every question fell back).
    from langchain_core.messages import HumanMessage

    seen = []

    class TypeSpy(TopicEmbeddings):
        def embed_query(self, text):
            seen.append(type(text))
            return super().embed_query(text)

    similar(HumanMessage("revenue by month").text, TRIOS, TypeSpy(), {})
    assert seen == [str]
