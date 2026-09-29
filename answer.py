"""
answer.py - Fully LOCAL retrieval + answer generation. No external AI API.

Flow for every question:
    question
      -> greeting / thanks / empty?  handle with simple Python logic
      -> otherwise: TF-IDF cosine search over the local index for the TOP_K chunks
      -> keep only chunks whose similarity passes SIMILARITY_THRESHOLD
      -> none pass?  return the polite fallback
      -> otherwise build a concise answer from the retrieved chunks (extractive)
      -> return (answer, retrieved sources)

Everything runs on your machine with plain Python + scikit-learn. There is no
API key, no cloud LLM, and no external answer-generation service.

You can also test it from the terminal:
    python answer.py "What services does the company offer?"
"""

import os
import re
import sys
from functools import lru_cache

import joblib
from sklearn.metrics.pairwise import cosine_similarity

from ingest import INDEX_PATH, get_int_env

# Minimum cosine SIMILARITY (0 = unrelated, 1 = identical) for a chunk to count
# as relevant. Higher = stricter. This is a practical heuristic for a prototype,
# NOT a validated confidence score - tune it in .env with SIMILARITY_THRESHOLD.
DEFAULT_SIMILARITY_THRESHOLD = 0.10

FALLBACK_SENTENCE = "I don't know based on the company documents I have available."
FALLBACK_MESSAGE = (
    FALLBACK_SENTENCE
    + " Try rephrasing your question, or ask about a topic covered in the knowledge base."
)

NOT_INDEXED_MESSAGE = (
    "The knowledge base has not been indexed yet. Please run the ingestion "
    "process first with: python ingest.py"
)

GREETING_REPLY = (
    "Hey! 👋 I'm the Liquid Intelligent Technologies knowledge assistant. "
    "Ask me something about the company, its services, careers, offices, or "
    "other information in my knowledge base."
)
THANKS_REPLY = "You're welcome! 👋"

# Simple conversational triggers handled without any retrieval.
_GREETING_WORDS = {"hi", "hello", "hey", "hiya", "howdy", "yo", "greetings", "good morning", "good afternoon", "good evening"}
_THANKS_WORDS = {"thanks", "thank you", "thank you very much", "thanks a lot", "cheers", "ta", "much appreciated"}

# Query expansion: map words people commonly use to the vocabulary that actually
# appears in the knowledge base. This lifts recall for paraphrased questions
# (e.g. "industries" / "opportunities") without any external service.
_SYNONYMS = {
    "industries": ["segments", "enterprise", "business", "education", "sectors", "customers"],
    "industry": ["segment", "enterprise", "business", "sector"],
    "sectors": ["segments", "enterprise", "business", "education"],
    "operate": ["countries", "presence", "network", "africa", "operations", "offices"],
    "operates": ["countries", "presence", "network", "africa", "operations"],
    "presence": ["countries", "network", "africa", "offices", "operations"],
    "opportunities": ["careers", "jobs", "roles", "vacancies", "recruit"],
    "employees": ["careers", "staff", "people", "work", "roles"],
    "employment": ["careers", "jobs", "work", "roles"],
    "jobs": ["careers", "roles", "vacancies", "recruit"],
    "reliable": ["reliability", "resilient", "network", "connectivity"],
    "reliability": ["resilient", "network", "connectivity"],
    "connectivity": ["network", "fibre", "broadband", "connection"],
    "profile": ["about", "story", "company", "mission", "vision"],
    "mission": ["vision", "purpose", "values", "belief"],
    "technologies": ["solutions", "services", "cloud", "infrastructure"],
    "solutions": ["services", "products", "offering"],
    "cybersecurity": ["cyber", "security", "threat"],
    "locations": ["offices", "countries", "presence"],
}


def _expand_query(question: str) -> str:
    """Append known synonyms of the question's words so retrieval matches paraphrases."""
    words = _normalise(question).split()
    extra: list[str] = []
    for word in words:
        extra.extend(_SYNONYMS.get(word, []))
    if not extra:
        return question
    return question + " " + " ".join(extra)


class RAGError(Exception):
    """An error whose message is safe and clear enough to show to the user."""


# --------------------------------------------------------------------------
# Configuration helpers
# --------------------------------------------------------------------------
def get_similarity_threshold() -> float:
    raw = (os.getenv("SIMILARITY_THRESHOLD") or "").strip()
    if not raw:
        return DEFAULT_SIMILARITY_THRESHOLD
    try:
        return float(raw)
    except ValueError:
        raise RAGError(f"SIMILARITY_THRESHOLD must be a number, but got '{raw}'.")


# --------------------------------------------------------------------------
# Conversational helpers (no retrieval needed)
# --------------------------------------------------------------------------
def _normalise(text: str) -> str:
    """Lowercase and strip punctuation for simple intent matching."""
    return re.sub(r"[^a-z\s]", "", text.lower()).strip()


def small_talk_reply(question: str) -> str | None:
    """Return a canned reply for greetings/thanks, or None if it is a real question."""
    norm = _normalise(question)
    if not norm:
        return None
    if norm in _GREETING_WORDS:
        return GREETING_REPLY
    if norm in _THANKS_WORDS:
        return THANKS_REPLY
    # Short openers like "hi there" / "thanks so much" still count as small talk.
    words = norm.split()
    if len(words) <= 3 and words and words[0] in {"hi", "hello", "hey", "thanks", "thank"}:
        return THANKS_REPLY if words[0] in {"thanks", "thank"} else GREETING_REPLY
    return None


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_index():
    """Load the persisted local TF-IDF index once and reuse it."""
    if not INDEX_PATH.is_file():
        raise RAGError(NOT_INDEXED_MESSAGE)
    try:
        index = joblib.load(INDEX_PATH)
    except Exception as error:  # noqa: BLE001
        raise RAGError(f"{NOT_INDEXED_MESSAGE} (could not read the index: {error})") from error
    if not index.get("chunks"):
        raise RAGError(NOT_INDEXED_MESSAGE)
    return index


# Common words that carry little topic meaning - ignored when checking whether a
# question's distinctive terms are actually supported by the knowledge base.
_STOPWORDS = {
    "what", "who", "where", "when", "why", "how", "which", "is", "are", "was", "were",
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "does", "do", "did",
    "can", "you", "tell", "me", "about", "provide", "offer", "say", "says", "give",
    "information", "company", "companys", "its", "it", "this", "that", "with", "at",
    "by", "as", "from", "have", "has", "there", "their", "they", "them", "please",
    "liquid", "intelligent", "technologies", "liquids", "us", "our", "we",
    # Generic verbs/nouns that describe the *asking*, not the topic. Excluding them
    # keeps the grounding check focused on the real subject of the question.
    "serve", "serves", "served", "support", "supports", "offer", "offers",
    "provides", "providing", "get", "want", "need", "looking", "know", "tell",
    "regarding", "concerning",
}


def _content_words(text: str) -> set[str]:
    """Distinctive words of a piece of text (lowercased, stopwords removed)."""
    return {w for w in _normalise(text).split() if len(w) > 2 and w not in _STOPWORDS}


# Intents the public knowledge base provably does NOT cover. These questions often
# share a generic word with the documents ("company", "microsoft", "services") and
# can scrape past a similarity threshold, so we screen them explicitly and fall
# back. This keeps the bot honest instead of guessing at people, secrets or money.
_OUT_OF_SCOPE_TERMS = {
    "password", "passwords", "login", "credential", "credentials", "secret",
    "revenue", "profit", "turnover", "salary", "salaries", "earnings", "valuation",
    "ceo", "cfo", "cto", "founder", "chairman", "director", "boss",
    "favourite", "favorite", "hobby", "hobbies", "religion", "age",
}


def _is_out_of_scope(question: str, retrieved_words: set[str]) -> bool:
    """
    True if the question centres on an intent the documents do not cover (people's
    identities/preferences, credentials, financials) AND that intent word is not
    actually present in the retrieved text. If the KB really did contain, say, a
    revenue figure, the term would appear in a chunk and we would NOT block it.
    """
    q_words = set(_normalise(question).split())
    triggered = q_words & _OUT_OF_SCOPE_TERMS
    if not triggered:
        return False
    # If none of the triggered terms are actually supported by the documents,
    # the question is asking for something outside the knowledge base.
    return not (triggered & retrieved_words)


def _is_grounded(question_terms: set[str], retrieved_words: set[str], vectorizer) -> bool:
    """
    Decide whether the question's distinctive terms are actually supported by the
    retrieved documents - the core anti-hallucination guard.

    We weight each question word by its IDF (inverse document frequency) from the
    fitted TF-IDF vocabulary: rare words that carry the real intent (e.g. "ceo",
    "revenue", "password", "football") get high weight, common words low weight.
    A question is grounded only if the MOST distinctive of its supported words
    carries at least half of the total distinctive weight of the question.

    Example: "Who is the CEO of Microsoft?" -> distinctive words {ceo, microsoft}.
    "microsoft" appears in the KB but "ceo" (the higher-IDF intent word) does not,
    so the supported weight is well under half and the bot falls back.
    """
    if not question_terms:
        return True  # nothing distinctive to check - rely on similarity alone

    vocab = vectorizer.vocabulary_
    idf = vectorizer.idf_
    max_idf = float(idf.max()) if len(idf) else 1.0

    def weight(word: str) -> float:
        # Words the vocabulary has never seen are maximally distinctive (unknown).
        idx = vocab.get(word)
        return float(idf[idx]) if idx is not None else max_idf

    total = sum(weight(w) for w in question_terms)
    if total <= 0:
        return True
    supported = sum(weight(w) for w in question_terms if w in retrieved_words)
    return (supported / total) >= 0.5


def retrieve_context(question: str) -> list[dict]:
    """
    Similarity search: vectorise the (synonym-expanded) question with the stored
    TF-IDF vectorizer and rank all chunks by cosine similarity. Returns the TOP_K
    closest chunks.

    A chunk is only marked `relevant` if BOTH are true:
      1. its cosine similarity is at or above the threshold, AND
      2. the question's distinctive content words actually appear in the retrieved
         text (grounding check).

    The second test is what makes the bot honest: a question like "Who is the CEO
    of Microsoft?" shares generic words ("company", "services") with the KB and can
    scrape past the similarity threshold, but its distinctive terms ("ceo",
    "microsoft") never appear in Liquid's documents - so it correctly falls back.
    """
    top_k = get_int_env("TOP_K", 4)
    threshold = get_similarity_threshold()
    index = get_index()

    vectorizer = index["vectorizer"]
    matrix = index["matrix"]
    chunks = index["chunks"]

    question_vector = vectorizer.transform([_expand_query(question)])
    scores = cosine_similarity(question_vector, matrix)[0]

    # Highest similarity first.
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]

    question_terms = _content_words(question)
    # Words the top retrieved chunks actually contain, plus the question's own
    # synonyms so a term counts as supported when the KB uses a different word
    # for the same idea (e.g. question "countries" vs document "presence").
    retrieved_words: set[str] = set()
    for i in ranked:
        retrieved_words |= _content_words(chunks[i].get("text", ""))
    for word in list(question_terms):
        for syn in _SYNONYMS.get(word, []):
            if syn in retrieved_words:
                retrieved_words.add(word)  # the idea is supported via a synonym
                break

    grounded = _is_grounded(question_terms, retrieved_words, vectorizer)
    if _is_out_of_scope(question, retrieved_words):
        grounded = False

    results = []
    for i in ranked:
        chunk = chunks[i]
        score = float(scores[i])
        results.append(
            {
                "source": chunk.get("source", "unknown"),
                "category": chunk.get("category", "unknown"),
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", ""),
                "score": score,
                "relevant": score >= threshold and grounded,
            }
        )
    return results


# --------------------------------------------------------------------------
# Answer generation (local, extractive - no LLM)
# --------------------------------------------------------------------------
def _is_low_value(sentence: str) -> bool:
    """
    True for lines that are titles/metadata rather than real facts, e.g. Markdown
    headings, "Sources: https://..." lines, or very short fragments. These match
    question words easily but make poor answers, so we skip them.
    """
    lowered = sentence.lower()
    if lowered.startswith(("source:", "sources:")):
        return True
    if "http://" in lowered or "https://" in lowered:
        return True
    # A heading-like line: few words, no sentence punctuation, often a title.
    words = sentence.split()
    if len(words) < 4 and not sentence.endswith((".", "!", "?")):
        return True
    return False


def _split_sentences(text: str) -> list[str]:
    """Split a passage into sentence-ish / bullet-ish units, dropping headings."""
    units: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            # Markdown heading - a title, not a fact. Skip it.
            continue
        if line.startswith(("-", "*")):
            units.append(line.lstrip("-*# ").strip())
        else:
            units.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", line) if part.strip())
    return [u for u in units if u and not _is_low_value(u)]


def generate_answer(question: str, documents: list[dict]) -> str:
    """
    Build a concise, natural answer from the retrieved chunks using Python only.

    Strategy: pick the sentences from the relevant chunks that overlap most with
    the question's words, keep them in their original order, and stitch them into
    a short grounded paragraph. Nothing is invented - every sentence comes
    verbatim from the knowledge base.
    """
    if not documents:
        raise RAGError("No documents were provided to answer from.")

    question_words = set(_normalise(question).split())

    scored_sentences: list[tuple[float, int, str]] = []
    for order, doc in enumerate(documents):
        for sentence in _split_sentences(doc["text"]):
            words = set(_normalise(sentence).split())
            if not words:
                continue
            overlap = len(question_words & words)
            # Small boost from the chunk's own retrieval score so sentences from
            # the most relevant chunk are preferred on ties.
            score = overlap + doc["score"]
            scored_sentences.append((score, order, sentence))

    # Prefer sentences that share words with the question; fall back to the top
    # chunk's leading sentences if nothing overlaps (rare once past the threshold).
    scored_sentences.sort(key=lambda t: (-t[0], t[1]))
    chosen: list[str] = []
    seen: set[str] = set()
    for score, _order, sentence in scored_sentences:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(sentence)
        if len(chosen) >= 4:
            break

    if not chosen:
        # Everything relevant but no sentence extracted: use the top chunk text.
        chosen = _split_sentences(documents[0]["text"])[:3]

    body = " ".join(s.rstrip(".") + "." for s in chosen)
    return (
        "According to the company information available in the knowledge base, "
        + body
    )


# --------------------------------------------------------------------------
# Public entry point used by the Gradio app
# --------------------------------------------------------------------------
def answer_question(question: str) -> tuple[str, list[dict]]:
    """
    Answer a question. Returns (answer_text, retrieved_sources).

    Errors are returned as a readable message instead of crashing the UI.
    """
    question = (question or "").strip()
    if not question:
        return "Please type a question first.", []

    # Friendly small talk is handled without touching the knowledge base.
    small_talk = small_talk_reply(question)
    if small_talk is not None:
        return small_talk, []

    try:
        results = retrieve_context(question)
        if not results:
            return FALLBACK_MESSAGE, []

        relevant = [item for item in results if item["relevant"]]
        if not relevant:
            # Everything retrieved is too weakly related: do NOT guess.
            return FALLBACK_MESSAGE, results

        answer = generate_answer(question, relevant)
        return answer, results

    except RAGError as error:
        return f"Error: {error}", []


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print('Usage: python answer.py "your question"')
        sys.exit(1)

    reply, sources = answer_question(query)
    print("\nAnswer:\n" + reply)
    print("\nRetrieved chunks (similarity: higher = closer; used only if >= threshold):")
    for number, item in enumerate(sources, start=1):
        flag = "USED" if item["relevant"] else "not used"
        print(f"  {number}. {item['source']}  similarity={item['score']:.3f}  [{flag}]")
