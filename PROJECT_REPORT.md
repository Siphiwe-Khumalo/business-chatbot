# Project Report: Business Knowledge AI Bot

## 1. Problem
General-purpose LLMs do not reliably know the facts of a specific company and can present invented details confidently. A company assistant needs answers grounded in the company's own documents.

## 2. Objective
Build a small, understandable RAG application that answers questions from a local Markdown knowledge base, cites the source files it used, and refuses to answer when the documents do not contain the information.

## 3. Architecture
Documents → loader → cleaning → chunking → Gemini embeddings → Chroma → similarity retrieval (top-K) → relevance check → Gemini generation → Gradio interface. Three scripts implement it: `ingest.py` (build the database), `answer.py` (retrieve and generate), `app.py` (interface).

## 4. Document ingestion
All `.md` files in `knowledge-base/` are loaded. Each document carries `source` (filename) and `category` (filename without extension, e.g. `local-offices`) metadata. The database is deleted and rebuilt on every run so it always mirrors the current files; this keeps behaviour simple and prevents duplicate records.

## 5. Cleaning
Cleaning removes HTML comments (the template instructions), normalises line endings and non-breaking spaces, trims trailing and repeated spaces, and collapses long runs of blank lines. Markdown headings are kept. Files with no content after cleaning are skipped with a warning.

## 6. Chunking
A recursive character splitter creates chunks of 800 characters with 150 characters of overlap (both configurable). Chunks are small enough to give focused embeddings and large enough to keep context; the overlap prevents ideas being cut at boundaries. Splitting prefers headings, then paragraphs, lines and words. Each chunk gets a `chunk_id`.

## 7. Embeddings
Chunks and questions are embedded with a Gemini embedding model chosen in `.env`, so both live in the same vector space. Embedding requests are sent in small batches with retries for free-tier rate limits.

## 8. Chroma retrieval
Chunks are stored in a persistent Chroma collection (`liquid_knowledge`, folder `chroma_db/`) using cosine distance. For each question the top-K chunks (default 4) are retrieved with their distance scores.

## 9. LLM generation
Chunks that pass the relevance threshold are passed to a Gemini chat model together with the question and system instructions. The prompt requires answers to come only from the supplied context.

## 10. Persona
The assistant is a friendly, professional knowledge assistant. It states that it is not an official company representative and has no access to internal systems.

## 11. Fallback
Two layers protect against invented answers. First, if no retrieved chunk is within `RELEVANCE_THRESHOLD`, the application returns a fixed "I don't know based on the company documents I have available" message without calling the LLM. Second, the system prompt tells the model to reply with that same sentence if the context is insufficient, and the application normalises such replies.

## 12. Design choices
Plain Python and a handful of well-known libraries; no agents, servers or extra infrastructure. Settings live in `.env`. Errors (missing key, missing database, empty question, API failures) return readable messages instead of stack traces. Retrieved chunks and their distances are shown in the UI to make the RAG process visible.

## 13. Limitations
The bot only knows what is in the knowledge base, and retrieval quality depends on document quality. The distance threshold is a heuristic that needs tuning for the chosen embedding model. LLM output can still contain errors. There is no live website sync and no authentication. It is a prototype.

## 14. Future improvements
Automatic synchronization, document versioning, hybrid search, reranking, better citations, authentication, evaluation datasets and production deployment.

## 15. Conclusion
The project demonstrates a complete, readable RAG pipeline with source visibility and a firm "I don't know" behaviour, suitable as a prototype and as a base for a larger knowledge assistant.
