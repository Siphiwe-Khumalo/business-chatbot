"""
ingest.py - Build the Chroma vector database from the Markdown knowledge base.

Pipeline:
    knowledge-base/*.md
        -> load_documents()    read every Markdown file
        -> clean_documents()   tidy whitespace, drop empty files
        -> split_documents()   cut into overlapping chunks + attach metadata
        -> create_vector_store()  embed chunks with Gemini and store them in Chroma

Run with:
    python ingest.py

The database is REBUILT from scratch on every run, so it always matches the
current contents of knowledge-base/ and never accumulates duplicate records.
"""

import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------------------------------
# Shared configuration (answer.py imports these so both files stay in sync)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge-base"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "liquid_knowledge"

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
PLACEHOLDER_API_KEY = "your_api_key_here"

# Load variables from .env (if it exists). Real environment variables win.
load_dotenv(BASE_DIR / ".env")


# --------------------------------------------------------------------------
# Small helpers for reading configuration
# --------------------------------------------------------------------------
def get_api_key() -> str:
    """Return the Gemini API key or raise a clear error if it is missing."""
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key or key == PLACEHOLDER_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and paste "
            "your Gemini API key into it."
        )
    return key


def get_int_env(name: str, default: int, minimum: int = 1) -> int:
    """Read a whole-number setting from the environment with validation."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be a whole number, but got '{raw}'.")
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}, but got {value}.")
    return value


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create the Gemini embedding model (name comes from .env)."""
    model = (os.getenv("GEMINI_EMBEDDING_MODEL") or "").strip() or DEFAULT_EMBEDDING_MODEL
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=get_api_key())


def get_chroma_store(embeddings) -> Chroma:
    """
    Open (or create) the persistent Chroma collection.

    Cosine distance is used so that scores are easy to read: 0 means
    "identical direction", larger numbers mean "less similar".
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


# --------------------------------------------------------------------------
# Step 1 - load
# --------------------------------------------------------------------------
def load_documents(directory: Path | None = None) -> list[Document]:
    """Read every .md file in the knowledge-base folder."""
    directory = directory or KNOWLEDGE_BASE_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"Knowledge-base folder not found: {directory}")

    documents = []
    for path in sorted(directory.glob("*.md")):
        # utf-8-sig quietly removes the BOM that some Windows editors add.
        text = path.read_text(encoding="utf-8-sig")
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,  # e.g. services.md
                    "category": path.stem,  # e.g. services, local-offices
                },
            )
        )
    return documents


# --------------------------------------------------------------------------
# Step 2 - clean
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Remove noise but keep the meaning and the Markdown headings.

    - drops HTML comments (the placeholder instructions in the template files)
    - normalises line endings and non-breaking spaces
    - trims trailing spaces and collapses repeated spaces inside a line
    - collapses 3+ blank lines into one blank line
    """
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")

    cleaned_lines = []
    for line in text.split("\n"):
        line = line.rstrip()
        # Collapse runs of spaces/tabs that follow text (keeps leading indentation).
        line = re.sub(r"(?<=\S)[ \t]{2,}", " ", line)
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_documents(documents: list[Document]) -> list[Document]:
    """Clean every document and skip files that have no real content."""
    cleaned = []
    for doc in documents:
        text = clean_text(doc.page_content)
        if not text:
            print(f"  ! Skipping {doc.metadata['source']}: no content yet (still the empty template?)")
            continue
        cleaned.append(Document(page_content=text, metadata=dict(doc.metadata)))
    return cleaned


# --------------------------------------------------------------------------
# Step 3 - chunk
# --------------------------------------------------------------------------
def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """
    Split documents into overlapping chunks.

    Why chunk? An embedding represents the meaning of a piece of text. A whole
    page mixes many topics, so its single vector would be vague and a question
    would rarely match it well. Small chunks give sharp, focused vectors, and
    they also keep the prompt sent to the LLM short. The overlap repeats a
    little text at each boundary so a sentence cut in half still appears whole
    in at least one chunk.
    """
    if chunk_overlap >= chunk_size:
        raise RuntimeError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Prefer to break at headings, then paragraphs, lines, and finally words.
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Give every chunk a stable id such as "services.md-0", "services.md-1", ...
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        index = counters.get(source, 0)
        chunk.metadata["chunk_id"] = f"{source}-{index}"
        counters[source] = index + 1
    return chunks


# --------------------------------------------------------------------------
# Step 4 - embed and store
# --------------------------------------------------------------------------
def _is_retryable(error: Exception) -> bool:
    """Free-tier limits and brief outages are worth retrying; other errors are not."""
    message = str(error).lower()
    return any(word in message for word in ("429", "quota", "rate", "resource_exhausted", "503", "unavailable"))


def add_chunks_in_batches(store: Chroma, chunks: list[Document], batch_size: int = 20) -> None:
    """Embed and store chunks in small batches, retrying on free-tier rate limits."""
    max_attempts = 4
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        ids = [chunk.metadata["chunk_id"] for chunk in batch]

        for attempt in range(1, max_attempts + 1):
            try:
                store.add_documents(batch, ids=ids)
                break
            except Exception as error:  # noqa: BLE001 - we re-raise below
                if attempt == max_attempts or not _is_retryable(error):
                    raise RuntimeError(f"Embedding/storing chunks failed: {error}") from error
                wait_seconds = 10 * attempt
                print(f"  ... rate limited, waiting {wait_seconds}s before retrying")
                time.sleep(wait_seconds)

        print(f"  Stored {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")


def create_vector_store(chunks: list[Document]) -> Chroma:
    """Rebuild the Chroma collection from the given chunks and persist it."""
    embeddings = get_embeddings()

    # Start clean: delete the old collection so nothing is duplicated or stale.
    old_store = get_chroma_store(embeddings)
    old_store.delete_collection()

    store = get_chroma_store(embeddings)
    add_chunks_in_batches(store, chunks)
    # Chroma writes to ./chroma_db automatically because persist_directory is set.
    return store


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    try:
        get_api_key()
        chunk_size = get_int_env("CHUNK_SIZE", 800)
        chunk_overlap = get_int_env("CHUNK_OVERLAP", 150, minimum=0)

        print("1/4 Loading documents ...")
        documents = load_documents()
        print(f"  Found {len(documents)} Markdown file(s) in {KNOWLEDGE_BASE_DIR.name}/")

        print("2/4 Cleaning documents ...")
        documents = clean_documents(documents)
        if not documents:
            raise RuntimeError(
                "No usable content found. Paste real company text into the "
                "Markdown files in knowledge-base/ and run this script again."
            )

        print(f"3/4 Splitting into chunks (size={chunk_size}, overlap={chunk_overlap}) ...")
        chunks = split_documents(documents, chunk_size, chunk_overlap)

        print("4/4 Creating embeddings and storing them in Chroma ...")
        create_vector_store(chunks)

    except (RuntimeError, FileNotFoundError) as error:
        print(f"\nERROR: {error}")
        sys.exit(1)

    print("\nSummary")
    print(f"  Documents loaded: {len(documents)}")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Database folder: {CHROMA_DIR}")
    print("Vector database created successfully.")


if __name__ == "__main__":
    main()
