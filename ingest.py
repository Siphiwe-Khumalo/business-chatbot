"""
ingest.py - Build a LOCAL TF-IDF search index from the Markdown knowledge base.

No external API, no API key, no cloud service. Everything runs on your machine
with plain Python + scikit-learn.

Pipeline:
    knowledge-base/*.md
        -> load_documents()   read every Markdown file
        -> clean_documents()  tidy whitespace, drop empty files
        -> split_documents()  cut into overlapping chunks + attach metadata
        -> build_index()      turn chunks into TF-IDF vectors and save them locally

Run with:
    python ingest.py

The index is REBUILT from scratch on every run, so it always matches the
current contents of knowledge-base/ and never accumulates stale records.
The result is saved to a single local file (see INDEX_PATH).
"""

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import joblib
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------------------------------
# Shared configuration (answer.py imports these so both files stay in sync)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = Path(os.getenv("KNOWLEDGE_BASE_PATH") or (BASE_DIR / "knowledge-base"))

# Where the local search index is stored. A single file, rebuilt on every run.
INDEX_PATH = Path(os.getenv("INDEX_PATH") or (BASE_DIR / "kb_index.joblib"))

# Load variables from .env (if it exists). Real environment variables win.
# Only LOCAL settings are read from .env now - no API key is ever required.
load_dotenv(BASE_DIR / ".env")


@dataclass
class Document:
    """A single knowledge-base document or chunk with its metadata."""

    page_content: str
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Small helpers for reading configuration
# --------------------------------------------------------------------------
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
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")

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

    Why chunk? Retrieval works best on small, focused passages: a whole page
    mixes many topics, so a question rarely matches all of it well. Small chunks
    give sharper matches and keep each answer passage short. The overlap repeats
    a little text at each boundary so a sentence cut in half still appears whole
    in at least one chunk.

    This is a dependency-free splitter that prefers to break on headings, then
    blank lines, then single newlines, then spaces - mirroring the old behaviour
    without needing an external text-splitter library.
    """
    if chunk_overlap >= chunk_size:
        raise RuntimeError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    separators = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", " "]
    chunks: list[Document] = []
    counters: dict[str, int] = {}

    for doc in documents:
        pieces = _split_text(doc.page_content, chunk_size, chunk_overlap, separators)
        source = doc.metadata["source"]
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            index = counters.get(source, 0)
            metadata = dict(doc.metadata)
            metadata["chunk_id"] = f"{source}-{index}"
            counters[source] = index + 1
            chunks.append(Document(page_content=piece, metadata=metadata))
    return chunks


def _split_text(text: str, chunk_size: int, chunk_overlap: int, separators: list[str]) -> list[str]:
    """Recursively split text so chunks stay under chunk_size where possible."""
    if len(text) <= chunk_size:
        return [text]

    # Find the first separator that actually appears in the text.
    separator = ""
    for candidate in separators:
        if candidate and candidate in text:
            separator = candidate
            break

    if not separator:
        # No separator left: hard-cut into fixed windows with overlap.
        step = max(1, chunk_size - chunk_overlap)
        return [text[i : i + chunk_size] for i in range(0, len(text), step)]

    parts = text.split(separator)
    chunks: list[str] = []
    current = ""
    for part in parts:
        piece = part if not current else current + separator + part
        if len(piece) <= chunk_size:
            current = piece
            continue
        if current:
            chunks.append(current)
        if len(part) > chunk_size:
            chunks.extend(_split_text(part, chunk_size, chunk_overlap, separators))
            current = ""
        else:
            current = part
    if current:
        chunks.append(current)

    # Add overlap: prepend a tail of the previous chunk to each following chunk.
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-chunk_overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        chunks = overlapped
    return chunks


# --------------------------------------------------------------------------
# Step 4 - build the local TF-IDF index
# --------------------------------------------------------------------------
def build_index(chunks: list[Document]):
    """
    Turn the chunks into TF-IDF vectors and save everything to one local file.

    TF-IDF (term frequency - inverse document frequency) scores each word by how
    often it appears in a chunk versus how common it is across all chunks. Rare,
    meaningful words get high weight; ubiquitous words get low weight. Comparing
    a question's TF-IDF vector to each chunk's with cosine similarity gives a
    good, fully-local relevance ranking - no embeddings API required.
    """
    texts = [chunk.page_content for chunk in chunks]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),  # single words and two-word phrases
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)

    index = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "chunks": [
            {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "category": chunk.metadata.get("category", "unknown"),
                "chunk_id": chunk.metadata.get("chunk_id", ""),
            }
            for chunk in chunks
        ],
    }

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, INDEX_PATH)
    return index


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    try:
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

        print("4/4 Building the local TF-IDF index ...")
        build_index(chunks)

    except (RuntimeError, FileNotFoundError) as error:
        print(f"\nERROR: {error}")
        sys.exit(1)

    print("\nSummary")
    print(f"  Documents loaded: {len(documents)}")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Index file: {INDEX_PATH}")
    print("Local knowledge-base index created successfully. No API key needed.")


if __name__ == "__main__":
    main()
