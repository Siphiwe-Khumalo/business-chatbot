# Business Knowledge AI Bot

A small conversational assistant that answers questions from a **local company knowledge base** using Retrieval-Augmented Generation (RAG). The prototype company is Liquid Intelligent Technologies. The bot is **not** an official Liquid representative.

**Runs 100% locally — no AI API, no API key, no cloud service.** Retrieval and answer generation are done in plain Python with scikit-learn (TF-IDF + cosine similarity).

## Problem

A general-purpose LLM does not reliably know a specific company's facts, and it may sound confident while inventing them. This project grounds every answer in documents you supply, and says "I don't know" when the documents don't cover the question — without sending anything to an external service.

## Architecture

```text
Documents (knowledge-base/*.md)
↓
Loader
↓
Cleaning
↓
Chunking
↓
TF-IDF vectors (scikit-learn)
↓
Local index (kb_index.joblib)
↓
Retriever (top-K cosine similarity + relevance threshold)
↓
Answer built locally in Python from the retrieved chunks only
↓
Gradio
```

## Technologies

- **Python**: everything is plain, readable Python — retrieval and answering included.
- **scikit-learn**: TF-IDF vectorisation + cosine similarity for local retrieval.
- **joblib**: saves/loads the local search index (`kb_index.joblib`).
- **Gradio**: simple web interface.
- **python-dotenv**: loads optional local settings from `.env` (no API key).

No AI API (Gemini/OpenAI/Anthropic) and no API key are used anywhere.

## Project structure

```text
business-knowledge-ai-bot/
├── knowledge-base/        # your Markdown content goes here
│   ├── about.md
│   ├── services.md
│   ├── careers.md
│   ├── local-offices.md
│   └── insights.md
├── ingest.py              # build the local TF-IDF search index
├── answer.py              # RAG logic (retrieve + generate + fallback), fully local
├── app.py                 # Gradio interface
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── PROJECT_REPORT.md
└── DEMO_SCRIPT.md
```

`kb_index.joblib` is created when you run ingestion.

## Installation

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

(macOS/Linux: `source venv/bin/activate` instead of `venv\Scripts\activate`.)

## Configuration (optional)

**No API key is required.** The app works out of the box. If you want to tweak
behaviour, copy `.env.example` to `.env` and set any of these local settings:

| Variable | Meaning |
| --- | --- |
| `KNOWLEDGE_BASE_PATH` | Folder with the Markdown files (default `knowledge-base`). |
| `INDEX_PATH` | Where the local index is saved (default `kb_index.joblib`). |
| `TOP_K` | Chunks retrieved per question (default 4). |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking settings (default 800 / 150). |
| `SIMILARITY_THRESHOLD` | **Minimum** cosine similarity (0–1) a chunk must reach to be used (default 0.10). Higher = stricter. If nothing reaches it, the bot returns the fallback. |

## Knowledge base

The five Markdown files in `knowledge-base/` are pre-filled with summaries, written in plain words, of Liquid's public web pages (liquid.tech). Each file lists its source URLs at the top and was summarised in September 2026, so check the live site if you need current details. You can edit or extend the files at any time: the bot only knows what is in them. A file containing only a comment is skipped during ingestion. Re-run `python ingest.py` after any change.

## Ingestion

```bash
python ingest.py
```

This loads the files, cleans them, splits them into chunks (with `source`, `category` and `chunk_id` metadata), turns them into TF-IDF vectors with scikit-learn and saves the index to `kb_index.joblib`. The index is rebuilt from scratch on every run, so it always matches the current files and never contains duplicates. Run it again whenever you edit the knowledge base, and restart the app afterwards. (The app also builds the index automatically on first start if it is missing.)

## Run the application

```bash
python app.py
```

Open the local address Gradio prints (usually http://127.0.0.1:7860).

You can also test from the terminal and see the similarity scores:

```bash
python answer.py "What services does the company offer?"
```

## Installable web app (PWA)

The app is served as an installable Progressive Web App. Once it is hosted over
HTTPS, visitors can install it straight from the browser — an **Install app**
icon appears in the desktop address bar, and mobile browsers offer **Add to Home
Screen**. It then opens in its own window like a native app. This is enabled by
`pwa=True` in `app.py`; Gradio serves the `/manifest.json` automatically.

## Hosting

Deploy once and share the URL — no local Python needed. Two easy options:

### Render (one click, via the included blueprint)

1. Push this repo to GitHub (already done if you are reading this on GitHub).
2. In Render: **New +** → **Blueprint** → connect this repo. Render reads
   [`render.yaml`](render.yaml).
3. No API key is needed. Leave `APP_USERNAME` / `APP_PASSWORD` blank for a public
   demo, or set both to require sign-in.
4. Deploy. Render gives you an `https://…onrender.com` URL that is HTTPS, so the
   app is installable as described above.

### Hugging Face Space (Gradio SDK)

1. Create a new Space, SDK = **Gradio**, and push these files to it.
2. Keep the Space metadata header (`sdk: gradio`, `app_file: app.py`) at the top
   of the Space's own `README.md`.
3. No secrets are needed — the app is fully local.

Notes for either host:

- No API key or secret is required.
- On first start the app builds `kb_index.joblib` automatically if it is missing
  (a quick, local TF-IDF pass — no network calls).
- Optional login: set `APP_USERNAME` and `APP_PASSWORD` on the host.

## How it works (fully local)

1. Your question is turned into a TF-IDF vector using the same vocabulary the knowledge base was indexed with.
2. Cosine similarity ranks every chunk against your question; the top‑K are kept.
3. Chunks below the similarity threshold are discarded. If none remain, the bot returns the fallback and does **not** guess.
4. Otherwise the answer is composed in plain Python: the sentences from the retrieved chunks that best match your question are selected (verbatim, so nothing is invented) and stitched into one concise, grounded response.
5. The answer is shown with the retrieved source files and chunk text.

Simple greetings ("hi") and thanks are handled with small conversational replies, and empty input asks you to type a question — all without any external service.

## Example questions

Answerable from the supplied documents:

- "What cloud services does Liquid offer?"
- "Where is the South Africa office and how do I contact it?"
- "How long is Liquid's fibre network?"
- "What job areas does Liquid recruit in?"

Should trigger the fallback:

- "What is the address of the Kenya office?" (only the South Africa office details are in the documents)
- "Who won the football World Cup in 2010?" (unrelated)

## Limitations

- Limited knowledge base: it only knows what you paste in.
- Retrieval quality depends on the quality of the source documents.
- The similarity threshold is a practical heuristic, not a validated confidence score. Tune `SIMILARITY_THRESHOLD` using `python answer.py` output.
- Answers are assembled from the source text (extractive), so they read as short quoted facts rather than free-flowing prose. This keeps them strictly grounded and API-free.
- No live website synchronization.
- No authentication.
- Prototype, not a production enterprise system.
- Very short greetings ("hi") have no matching chunks, so they get the fallback message.

## Future improvements

- Automatic document synchronization
- Document versioning
- Hybrid (keyword + vector) search
- Reranking
- Better citations
- Authentication
- Evaluation datasets
- Production deployment
