# Business Knowledge AI Bot

A small conversational assistant that answers questions from a **local company knowledge base** using Retrieval-Augmented Generation (RAG). The prototype company is Liquid Intelligent Technologies. The bot is **not** an official Liquid representative.

## Problem

A general-purpose LLM does not reliably know a specific company's facts, and it may sound confident while inventing them. This project grounds every company-specific answer in documents you supply, and says "I don't know" when the documents don't cover the question.

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
Embeddings (Gemini)
↓
Chroma (persisted in ./chroma_db)
↓
Retriever (top-K similarity search + relevance threshold)
↓
Gemini (answers from the retrieved chunks only)
↓
Answer
↓
Gradio
```

## Technologies

- **Python**: everything is plain, readable Python.
- **LangChain** (`langchain-core`, `langchain-text-splitters`, `langchain-chroma`, `langchain-google-genai`): text splitting and the Chroma / Gemini integrations.
- **ChromaDB**: local persistent vector database.
- **Gemini API**: embeddings and the chat model (free-tier compatible; model names set in `.env`).
- **Gradio**: simple web interface.
- **python-dotenv**: loads settings from `.env`.

## Project structure

```text
business-knowledge-ai-bot/
├── knowledge-base/        # your Markdown content goes here
│   ├── about.md
│   ├── services.md
│   ├── careers.md
│   ├── local-offices.md
│   └── insights.md
├── ingest.py              # build the vector database
├── answer.py              # RAG logic (retrieve + generate + fallback)
├── app.py                 # Gradio interface
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── PROJECT_REPORT.md
└── DEMO_SCRIPT.md
```

`chroma_db/` is created when you run ingestion.

## Installation

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

(macOS/Linux: `source venv/bin/activate` instead of `venv\Scripts\activate`.)

## Environment setup

Copy `.env.example` to `.env` and fill it in:

| Variable | Meaning |
| --- | --- |
| `GEMINI_API_KEY` | Your key from Google AI Studio. Keep it only in `.env`. |
| `GEMINI_MODEL` | Chat model (default `gemini-2.5-flash`). |
| `GEMINI_EMBEDDING_MODEL` | Embedding model (default `gemini-embedding-001`). |
| `TOP_K` | Chunks retrieved per question (default 4). |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking settings (default 800 / 150). |
| `RELEVANCE_THRESHOLD` | Maximum cosine distance for a chunk to be used (default 0.65). |

If Gemini reports "model not found", check the current model names in Google AI Studio and update `.env`.

## Knowledge base

The five Markdown files in `knowledge-base/` are pre-filled with summaries, written in plain words, of Liquid's public web pages (liquid.tech). Each file lists its source URLs at the top and was summarised in September 2026, so check the live site if you need current details. You can edit or extend the files at any time: the bot only knows what is in them. A file containing only a comment is skipped during ingestion. Re-run `python ingest.py` after any change.

## Ingestion

```bash
python ingest.py
```

This loads the files, cleans them, splits them into chunks (with `source`, `category` and `chunk_id` metadata), embeds them with Gemini and stores them in Chroma. The database is rebuilt from scratch on every run, so it always matches the current files and never contains duplicates. Run it again whenever you edit the knowledge base, and restart the app afterwards.

## Run the application

```bash
python app.py
```

Open the local address Gradio prints (usually http://127.0.0.1:7860).

You can also test from the terminal and see the distance scores:

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
3. When prompted, paste your `GEMINI_API_KEY` (from
   [Google AI Studio](https://aistudio.google.com/apikey)) as the secret value.
   Leave `APP_USERNAME` / `APP_PASSWORD` blank for a public demo, or set both to
   require sign-in.
4. Deploy. Render gives you an `https://…onrender.com` URL that is HTTPS, so the
   app is installable as described above.

### Hugging Face Space (Gradio SDK)

1. Create a new Space, SDK = **Gradio**, and push these files to it.
2. Keep the Space metadata header (`sdk: gradio`, `app_file: app.py`) at the top
   of the Space's own `README.md`.
3. In **Settings → Variables and secrets**, add `GEMINI_API_KEY` as a *secret*.

Notes for either host:

- Never upload `.env`; the key lives only in the host's secret settings.
- On first start the app builds `chroma_db/` automatically if it is missing
  (this makes one round of Gemini embedding calls, so the first boot is slower).
- Optional login: set `APP_USERNAME` and `APP_PASSWORD` on the host.

## How RAG works

1. Your question is turned into an embedding (a list of numbers capturing its meaning).
2. Chroma finds the chunks whose embeddings are closest to it.
3. Chunks that are too far away (beyond the relevance threshold) are discarded. If none are left, the bot answers with the fallback and never calls the LLM.
4. Otherwise the remaining chunks are sent to Gemini together with the question and strict instructions to answer only from them.
5. The answer is shown with the retrieved source files and chunk text.

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
- The similarity threshold is a practical heuristic, not a validated confidence score. Tune `RELEVANCE_THRESHOLD` using `python answer.py` output.
- LLM responses can still contain errors.
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
