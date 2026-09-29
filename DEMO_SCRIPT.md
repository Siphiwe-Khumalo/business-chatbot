# Demo Script (2–3 minutes)

## Introduction (≈30s)
"This is the Business Knowledge AI Bot. It answers questions about a company using RAG, Retrieval-Augmented Generation: instead of trusting what a language model remembers, the bot first looks up the relevant passages in a company knowledge base and only then writes an answer from them. That keeps answers grounded in the company's own documents."

## Show the knowledge base (≈15s)
Open the `knowledge-base/` folder and show the five Markdown files (about, services, careers, local-offices, insights). "This is a summary of Liquid's public web pages. It is the only source of company facts."

## Run ingestion (≈30s)
```bash
python ingest.py
```
"The documents are loaded, divided into smaller chunks, converted into TF-IDF vectors with scikit-learn and saved to a local index file — all offline, no API." Point at the summary: documents loaded, chunks created, local index created successfully.

## Launch the application (≈10s)
```bash
python app.py
```
Open the local address in the browser.

## Demonstrate five questions (≈75s)
**These match the current knowledge base. If you edit the files, adjust the questions, and run each one before recording to check the answer matches the documents.**

1. "What cloud services does Liquid offer?" (answerable from `services.md`).
2. "Where is the South Africa office and how do I contact it?" (answerable from `local-offices.md`).
3. "What job areas does Liquid recruit in?" (answerable from `careers.md`).
4. "What is the address of the Kenya office?" Only the South Africa office details are in the documents, so expect the "I don't know based on the company documents I have available" fallback.
5. "Who won the football World Cup in 2010?" Unrelated, so expect the fallback again.

For each answer, expand **Retrieved sources and context** and show which file and chunk were used.

## Explain RAG (≈30s)
```text
Question
↓
Similarity Retrieval
↓
Relevant Chunks
↓
Context
↓
Local Python answer builder
↓
Answer
```
"The question is turned into a TF-IDF vector, cosine similarity finds the most similar chunks, and only those are used to build the answer — assembled locally in Python from the documents themselves. If nothing is close enough, the bot says it doesn't know. So the chatbot is grounded in the supplied knowledge base, runs fully offline, and needs no API key."
