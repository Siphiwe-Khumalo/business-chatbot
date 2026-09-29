"""
app.py - A small Gradio web interface for the Business Knowledge AI Bot.

Run with:
    python app.py
then open the local URL that Gradio prints (usually http://127.0.0.1:7860).
"""

import os

import gradio as gr

from answer import answer_question
from ingest import CHROMA_DIR
from ingest import main as build_database

TITLE = "Liquid Intelligent Technologies — Business Knowledge AI"
DESCRIPTION = "Ask questions about information contained in the provided company knowledge base."

# Generic starter questions. The last one is unrelated on purpose, to show the fallback.
EXAMPLE_QUESTIONS = [
    ["What services does the company offer?"],
    ["Where are the company's offices located?"],
    ["What careers information is available?"],
    ["Who won the football World Cup in 2010?"],
]


def format_sources(sources: list[dict]) -> str:
    """Render the retrieved chunks as Markdown so the RAG step is visible."""
    if not sources:
        return "_No sources were retrieved._"

    blocks = []
    for number, item in enumerate(sources, start=1):
        status = (
            "within relevance threshold, sent to the model"
            if item["relevant"]
            else "beyond relevance threshold, NOT sent to the model"
        )
        quoted_text = item["text"].replace("\n", "\n> ")
        blocks.append(
            f"**{number}. Source: {item['source']}** · distance {item['score']:.3f} · {status}\n\n> {quoted_text}"
        )
    return "\n\n".join(blocks)


def ask(question: str, show_sources: bool):
    answer, sources = answer_question(question)
    if show_sources:
        return answer, format_sources(sources)
    return answer, "_Sources hidden. Tick “Show retrieved sources” to display them._"


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}\n\n{DESCRIPTION}")

    question_box = gr.Textbox(
        label="Your question",
        placeholder="Type a question about the company documents…",
        lines=2,
    )
    show_sources_box = gr.Checkbox(label="Show retrieved sources", value=True)
    ask_button = gr.Button("Ask", variant="primary")

    gr.Markdown("### Answer")
    answer_output = gr.Markdown()

    with gr.Accordion("Retrieved sources and context", open=True):
        sources_output = gr.Markdown()

    gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=question_box)

    gr.Markdown(
        "*Prototype knowledge assistant. It is not an official Liquid representative "
        "and only answers from the documents in the local knowledge base.*"
    )

    ask_button.click(ask, inputs=[question_box, show_sources_box], outputs=[answer_output, sources_output])
    question_box.submit(ask, inputs=[question_box, show_sources_box], outputs=[answer_output, sources_output])


def ensure_database() -> None:
    """
    On a fresh host (Hugging Face Space, Render, ...) there is no chroma_db/ yet,
    because it is generated data and is not stored in Git. Build it once at startup.
    """
    if CHROMA_DIR.is_dir() and any(CHROMA_DIR.iterdir()):
        return
    print("No vector database found - building it from knowledge-base/ ...")
    try:
        build_database()
    except SystemExit:
        print("Database build failed (check GEMINI_API_KEY). The app will show an error until it is fixed.")


if __name__ == "__main__":
    ensure_database()

    # Hosting platforms set PORT (Render) or SPACE_ID (Hugging Face); they need 0.0.0.0.
    # Locally we stay on 127.0.0.1 so the app is not exposed to your network.
    hosted = bool(os.getenv("PORT") or os.getenv("SPACE_ID"))

    # Optional login: set APP_USERNAME and APP_PASSWORD on the host to require it.
    username, password = os.getenv("APP_USERNAME"), os.getenv("APP_PASSWORD")

    demo.launch(
        server_name="0.0.0.0" if hosted else "127.0.0.1",
        server_port=int(os.getenv("PORT", "7860")),
        auth=(username, password) if username and password else None,
        # Serve a web app manifest so the site is installable from the browser
        # ("Install app" on desktop, "Add to Home Screen" on mobile).
        pwa=True,
    )
