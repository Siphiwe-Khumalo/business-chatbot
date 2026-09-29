"""
app.py - A small Gradio web interface for the Business Knowledge AI Bot.

Run with:
    python app.py
then open the local URL that Gradio prints (usually http://127.0.0.1:7860).
"""

import gradio as gr

from answer import answer_question

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


if __name__ == "__main__":
    demo.launch()
