from pathlib import Path

import streamlit as st

from prompts import SYSTEM_PROMPT
from scripts.summarize_news import QwenModel


def load_gguf_models():
    """Load available GGUF models from the models directory"""

    models_dir = Path("/app/data/models")
    if not models_dir.exists():
        return []

    gguf_files = list(models_dir.glob("*.gguf"))
    return [model.name for model in gguf_files]


@st.cache_resource(show_spinner="🔄 Loading model…")
def get_model(path: str):
    return QwenModel(
        model_path=path, enable_thinking=True, enable_few_shot_examples=False
    )


def summarize_text(text: str, model_name: str, llm: QwenModel | None):
    """
    Stream summarize text using GGUF model or fallback to simple summarization
    """
    if model_name != "Simple Fallback" and llm:
        return llm.run(SYSTEM_PROMPT, text, stream=True)
    else:
        # Simple fallback summarization
        words = text.split()
        if len(words) <= 50:
            return text
        sentences = text.split(". ")
        summary = ". ".join(sentences[:3])
        if not summary.endswith("."):
            summary += "."
        return f"**Simple Fallback Summary:**\n{summary}"


def main():
    st.set_page_config(
        page_title="Financial News Summarizer", page_icon="📰", layout="wide"
    )

    st.title("📰 Financial News Summarizer")
    st.markdown("---")

    # Load available models
    available_models = load_gguf_models()
    model_options = ["Simple Fallback"] + available_models

    # Model selection
    col1, col2 = st.columns([2, 1])
    with col1:
        article_text = st.text_area(
            label="Enter article text for summarization:",
            placeholder="Paste your article text here that you want to summarize...",
            height=200,
            help="Enter the full text of the article you want to summarize",
        )

    with col2:
        st.markdown("### 🎛️ Model Settings")
        selected_model = st.selectbox(
            "Choose model:",
            options=model_options,
            help="Select a GGUF model for summarization or use simple fallback",
        )

        if selected_model != "Simple Fallback":
            llm = get_model(f"/app/data/models/{selected_model}")
            st.success(f"✅ Model loaded: {selected_model}")
        else:
            llm = None
            st.info("ℹ️ Using simple text processing")

        # Show available models
        if available_models:
            st.markdown("**Available GGUF Models:**")
            for model in available_models:
                st.markdown(f"- `{model}`")
        else:
            st.warning("⚠️ No GGUF models found in `/app/data/models/`")

    # Summarize button
    if st.button("🚀 Summarize", type="primary", use_container_width=True):
        if article_text.strip():
            with st.spinner("Processing text..."):
                result_placeholder = st.empty()

                response = summarize_text(article_text, selected_model, llm)

                if isinstance(response, str):
                    result_placeholder.markdown(response)
                else:
                    collected = ""
                    for tok in response:
                        collected += tok
                        result_placeholder.markdown(collected)

        else:
            st.error("⚠️ Please enter text for summarization!")


if __name__ == "__main__":
    main()
