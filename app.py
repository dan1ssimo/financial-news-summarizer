from pathlib import Path

import streamlit as st

from scripts.prompts import SYSTEM_PROMPT, USER_PROMPT
from scripts.summarize_news import QwenModel


def load_gguf_models():
    """Load available GGUF models from the models directory"""

    models_dir = Path("/app/data/models")
    if not models_dir.exists():
        return []

    gguf_files = list(models_dir.glob("*.gguf"))
    return [model.name for model in gguf_files]


def summarize_text(text, model_name=None, llm=None):
    """
    Stream summarize text using GGUF model or fallback to simple summarization
    """
    if model_name and model_name != "Simple Fallback" and llm is not None:
        try:
            return llm.run(SYSTEM_PROMPT, USER_PROMPT, text, stream=False)
        except Exception as e:
            yield f"**Error with model {model_name}:**\n{str(e)}\n"
    else:
        # Simple fallback summarization
        words = text.split()
        if len(words) <= 50:
            yield text
        else:
            sentences = text.split(". ")
            summary_sentences = sentences[:3]
            summary = ". ".join(summary_sentences) + (
                "." if not summary_sentences[-1].endswith(".") else ""
            )
            yield f"**Simple Fallback Summary:**\n{summary}"


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
            llm = QwenModel(
                model_path=f"/app/data/models/{selected_model}", enable_thinking=True
            )
            st.success(f"✅ Model loaded: {selected_model}")
        else:
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
                st.markdown("### 📋 Model Results")
                st.markdown("---")

                # Создаем placeholder для стриминга
                result_placeholder = st.empty()

                full_response = summarize_text(article_text, selected_model, llm)
                result_placeholder.markdown(full_response)

        else:
            st.error("⚠️ Please enter text for summarization!")


if __name__ == "__main__":
    main()
