import os
from typing import Optional

from huggingface_hub import hf_hub_download
from llama_cpp import Llama


class DeepSeekModel:
    def __init__(
        self,
        model_name: Optional[str] = None,
        filename: Optional[str] = None,
        model_path: Optional[str] = None,
        enable_thinking: bool = True,
    ):
        if not os.path.exists(model_path):
            if model_name and filename:
                model_path = hf_hub_download(model_name, filename=filename)
            else:
                raise ValueError(
                    "Model path are required for local inference. Otherwise fill model_name and filename to huggingface download"
                )

        self.generation_kwargs = {
            "max_tokens": 32768,  # Adequate output length for most queries
            "stop": ["<|im_end|>", "<|endoftext|>"],  # Qwen3 stop tokens
            "echo": False,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
        }

        self.llm = Llama(
            model_path=model_path,
            n_ctx=16000,  # Context length to use
            n_threads=32,  # Number of CPU threads to use
            n_gpu_layers=0,  # Number of model layers to offload to GPU
        )

    def run(self, prompt: str) -> str:
        return self.llm(prompt, **self.generation_kwargs)


if __name__ == "__main__":
    # from local gguf model
    # model_path = "/app/data/models/lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf"
    model_path = "/app/data/models/Qwen/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf"
    summarizer = DeepSeekModel(model_path=model_path, enable_thinking=True)

    # Qwen3 chat format
    prompt = """<|im_start|>user
Summarize the key facts from the following financial news article in 2-3 neutral sentences.
Focus on the main event, numbers, and direct consequences.
Do not include questions, opinions, or calls to action.

Article:
This is test article. The main event is that the company made 1000000 dollars. The direct consequence is that the company is now bankrupt.

Please provide your response in the following JSON format with two fields:
- "reasoning": Your step-by-step analysis and thinking process
- "summary": The final 2-3 sentence summary.
<|im_end|>
<|im_start|>assistant
"""

    print("Generating summary...")
    result = summarizer.run(prompt)
