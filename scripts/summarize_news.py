import os
from typing import Iterator, Optional

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from prompts import SYSTEM_PROMPT, USER_PROMPT


class QwenModel:
    def __init__(
        self,
        model_name: Optional[str] = None,
        filename: Optional[str] = None,
        model_path: Optional[str] = None,
        enable_thinking: bool = False,
    ):
        self.enable_thinking = enable_thinking

        if not os.path.exists(model_path):
            if model_name and filename:
                model_path = hf_hub_download(model_name, filename=filename)
            else:
                raise ValueError(
                    "Model path are required for local inference. Otherwise fill model_name and filename to huggingface download"
                )

        self.generation_kwargs = {
            "max_tokens": 32768,
            "stop": ["<|im_end|>", "<|endoftext|>"],
            "temperature": 0.6,
            "top_p": 0.90,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
        }

        self.llm = Llama(
            model_path=model_path,
            n_ctx=16000,
            n_threads=32,
            n_gpu_layers=0,
            no_perf=True,
            verbose=False,
        )

    def run(
        self, system_prompt: str, user_prompt: str, stream: bool = False
    ) -> Iterator[str]:
        """
        Run with system prompt using streaming
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(
                    think_mode="/think" if self.enable_thinking else "/no_think"
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        result = self.llm.create_chat_completion(
            messages=messages, **self.generation_kwargs, stream=stream
        )
        if stream:
            for chunk in result:
                if chunk["choices"][0]["finish_reason"] is not None:
                    break
                token = chunk["choices"][0]["delta"].get("content", "")
                if token:
                    yield token
        else:
            return result["choices"][0]["message"]["content"]


if __name__ == "__main__":
    model_path = "/app/data/models/Qwen/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf"
    llm = QwenModel(model_path=model_path, enable_thinking=False)
    response = ""
    print("Response: ", end="", flush=True)

    # with stream
    for token in llm.run(SYSTEM_PROMPT, USER_PROMPT, stream=True):
        response += token
        print(token, end="", flush=True)

    # without stream
    # response = llm.run(SYSTEM_PROMPT, USER_PROMPT)
    # print(response)
