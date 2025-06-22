import os
from typing import Iterator, Optional, Union

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from scripts.prompts import SYSTEM_PROMPT, USER_PROMPT


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
        self, system_prompt: str, user_prompt: str, article: str, stream: bool = False
    ) -> Union[str, Iterator[str]]:
        """
        Run with system prompt using streaming
        """
        if stream:
            return self.run_stream(system_prompt, user_prompt, article)
        else:
            return self.run_sync(system_prompt, user_prompt, article)

    def run_sync(self, system_prompt: str, user_prompt: str, article: str) -> str:
        """
        Run with system prompt without streaming
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(
                    think_mode="/think" if self.enable_thinking else "/no_think"
                ),
            },
            {"role": "user", "content": user_prompt.format(article=article)},
        ]

        result = self.llm.create_chat_completion(
            messages=messages, **self.generation_kwargs, stream=False
        )

        return result["choices"][0]["message"]["content"]

    def run_stream(
        self, system_prompt: str, user_prompt: str, article: str
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
            {"role": "user", "content": user_prompt.format(article=article)},
        ]

        result = self.llm.create_chat_completion(
            messages=messages, **self.generation_kwargs, stream=True
        )

        for chunk in result:
            if chunk["choices"][0]["finish_reason"] is not None:
                break
            token = chunk["choices"][0]["delta"].get("content", "")
            if token:
                yield token


if __name__ == "__main__":
    model_path = "/app/data/models/Qwen3-1.7B-Q8_0.gguf"
    llm = QwenModel(model_path=model_path, enable_thinking=True)
    article = "This is test article. The main event is that the company made 1000000 dollars. The direct consequence is that the company is now bankrupt."

    response = ""
    print("Response: ", end="", flush=True)

    # with stream
    # for token in llm.run(SYSTEM_PROMPT, USER_PROMPT, article, stream=True):
    #     response += token
    #     print(token, end="", flush=True)

    # without stream
    response = llm.run(SYSTEM_PROMPT, USER_PROMPT, article, stream=False)
    print(response)
