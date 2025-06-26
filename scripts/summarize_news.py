import os
from typing import Iterator, Optional, Union

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT


class QwenModel:
    def __init__(
        self,
        model_name: Optional[str] = None,
        filename: Optional[str] = None,
        model_path: Optional[str] = None,
        enable_thinking: bool = False,
        enable_few_shot_examples: bool = False,
    ):
        self.enable_thinking = enable_thinking
        self.enable_few_shot_examples = enable_few_shot_examples

        if not os.path.exists(model_path):
            if model_name and filename:
                model_path = hf_hub_download(model_name, filename=filename)
            else:
                raise ValueError(
                    "Model path are required for local inference. Otherwise fill model_name and filename to huggingface download"
                )

        self.generation_kwargs = {
            "max_tokens": 4096,
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
            n_gpu_layers=-1,
            no_perf=True,
            verbose=False,
        )

    def run(
        self, system_prompt: str, article: str, stream: bool = False
    ) -> Union[str, Iterator[str]]:
        """
        Run with system prompt using streaming
        """
        if stream:
            return self.run_stream(system_prompt, article)
        else:
            return self.run_sync(system_prompt, article)

    def run_sync(self, system_prompt: str, article: str) -> str:
        """
        Run with system prompt without streaming
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(
                    think_mode="/think" if self.enable_thinking else "/no_think",
                    few_shot_examples=(
                        FEW_SHOT_EXAMPLES if self.enable_few_shot_examples else ""
                    ),
                ),
            },
            {"role": "user", "content": article},
        ]

        result = self.llm.create_chat_completion(
            messages=messages, **self.generation_kwargs, stream=False
        )

        return result["choices"][0]["message"]["content"]

    def run_stream(self, system_prompt: str, article: str) -> Iterator[str]:
        """
        Run with system prompt using streaming
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(
                    think_mode="/think" if self.enable_thinking else "/no_think",
                    few_shot_examples=(
                        FEW_SHOT_EXAMPLES if self.enable_few_shot_examples else ""
                    ),
                ),
            },
            {"role": "user", "content": article},
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

    def count_tokens(self, text: str) -> int:
        """
        Подсчитать количество токенов в тексте
        """
        tokens = self.llm.tokenize(text.encode("utf-8"))
        return len(tokens)

    def tokenize_text(self, text: str) -> list:
        """
        Токенизировать текст и вернуть список токенов
        """
        tokens = self.llm.tokenize(text.encode("utf-8"))
        return tokens


if __name__ == "__main__":
    model_path = "/app/data/models/Qwen3-1.7B-Q8_0.gguf"  # Docker
    # model_path = "/Users/danildorofeev/.lmstudio/models/Qwen/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf"  # Local (venv)
    llm = QwenModel(
        model_path=model_path, enable_thinking=True, enable_few_shot_examples=False
    )
    article = "S P 500  SPY \nThis week will be packed with economic data and earnings  It will be  to some extent  a defining week for the market  with many of the biggest companies reporting results \n\nThe good news for the S P 500 is that the index found some support around 3 282  but it may find it hard to get back above 3 300  If the index fails to rise above 3 300 on Monday  it seems likely we are heading lower to 3 255  and then 3 240 \n\n\n\n\n\n\nApple  AAPL Apple  NASDAQ AAPL  will report results on Tuesday  and this company will have a ton of pressure on it to deliver a beat and raise quarter  I own the stock  but I think it might be hard to live up to that pressure  The shares are overbought based on the RSI  Additionally  the RSI is starting to diverge from the rising stock price  There is also a rising wedge pattern forming in the chart  and it suggests to me the stock falls after results  with support first at  310  then  300  and  290 \n\n\n\n\n\n\nAMD  AMD \nAdvanced Micro Devices  NASDAQ AMD  will report results on Tuesday  too  I still think this stock will rise to around  59  60 \n\n\n\n\n\n\nMicrosoft  MSFT \nMicrosoft  NASDAQ MSFT  will report results on Wednesday  and I saw some bullish option betting in this stock a couple of weeks ago  which suggests the stock keeps rising after results  The chart shows the shares are on pace to increase to around  179   \n\n\n\n\n\n\nFacebook  FB Facebook  NASDAQ FB  will also report on Wednesday  and again I have recently seen some bullish option betting in this one too  and based on the chart  I think we can see the stock head towards  237  Free story  Strong Quarterly Results May Push Facebook s Stock Even Higher\n\n\n\n\n\n\nTesla  TSLA There is probably no company that has higher expectations on it than Tesla  NASDAQ TSLA   and honestly  anything can happen here  The risk here is higher towards a disappointment  because who knows what the right expectations are for this company given the massive move higher  I think the stock could fall back to the uptrend line  pulling the stock down to around  500 \n\n\n\n\n\n\nAmazon  AMZN Amazon  NASDAQ AMZN  will report on Thursday  and I can t remember the last time the stock has been this boring for this long  I think that is ending  It appears a cup and handle pattern is forming  and it would suggest the stock rises following results  The relative strength index is also pointing higher  and it indicates to me the stock moves up to  1970"

    response = ""
    print("Response: ", end="", flush=True)

    # with stream
    for token in llm.run(SYSTEM_PROMPT, article, stream=True):
        response += token
        print(token, end="", flush=True)

    # without stream
    # response = llm.run(SYSTEM_PROMPT, article, stream=False)
    # print(response)
