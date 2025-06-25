import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from prompts import SYSTEM_PROMPT

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="api_key",
)

# df = pd.read_csv("/app/data/dataset.csv")
df = pd.read_csv(
    "/Users/danildorofeev/Desktop/financial-news-summarizer/data/dataset.csv"
)

for index, row in tqdm(df.iterrows(), total=len(df)):
    article = row["article"]
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": article}
    #     ]
    # )
    response = client.chat.completions.create(
        model="qwen/qwen3-235b-a22b",
        stream=True,
        extra_body={"reasoning": True},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": article},
        ],
    )
    thinking, answer = "", ""
    for chunk in response:
        d = chunk.choices[0].delta
        if hasattr(d, "reasoning_content"):
            thinking += d.reasoning_content
        if hasattr(d, "content"):
            answer += d.content

    df.loc[index, "ground_truth"] = response.choices[0].message.content
    break

df.to_csv("reasoning_dataset_by_qwen.csv", index=False)
