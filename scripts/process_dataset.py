import pandas as pd
from tqdm import tqdm

from prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
from scripts.summarize_news import QwenModel

if __name__ == "__main__":
    model_path = "/app/data/models/Qwen3-1.7B-Q8_0.gguf"
    dataset_path = "/app/data/dataset/clean_dataset.csv"
    # model_path = "/Users/danildorofeev/.lmstudio/models/Qwen/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf"
    llm = QwenModel(
        model_path=model_path, enable_thinking=False, enable_few_shot_examples=False
    )

    df = pd.read_csv(dataset_path)

    for index, row in tqdm(df.iterrows(), total=len(df)):
        article = row["article"]
        response = llm.run(
            SYSTEM_PROMPT.format(
                think_mode="/no_think", few_shot_examples=FEW_SHOT_EXAMPLES
            ),
            article,
            stream=False,
        )
        df.loc[index, "prediction"] = response

    df.to_csv("few_shot_dataset_by_qwen.csv", index=False)
