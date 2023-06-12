# coding=utf-8

import json
from typing import Tuple


PROMPT_ALONE = "Below is an instruction that describes a task. "
PROMPT_INPUT = "Below is an instruction that describes a task, paired with an input that provides further context. "
INSTRUCT = "Write a response that appropriately completes the request.\n\n### Instruction:\n"
SPLITTEXT = "\n\n### Input:\n"


def process_prompt(prompt: str) -> Tuple[str]:
    prompt = prompt.replace(PROMPT_ALONE, "")
    prompt = prompt.replace(PROMPT_INPUT, "")
    prompt = prompt.replace(INSTRUCT, "")
    results = prompt.split(SPLITTEXT)
    if len(results) == 1:
        return results[0], ""
    else:
        return results


if __name__ == "__main__":

    dataset = []

    with open("comparison_data_v2.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for example in data:
        instruction, query = process_prompt(example["user_input"])
        resp_with_score = [(float(resp["score"]), resp["response"]) for resp in example["responses_and_scores"]]
        resp_with_score.sort()

        while len(resp_with_score[0][1]) == 0:
            resp_with_score.pop(0)

        if len(resp_with_score) == 0:
            continue

        min_score, max_score = resp_with_score[0][0], resp_with_score[-1][0]
        if min_score < 5.0 and max_score > 5.0:
            dataset.append({
                "instruction": instruction,
                "input": query,
                "output": [resp_with_score[-1][1], resp_with_score[0][1]]
            })

    with open("comparison_gpt4_data_en.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
