# coding=utf-8

import json


if __name__ == "__main__":

    dataset = []

    with open("comparison_data_v2.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for example in data:
        instruction = example["user_input"]
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
                "input": "",
                "output": [resp_with_score[-1][1], resp_with_score[0][1]]
            })

    with open("comparison_gpt4_data_en.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
