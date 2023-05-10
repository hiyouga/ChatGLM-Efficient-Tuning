# coding=utf-8

import os
import json
import time
from datasets import load_dataset
from googletrans import Translator


def main():
    split = "train"

    translator = Translator()
    def translate(text):
        if len(text) == 0:
            return ""

        local_patience = 0
        while local_patience < 5:
            try:
                result = translator.translate(text, dest="zh-cn", src="en")
                print("translate: {} -> {}".format(text, result.text))
                time.sleep(1)
                return result.text
            except Exception:
                print("Error occurred, retrying...")
                local_patience += 1
                time.sleep(5)

        raise Exception

    dataset = load_dataset("../data/hh_rlhf_en", split=split)

    if os.path.exists(f"{split}.json"):
        with open(f"{split}.json", "r", encoding="utf-8", newline="\n") as f:
            jsondata = json.load(f)
    else:
        jsondata = []

    
    global_patience = 0
    i = len(jsondata)
    while i < len(dataset):
        try:
            jsondata.append({
                "instruction": translate(dataset[i]["instruction"]),
                "output": [translate(output) for output in dataset[i]["output"]],
                "history": [[translate(hist[0]), translate(hist[1])] for hist in dataset[i]["history"]]
            })
            i += 1
            global_patience = 0

            if i % 10 == 0:
                with open(f"{split}.json", "w", encoding="utf-8", newline="\n") as f:
                    json.dump(jsondata, f, indent=2, ensure_ascii=False)

        except Exception:
            print(f"Error occurred at {i}-th data, retrying...")
            global_patience += 1
            time.sleep(100)

        if global_patience > 10:
            print("Stop")
            return

if __name__ == "__main__":
    main()
