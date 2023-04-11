"""
List all the available datasets.

Data format:
"dataset_name": {
    "hf_hub_url": the name of the dataset repository on the HF hub. (if specified, ignore below 3 arguments)
    "script_url": the name of the script in the local `dataset_dir` directory. (if specified, ignore below 2 arguments)
    "file_name": the name of the dataset file in the local `dataset_dir` directory. (required if hf_hub_url not specified)
    "file_sha1": the SHA-1 hash value of the dataset file. (required if hf_hub_url not specified)
    "columns": { (optional, if not provided, use the default values)
        "prompt": the name of the column in the datasets containing the prompts. (default: instruction)
        "query": the name of the column in the datasets containing the queries. (default: input)
        "response": the name of the column in the datasets containing the responses. (default: output)
        "history": the name of the column in the datasets containing the history of chat. (default: None)
    }
}
"""

CHATGLM_LASTEST_HASH = 'aa51e62ddc9c9f334858b0af44cf59b05c70148a'
DATASETS = {
    "alpaca_en": {"hf_hub_url": "tatsu-lab/alpaca"},
    "alpaca_zh": {
        "file_name": "alpaca_data_zh_51k.json",
        "file_sha1": "e655af3db557a4197f7b0cf92e1986b08fae6311"
    },
    "alpaca_gpt4_en": {
        "file_name": "alpaca_gpt4_data_en.json",
        "file_sha1": "647f4ad447bd993e4b6b6223d1be15208bab694a"
    },
    "alpaca_gpt4_zh": {
        "file_name": "alpaca_gpt4_data_zh.json",
        "file_sha1": "736d3a9d0fcbb252d1e8f902920961ecfd310e41"
    },
    "belle_0.5m": {"hf_hub_url": "BelleGroup/train_0.5M_CN"},
    "belle_1m": {"hf_hub_url": "BelleGroup/train_1M_CN"},
    "belle_2m": {"hf_hub_url": "BelleGroup/train_2M_CN"},
    "belle_dialog": {"hf_hub_url": "BelleGroup/generated_chat_0.4M"},
    "belle_math": {"hf_hub_url": "BelleGroup/school_math_0.25M"},
    "belle_multiturn": {"hf_hub_url": "BelleGroup/multiturn_chat_0.8M"},
    "belle_multiturn_chatglm": {
        "script_url": "belle_multiturn",
        "columns": {
            "prompt": "instruction",
            "query": None,
            "response": "output",
            "history": "history"
        }
    },
    "guanaco": {"hf_hub_url": "JosephusCheung/GuanacoDataset"},
    "firefly": {
        "hf_hub_url": "YeungNLP/firefly-train-1.1M",
        "columns": {
            "prompt": "input",
            "query": None,
            "response": "target",
            "history": None
        }
    }
}
