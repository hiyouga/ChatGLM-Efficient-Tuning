Data format in `dataset_info.json`:
```json
"dataset_name": {
    "hf_hub_url": "the name of the dataset repository on the HuggingFace hub. (if specified, ignore below 3 arguments)",
    "script_url": "the name of the directory containing a dataset loading script. (if specified, ignore below 2 arguments)",
    "file_name": "the name of the dataset file in the this directory. (required if above are not specified)",
    "file_sha1": "the SHA-1 hash value of the dataset file. (optional)",
    "columns": {
        "prompt": "the name of the column in the datasets containing the prompts. (default: instruction)",
        "query": "the name of the column in the datasets containing the queries. (default: input)",
        "response": "the name of the column in the datasets containing the responses. (default: output)",
        "history": "the name of the column in the datasets containing the history of chat. (default: None)"
    }
}
```

`dataset_info.json` 中的数据集定义格式：
```json
"数据集名称": {
    "hf_hub_url": "HuggingFace上的项目地址（若指定，则忽略下列三个参数）",
    "script_url": "包含数据加载脚本的本地文件夹名称（若指定，则忽略下列两个参数）",
    "file_name": "该目录下数据集文件的名称（若上述参数未指定，则此项必需）",
    "file_sha1": "数据集文件的SHA-1哈希值（可选）",
    "columns": {
        "prompt": "数据集代表提示词的表头名称（默认：instruction）",
        "query": "数据集代表请求的表头名称（默认：input）",
        "response": "数据集代表回答的表头名称（默认：output）",
        "history": "数据集代表历史对话的表头名称（默认：None）"
    }
}
```
