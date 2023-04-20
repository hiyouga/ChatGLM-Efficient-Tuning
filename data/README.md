Data format in `dataset_info.json`:
```json
"dataset_name": {
    "hf_hub_url": the name of the dataset repository on the HuggingFace hub. (if specified, ignore below 3 arguments)
    "script_url": the name of the directory containing a dataset loading script. (if specified, ignore below 2 arguments)
    "file_name": the name of the dataset file in the this directory. (required if above are not specified)
    "file_sha1": the SHA-1 hash value of the dataset file. (optional)
    "columns": { (optional, if not provided, use the default values)
        "prompt": the name of the column in the datasets containing the prompts. (default: instruction)
        "query": the name of the column in the datasets containing the queries. (default: input)
        "response": the name of the column in the datasets containing the responses. (default: output)
        "history": the name of the column in the datasets containing the history of chat. (default: None)
    }
}
```
