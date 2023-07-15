IGNORE_INDEX = -100

VALUE_HEAD_FILE_NAME = "value_head.bin"

FINETUNING_ARGS_NAME = "finetuning_args.json"

LAYERNORM_NAMES = ["layernorm"]

SUPPORTED_MODEL_LIST = [
    {
        "name": "chatglm-6b",
        "pretrained_model_name": "THUDM/chatglm-6b",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    {
        "name": "chatglm2-6b",
        "pretrained_model_name": "THUDM/chatglm2-6b",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    {
        "name": "chatglm-6b-int8",
        "pretrained_model_name": "THUDM/chatglm-6b-int8",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    {
        "name": "chatglm2-6b-int8",
        "pretrained_model_name": "THUDM/chatglm2-6b-int8",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    {
        "name": "chatglm-6b-int4",
        "pretrained_model_name": "THUDM/chatglm-6b-int4",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    {
        "name": "chatglm2-6b-int4",
        "pretrained_model_name": "THUDM/chatglm2-6b-int4",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    }
]