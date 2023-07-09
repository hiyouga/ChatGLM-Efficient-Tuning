elements = {}
web_log_dir = "logs"
data_dir = "data"
css_dir = 'src/webui/css'

settings = {
    "base_model": None,
    "path_to_model": {"llama-7b": "/home/incoming/zhengyw/llama/7b", "chatglm1": "/home/incoming/zhengyw/chatglm1"}
}

def set_base_model(model_name):
    settings["base_model"] = model_name
