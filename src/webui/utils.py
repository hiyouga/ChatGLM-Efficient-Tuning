from datetime import datetime
import uuid
import json

def current_time():
    return f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"

def generate_ckpt_name():
    return current_time() + "_" + str(uuid.uuid1())[:8]

def get_eval_result(path):
    with open(path) as f:
        result = json.load(f)
    return f"```javascript \
        {result} \
    ```"
