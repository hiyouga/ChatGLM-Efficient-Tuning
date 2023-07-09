from datetime import datetime
import uuid

def current_time():
    return f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"

def generate_ckpt_name():
    return current_time() + "_" + str(uuid.uuid1())[:8]
