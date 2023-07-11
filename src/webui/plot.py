import matplotlib.pyplot as plt
import os
import json
from webui import common

def plt_loss(ckpt):
    save_dir = common.get_save_dir()
    if not save_dir:
        return None
    log_dir = os.path.join(save_dir, ckpt, 'trainer_log.jsonl')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if os.path.isfile(log_dir):
        steps = []
        losses = []
        with open(log_dir) as log_f:
            for line in log_f:
                log_info = json.loads(line)
                steps.append(log_info['current_steps'])
                losses.append(log_info['loss'])
        ax.plot(steps, losses)
    return fig
