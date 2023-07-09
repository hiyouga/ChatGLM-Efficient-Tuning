import matplotlib.pyplot as plt
import os
import json

def plt_loss(ckpt):
    log_dir = os.path.join(ckpt, 'trainer_log.jsonl')
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
