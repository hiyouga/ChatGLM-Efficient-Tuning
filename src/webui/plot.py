import matplotlib.pyplot as plt

def plt_loss(log_dir):
    log_dir = './path_to_sft_checkpoint/trainer_log.jsonl'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    steps = []
    losses = []
    with open(log_dir) as log_f:
        for line in log_f:
            log_info = json.loads(line)
            steps.append(log_info['current_steps'])
            losses.append(log_info['loss'])
    ax.plot(steps, losses)
    return fig
