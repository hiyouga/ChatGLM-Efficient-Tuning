import os
import json
import matplotlib.pyplot as plt
from arguments import ModelArguments
from transformers import HfArgumentParser
from transformers.trainer import TRAINER_STATE_NAME


FIGURE_NAME = "trainer_state.png"


if __name__ == "__main__":
    parser = HfArgumentParser(ModelArguments)
    model_args, = parser.parse_args_into_dataclasses()
    if model_args.checkpoint_dir is None:
        raise ValueError("Please provide checkpoint_dir")
    data = json.load(open(os.path.join(model_args.checkpoint_dir, TRAINER_STATE_NAME), 'r'))
    train_steps, train_losses = [], []
    for i in range(len(data["log_history"]) - 1):
        train_steps.append(data["log_history"][i]["step"])
        train_losses.append(data["log_history"][i]["loss"])
    plt.figure()
    plt.plot(train_steps, train_losses)
    plt.title("training loss of {}".format(model_args.checkpoint_dir))
    plt.xlabel("step")
    plt.ylabel("training loss")
    plt.savefig(os.path.join(model_args.checkpoint_dir, FIGURE_NAME), format="png", transparent=True, dpi=300)
    print("Figure saved: {}".format(os.path.join(model_args.checkpoint_dir, FIGURE_NAME)))
