import json
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.style.use("seaborn-white")


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def generate_timestr():
    now = datetime.now()  # current date and time
    return now.strftime("%m-%d-%Y_%H-%M-%S")


def replace_punct(title):
    title = title.replace(" ", "_")
    title = title.replace(")", "_")
    title = title.replace("(", "_")
    return title


def generate_filename(title):
    title = replace_punct(title)
    return f"{generate_timestr()}_{title}.png"


class Logger:
    def __init__(self, hparams) -> None:
        self.hparams = hparams
        self.output = os.path.join(
            "output", f"{hparams.dataset}_{hparams.num_classes}classes_{hparams.num_tasks}tasks_{hparams.name}"
        )

        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)

    def save_plt_img(self, filename, **plt_kwargs):

        fullpath = os.path.join(self.output, filename)
        plt.savefig(fullpath, **plt_kwargs)

        print(f'Saved image "{fullpath}"')

    def write_json(self, values, name):
        path = os.path.join(self.output, f"{generate_timestr()}_{replace_punct(name)}.json")
        with open(path, "w") as f:
            json.dump(
                values, f, indent=2,
            )
        print(f'Wrote values at "{path}"')

    def loss_plot(self, x, title):
        epochs = self.hparams.epochs
        for t, v in x.items():
            plt.plot(list(range(t * epochs, (t + 1) * epochs)), v, label=f"task-{t}")
        plt.xlabel("epoch")
        plt.title(title)
        plt.legend()

        self.save_plt_img(generate_filename(title))
        plt.show()

    def accuracy_plot(self, x, title):
        epochs = self.hparams.epochs
        for t, v in x.items():
            xticks = [i + t * epochs for i in range(len(v))]
            plt.plot(xticks, v, label=f"task-{t}")
        plt.ylim(-0.1, 1.1)
        plt.xlabel("epoch")
        plt.title(title)
        plt.legend()

        self.save_plt_img(generate_filename(title))
        plt.show()

    def log_experiment_results(self, loss, acc, name):
        hparams = self.hparams

        plt.figure()
        self.loss_plot(loss, f"Losses ({name})")

        plt.figure()
        self.accuracy_plot(acc, f"Accuracies ({name})")

        # calculate average accuracy
        avg_acc_task = [0] * hparams.num_tasks
        for task in range(hparams.num_tasks):
            avg_acc_task[task] = sum(acc[task]) / len(acc[task])
        
        # calculate average R
        avg_acc_task = [0] * hparams.num_tasks
        for task in range(hparams.num_tasks):
            avg_acc_task[task] = sum(alp[task]) / len(alp[task])

        # calculate last accuracy
        final_acc_task = [0] * hparams.num_tasks
        for task in range(hparams.num_tasks):
            final_acc_task[task] = acc[task][-1]

        # calculate average forgetting
        avg_forgetting = [0] * hparams.num_tasks
        for task in range(hparams.num_tasks - 1):
            max_acc = max(acc[task])
            last_acc = acc[task][-1]
            avg_forgetting[task] = max_acc - last_acc

        # write loss, acc
        self.write_json(
            {
                "avg_forgetting": avg_forgetting,
                "avg_acc_task": avg_acc_task,
                "final_acc_task": final_acc_task,
                "hparams": self.hparams.__dict__,
                "loss": loss,
                "acc": acc,
            },
            name,
        )
