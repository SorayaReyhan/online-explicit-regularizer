import json
import os
import random
from datetime import datetime
from PIL import UnidentifiedImageError

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.style.use("seaborn-white")

OUTPUT_DIR = "output"


if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


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


def save_plt_img(dir, filename, **plt_kwargs):

    fullpath = os.path.join(dir, filename)
    plt.savefig(fullpath, **plt_kwargs)

    print(f'Saved image "{fullpath}"')


def write_json(dir, values, name):
    path = os.path.join(dir, f"{generate_timestr()}_{replace_punct(name)}.json")
    with open(path, "w") as f:
        json.dump(
            values, f, indent=2,
        )
    print(f'Wrote values at "{path}"')


def loss_plot(x, epochs, title):
    for t, v in x.items():
        plt.plot(list(range(t * epochs, (t + 1) * epochs)), v, label=f"task-{t}")
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend()
    plt.show()

    save_plt_img(OUTPUT_DIR, generate_filename(title))


def accuracy_plot(x, epochs, title):
    for t, v in x.items():
        xticks = [i + t * epochs for i in range(len(v))]
        plt.plot(xticks, v, label=f"task-{t}")
    plt.ylim(-0.1, 1.1)
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend()
    plt.show()

    save_plt_img(OUTPUT_DIR, generate_filename(title))


def plot(hparams, loss, acc, name):

    plt.figure()
    loss_plot(loss, hparams.epochs, f"Losses ({name})")

    plt.figure()
    accuracy_plot(acc, hparams.epochs, f"Accuracies ({name})")

    # calculate average accuracy
    avg_acc_task = [0] * hparams.num_tasks
    for task in range(hparams.num_tasks):
        # last accuracy for task
        avg_acc_task[task] = sum(acc[task]) / len(acc[task])

    # calculate last accuracy
    final_acc_task = [0] * hparams.num_tasks
    for task in range(hparams.num_tasks):
        # last accuracy for task
        final_acc_task[task] = acc[task][-1]

    # calculate average forgetting
    avg_forgetting = [0] * hparams.num_tasks
    for task in range(hparams.num_tasks):
        # last accuracy for task
        max_acc = max(acc[task])
        last_acc = acc[task][-1]
        avg_forgetting[task] = max_acc - last_acc

    # write loss, acc
    write_json(
        OUTPUT_DIR,
        {
            "avg_forgetting": avg_forgetting,
            "avg_acc_task": avg_acc_task,
            "final_acc_task": final_acc_task,
            "hparams": vars(hparams),
            "loss": loss,
            "acc": acc,
        },
        name,
    )
