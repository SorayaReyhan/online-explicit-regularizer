import copy
import random

import numpy as np
from torch.utils.data import Dataset, Subset


def split_dataset(dataset: Dataset, num_classes: int, num_tasks: int, seed=0):
    """Splits the dataset by classes for the given number of tasks (num_tasks) and number of classes in each task (num_classes)."""

    assert hasattr(dataset, "targets"), (
        "dataset must have 'targets' attribute where targets is "
        "a 1d array containing labels for all samples in the dataset. "
        "See CIFAR10 as an example."
    )
    targets = dataset.targets

    # list of classes to be split
    classes = list(range(num_classes * num_tasks))

    # check that the dataset contains the classes we need
    existing_classes = set(targets)
    assert set(classes).issubset(
        existing_classes
    ), f"Dataset classes are {existing_classes} and cannot be split with num_classes={num_classes} and num_tasks={num_tasks}"

    # randomly shuffle classes
    random_gen = random.Random(seed)
    random_gen.shuffle(classes)

    # split classes into groups for each task
    task_classes = [classes[i : i + num_classes] for i in range(0, len(classes), num_classes)]
    class2task = {}
    for task, classes in enumerate(task_classes):
        for class_ in classes:
            class2task[class_] = task

    # function to map targets to range 0..num_classes.
    # the classes are originally in range 0..N where N is the total number of classes in the dataset
    # but we would like each dataset split have class labels in range 0..num_classes
    # the following function does this for us
    def target_transform(target):
        task = class2task[target]
        return task_classes[task].index(target)

    # create a shallow copy of the dataset and assign the target transform function
    dataset = copy.copy(dataset)
    dataset.target_transform = target_transform

    # split dataset into num_tasks subsets where each subset consists of samples of classes belonging to a certain task
    datasets = []
    for task in range(num_tasks):
        classes_for_this_tasks = task_classes[task]
        indices = np.nonzero(np.in1d(targets, classes_for_this_tasks))[0]
        subset = Subset(dataset, indices)
        datasets.append(subset)

    return datasets
