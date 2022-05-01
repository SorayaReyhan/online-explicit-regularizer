import pytest
from torchvision.datasets import CIFAR10

from src.dataset.utils import split_dataset


@pytest.mark.parametrize(
    "num_tasks,num_classes",
    [
        (5, 2),
        (3,3),
        (10,1)
    ],
)
def test_split_dataset(num_tasks, num_classes):
    dataset = CIFAR10(root="data", download=True)
    assert max(dataset.targets) == 9

    subsets = split_dataset(dataset, num_classes=num_classes, num_tasks=num_tasks)

    # check that all subsets contain different split
    for task in range(num_tasks):
        subset = subsets[task]
        # retrieve all targets samples
        targets = set(subset[i][1] for i in range(len(subset)))
        assert len(targets) == num_classes
        assert max(targets) < num_classes
