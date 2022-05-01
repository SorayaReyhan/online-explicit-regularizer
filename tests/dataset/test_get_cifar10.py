import itertools

import pytest
import torch

from src.dataset.cifar import get_cifar10, get_cifar100


@pytest.mark.parametrize(
    "num_tasks,num_classes,get_function",
    [(5, 2, get_cifar10), (3, 3, get_cifar10), (5, 2, get_cifar100), (3, 3, get_cifar100)],
)
def test_get_cifar10(num_tasks, num_classes, get_function):
    train_dataloaders, test_dataloaders = get_function(num_tasks, num_classes, batch_size=16)
    assert len(train_dataloaders) == num_tasks
    assert len(test_dataloaders) == num_tasks

    for dataloader in itertools.chain(train_dataloaders, test_dataloaders):
        iterator = iter(dataloader)
        assert len(dataloader) >= 5
        for _ in range(5):
            batch = next(iterator)
            assert len(batch) == 2
            assert isinstance(batch[0], torch.Tensor)
