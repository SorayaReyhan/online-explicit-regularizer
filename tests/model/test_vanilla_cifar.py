from unittest.mock import MagicMock

import torch

from src.model.vanilla_cifar import VanillaCIFAR


def test_vanilla_cifar():
    num_classes = 2
    hparams = MagicMock(num_classes=num_classes, num_tasks=5, dropout=0.1)
    model = VanillaCIFAR(hparams)
    assert model.task is None

    model.set_task(0)
    print(model)
    batch_size, C, H, W = 10, 3, 32, 32
    y = torch.randn(batch_size, C, H, W)
    output = model(y)
    assert output.shape == (10, num_classes)
