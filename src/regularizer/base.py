from copy import deepcopy
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader


class RegularizerHParams:
    n_batches_for_importance_estimation = -1
    saliency_momentum = 0.8


class Regularizer(nn.Module):
    def __init__(
        self, hparams: RegularizerHParams, net: nn.Module, criterion: nn.Module, device: torch.device,
    ) -> None:
        super().__init__()

        self.net = net
        self.criterion = criterion
        self.hparams = hparams
        self.device = device
        self.importance: Dict[str, torch.Tensor] = None

    def get_parameter_importance(self):
        return deepcopy(self.importance)

    def online_step(self, batch=None):
        """Called at each training step. For online regularization methods, the parameter importance is calculated 
        online at each training step. Note that in online setting, the model can only see the current batch and not 
        the previous batches or previous task datasets."""
        pass

    def task_start(self, dataloader: DataLoader):
        """Called at the start of a task with the task's dataloader."""
        pass

    def task_end(self, dataloader: DataLoader):
        """Called when done with a task and need to update the parameter importance including this task."""
        pass

    @staticmethod
    def calculate_parameter_importance(net: nn.Module, criterion: nn.Module, dataloader: DataLoader, n_batches, device):
        """Calculate parameter importance of a network on a given dataset."""
        ...
