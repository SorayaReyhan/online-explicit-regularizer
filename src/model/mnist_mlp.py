import torch.nn.functional as F
from torch import nn

from src.model.multihead_classifier_base import MultiHeadClassifierBase


class MnistMLPHParams:
    hidden_size: int = 200
    num_tasks: int = 3
    num_classes: int = 10


class MnistMLP(MultiHeadClassifierBase):
    def __init__(self, hparams: MnistMLPHParams):
        super(MnistMLP, self).__init__()

        self.drop = nn.Dropout(hparams.dropout)

        hidden_size = hparams.hidden_size
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

        self.heads = nn.ModuleList([nn.Linear(hidden_size, hparams.num_classes) for _ in range(hparams.num_tasks)])

    def forward(self, input, task=None):
        if task is None:
            task = self.task

        x = F.relu(self.fc1(input))
        x = self.drop(x)

        x = F.relu(self.fc2(x))
        x = self.drop(x)

        x = F.relu(self.fc3(x))
        x = self.drop(x)

        head = self.heads[task]
        x = head(x)

        return x
