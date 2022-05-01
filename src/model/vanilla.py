import torch.nn as nn
import torch.nn.functional as F

from src.model.multihead_classifier_base import MultiHeadClassifierBase


class Vanilla_cnn(MultiHeadClassifierBase):
    def __init__(self, hparams):
        super().__init__()
        self.dropout = nn.Dropout(hparams.dropout)

        self.conv1 = nn.Conv2d(1, 30, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(30, 15, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(2)

        self.linear3 = nn.Linear(15 * 14 * 14, 128)
        self.relu1 = nn.ReLU()

        self.linear4 = nn.Linear(128, 50)
        self.relu2 = nn.ReLU()

        self.heads = nn.ModuleList([nn.Linear(50, hparams.num_classes) for _ in range(hparams.num_tasks)])

    def forward(self, input, task=None):
        assert self.task is not None, "Did you forget to set the task?"
        if task is None:
            task = self.task

        input = F.max_pool2d(F.relu(self.conv1(input)), (2, 2))
        input = F.max_pool2d(F.relu(self.conv2(input)), 2)
        input = self.dropout(input)

        input = input.view(input.size(0), -1)
        input = F.relu(self.linear3(input))
        input = self.dropout(input)

        input = F.relu(self.linear4(input))

        head = self.heads[task]
        input = head(input)

        return input


class VanillaCNNMalware(MultiHeadClassifierBase):
    def __init__(self, hparams):
        super().__init__()
        self.dropout = nn.Dropout(hparams.dropout)

        # 1, 64, 64
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(2)  # 32, 32, 32

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(2)  # 64, 16, 16

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.pool3 = nn.MaxPool2d(2)  # 128, 8, 8

        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.pool4 = nn.MaxPool2d(2)  # 128, 4, 4

        self.linear5 = nn.Linear(128 * 4 * 4, 128)

        self.heads = nn.ModuleList([nn.Linear(128, hparams.num_classes) for _ in range(hparams.num_tasks)])

    def forward(self, x, task=None):
        assert self.task is not None, "Did you forget to set the task?"
        if task is None:
            task = self.task

        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.pool4(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.linear5(x)
        x = F.relu(x)

        head = self.heads[task]
        x = head(x)

        return x
