import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.multihead_classifier_base import MultiHeadClassifierBase


class Vanilla_cnn(MultiHeadClassifierBase):
    def __init__(self, hparams):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 30, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(30, 15, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)

        self.linear3 = nn.Linear(15 * 14 * 14, 128)
        self.relu1 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)

        self.linear4 = nn.Linear(128, 50)
        self.relu2 = nn.ReLU()

        self.heads = nn.ModuleList([nn.Linear(50, hparams.num_classes) for _ in range(hparams.num_tasks)])

    def set_task(self, task):
        self.task = task

    def forward(self, input, task=None):
        if task is None:
            task = self.task

        input = F.max_pool2d(F.relu(self.conv1(input)), (2, 2))
        input = F.max_pool2d(F.relu(self.conv2(input)), 2)
        input = self.drop1(input)

        input = input.view(input.size(0), -1)
        input = F.relu(self.linear3(input))
        input = self.drop2(input)

        input = F.relu(self.linear4(input))

        head = self.heads[task]
        input = head(input)

        return input


if __name__ == "__main__":
    model = Vanilla_cnn()
    print(model)
    batch_size, C, H, W = 64, 1, 64, 64
    y = torch.randn(batch_size, C, H, W)
    output = model(y)
