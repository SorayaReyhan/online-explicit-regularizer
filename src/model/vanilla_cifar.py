import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.multihead_classifier_base import MultiHeadClassifierBase


class VanillaCIFAR(MultiHeadClassifierBase):
    def __init__(self, hparams):
        super().__init__()
        self.dropout = nn.Dropout(hparams.dropout)

        # 3, 32, 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(2)  # 32, 16, 16

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(2)  # 64, 8, 8

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.pool3 = nn.MaxPool2d(2)  # 128, 4, 4

        self.linear4 = nn.Linear(128 * 4 * 4, 128)

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

        x = x.view(x.size(0), -1)

        x = self.linear4(x)
        x = F.relu(x)

        head = self.heads[task]
        x = head(x)

        return x


if __name__ == "__main__":
    model = VanillaCIFAR()
    print(model)
    batch_size, C, H, W = 10, 3, 32, 32
    y = torch.randn(batch_size, C, H, W)
    output = model(y)
