from torch import nn
import torch

from src.model.multihead_classifier_base import MultiHeadClassifierBase


class DummyClassifier(MultiHeadClassifierBase):
    def __init__(self, hparams):
        super().__init__()

        # logits: [[1.0, 1.0, 1.0, ...]]
        logits = torch.tensor([1.0] * hparams.num_classes).unsqueeze(0)

        self.heads = nn.ModuleList([nn.Parameter(logits, requires_grad=True)] * hparams.num_tasks)

    def set_task(self, task):
        self.task = task

    def forward(self, input, task=None):
        if task is None:
            task = self.task

        bsize = input.size(0)
        prediction = self.heads[task]

        # predicts mu for all the inputs
        return prediction.expand(bsize, -1)
