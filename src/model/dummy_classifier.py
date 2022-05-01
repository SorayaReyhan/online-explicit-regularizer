import torch
from torch import nn

from src.model.multihead_classifier_base import MultiHeadClassifierBase


class DummyClassifier(MultiHeadClassifierBase):
    def __init__(self, hparams):
        super().__init__()

        # logits: [[1.0, 1.0, 1.0, ...]]
        logits = torch.tensor([1.0] * hparams.num_classes).unsqueeze(0)

        self.heads = nn.ParameterList([nn.Parameter(logits, requires_grad=True)] * hparams.num_tasks)

    def forward(self, input, task=None):
        if task is None:
            task = self.task

        bsize = input.size(0)
        prediction = self.heads[task]

        # predicts mu for all the inputs
        return prediction.expand(bsize, -1)
