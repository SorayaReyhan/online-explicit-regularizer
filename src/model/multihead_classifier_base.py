from torch import nn


class MultiHeadClassifierBase(nn.Module):
    def __init__(self):
        self.task = None
        super().__init__()

    def set_task(self, task):
        self.task = task
