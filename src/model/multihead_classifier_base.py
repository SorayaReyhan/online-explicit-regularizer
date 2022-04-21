from torch import nn


class MultiHeadClassifierBase(nn.Module):
    def set_task(self):
        ...
