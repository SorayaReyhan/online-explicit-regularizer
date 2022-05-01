import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def test_model(net: nn.Module, dataloader: DataLoader, device: torch.device):
    net.eval()
    correct = 0
    for input, target in dataloader:
        input, target = input.to(device), target.to(device)
        output = net(input)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).sum().item()
    return correct / len(dataloader.dataset)
