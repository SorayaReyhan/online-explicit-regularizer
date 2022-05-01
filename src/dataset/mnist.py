import random

import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


class PermutedMNIST(datasets.MNIST):
    def __init__(self, root="~/.torch/data/mnist", train=True, permute_idx=None):

        super(PermutedMNIST, self).__init__(root, train, download=True)

        assert len(permute_idx) == 28 * 28

        # use another variable name instead of train_data like
        self.my_data = torch.stack([(img.float().view(-1)[permute_idx] / 255) for img in self.data])

    def __getitem__(self, index):

        img, target = self.my_data[index], self.targets[index]

        return img, target

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        return [img for img in self.my_data[sample_idx]]


def get_permute_mnist(num_tasks, batch_size):
    train_dataloaders = {}
    test_dataloaders = {}

    idx = list(range(28 * 28))

    for i in range(num_tasks):
        dataset = PermutedMNIST(train=True, permute_idx=idx)
        train_dataloaders[i] = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataset = PermutedMNIST(train=False, permute_idx=idx)
        test_dataloaders[i] = DataLoader(dataset, batch_size=batch_size)
        random.shuffle(idx)

    return train_dataloaders, test_dataloaders
