import torch
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    def __init__(self, shape, target, size=1000):
        super().__init__()

        # all samples are a clones of these 10 samples
        shape = (10, *shape)
        self.samples = torch.rand(shape)
        self.target = target

        # num samples
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index):
        index = index % 10
        sample = self.samples[index]

        # apply random perturbation
        sample = sample + torch.rand_like(sample) * 0.01

        return sample, self.target

    def get_sample(self, sample_size):
        return [self[i][0] for i in range(sample_size)]


def get_dummy_dataloaders(num_tasks, batch_size, input_shape=(32, 32)):
    train_dataloaders = {}
    test_dataloaders = {}

    for i in range(num_tasks):

        # the target (i.e. label) set equal to task id
        dataset = DummyDataset(input_shape, target=i)

        # same train and test dataset
        train_dataloaders[i] = DataLoader(dataset, batch_size=batch_size)
        test_dataloaders[i] = DataLoader(dataset, batch_size=batch_size)

    return train_dataloaders, test_dataloaders
