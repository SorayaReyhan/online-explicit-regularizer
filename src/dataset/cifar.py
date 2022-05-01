import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100

from src.dataset.utils import split_dataset

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def _get_cifar(cifar_cls, num_tasks, num_classes, batch_size):
    train_dataloaders = []
    test_dataloaders = []
    train_dataset = cifar_cls(root="data", train=True, download=True, transform=transform)
    test_dataset = cifar_cls(root="data", train=False, download=True, transform=transform)

    train_subsets = split_dataset(train_dataset, num_classes=num_classes, num_tasks=num_tasks)
    test_subsets = split_dataset(test_dataset, num_classes=num_classes, num_tasks=num_tasks)

    for train_subset, test_subset in zip(train_subsets, test_subsets):

        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        train_dataloaders.append(train_dataloader)

        test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)
        test_dataloaders.append(test_dataloader)

    return train_dataloaders, test_dataloaders


def get_cifar10(num_tasks, num_classes, batch_size):
    return _get_cifar(CIFAR10, num_tasks, num_classes, batch_size)


def get_cifar100(num_tasks, num_classes, batch_size):
    return _get_cifar(CIFAR100, num_tasks, num_classes, batch_size)
