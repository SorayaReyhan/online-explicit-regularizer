import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from src.trainer.test import test_model
from torch.optim import SGD


def normal_train_new(
    net: nn.Module, criterion: nn.Module, dataloader: DataLoader, device: torch.device,
):
    net.train()
    epoch_loss = 0
    for input, target in dataloader:
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(dataloader)


def standard_process_new(hparams, net, criterion, train_dataloaders, test_dataloaders, device):
    net = net.to(device)
    #optimizer = SGD(params=net.parameters(), lr=hparams.lr)

    loss, acc = {}, {}
    num_tasks = len(train_dataloaders)

    for task in range(num_tasks):

        loss[task] = []
        acc[task] = []

        # train model on the current task for n epochs
        for _ in tqdm.tqdm(range(hparams.epochs)):

            trainloader = train_dataloaders[task]
            net.set_task(task)
            train_loss = normal_train_new(net, criterion, trainloader, device)
            loss[task].append(train_loss)

            # test model on current and previous tasks
            for sub_task in range(task + 1):
                testloader = test_dataloaders[sub_task]
                net.set_task(sub_task)
                test_acc = test_model(net, testloader, device)
                acc[sub_task].append(test_acc)

    return loss, acc