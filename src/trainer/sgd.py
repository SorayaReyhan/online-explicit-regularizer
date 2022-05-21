import torch
import torch.nn as nn
import tqdm
from torch.optim import SGD
from torch.utils.data import DataLoader

from src.trainer.test import test_model


def normal_train(
    net: nn.Module, optimizer: torch.optim, criterion: nn.Module, dataloader: DataLoader, device: torch.device,
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


def standard_process(hparams, net, criterion, train_dataloaders, test_dataloaders, device):
    net = net.to(device)
    optimizer = SGD(params=net.parameters(), lr=hparams.lr)

    loss, acc = {}, {}
    num_tasks = len(train_dataloaders)

    for task in range(num_tasks):

        loss[task] = []
        acc[task] = []
        if task==1 : 
            # freezing the first convolutional layer after the traing and testing of the first task
            index = 0
            for name, child in net.named_children():
                if (isinstance(child, nn.Conv2d) and name=='conv1'):
                    print(child)
                    for param in child.parameters():
                        param.requires_grad = False
                        print(param.requires_grad)
                    index+=1

            # freezing the first two convolutional layers after the traing and testing of the first task
            index = 0
            for child in net.modules():
                if (isinstance(child, nn.Conv2d)):
                    for param in child.parameters():
                        param.requires_grad = False
                    print(param.requires_grad)
                    index+=1

            # train model on the current task for n epochs
            for _ in tqdm.tqdm(range(hparams.epochs)):

                trainloader = train_dataloaders[task]
                net.set_task(task)
                train_loss = normal_train(net, optimizer, criterion, trainloader, device)
                loss[task].append(train_loss)

                ## test model on current and previous tasks
                for sub_task in range(task + 1):
                    testloader = test_dataloaders[sub_task]
                    net.set_task(sub_task)
                    test_acc = test_model(net, testloader, device)
                    acc[sub_task].append(test_acc)
            print(net.conv2.weight)
        else:
            # train model on the current task for n epochs
            for _ in tqdm.tqdm(range(hparams.epochs)):

                trainloader = train_dataloaders[task]
                net.set_task(task)
                train_loss = normal_train(net, optimizer, criterion, trainloader, device)
                loss[task].append(train_loss)

                ## test model on current and previous tasks
                for sub_task in range(task + 1):
                    testloader = test_dataloaders[sub_task]
                    net.set_task(sub_task)
                    test_acc = test_model(net, testloader, device)
                    acc[sub_task].append(test_acc)
            print(net.conv2.weight)

        # train model on the current task for n epochs
        #for _ in tqdm.tqdm(range(hparams.epochs)):

            #trainloader = train_dataloaders[task]
            #net.set_task(task)
            #train_loss = normal_train(net, optimizer, criterion, trainloader, device)
            #loss[task].append(train_loss)

            ## test model on current and previous tasks
            #for sub_task in range(task + 1):
                #testloader = test_dataloaders[sub_task]
                #net.set_task(sub_task)
                #test_acc = test_model(net, testloader, device)
                #acc[sub_task].append(test_acc)

    return loss, acc
