from dataclasses import dataclass
from typing import List

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.multihead_classifier_base import MultiHeadClassifierBase
from src.regularizer.ewc import EWC, EWCHparams
from src.trainer.sgd import normal_train
from src.trainer.test import test_model


@dataclass
class EWCTrainerHParams(EWCHparams):
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 128
    num_tasks: int = 3
    importance: float = 1000


class EWCTrainer:
    """Trainer class that sequentially trains on a set of tasks with EWC regularization.
    """

    def __init__(
        self,
        hparams: EWCTrainerHParams,
        net: MultiHeadClassifierBase,
        criterion: nn.Module,
        train_dataloaders: List[DataLoader],
        test_dataloaders: List[DataLoader],
        device=torch.device,
    ) -> None:

        self.hparams = hparams
        self.net = net.to(device)
        self.train_dataloaders = train_dataloaders
        self.test_dataloaders = test_dataloaders

        self.optimizer = self.setup_optimizer(net)
        self.ewc = EWC(hparams, net, criterion, device)
        self.device = device
        self.criterion = criterion

    def setup_optimizer(self, model):
        lr = self.hparams.lr
        optimizer = SGD(params=model.parameters(), lr=lr)
        return optimizer

    def ewc_train_epoch(self, dataloader: DataLoader):
        """Train model on a task with EWC for one epoch"""
        net = self.net
        ewc = self.ewc
        optimizer = self.optimizer
        device = self.device

        net.train()
        importance = self.hparams.importance
        epoch_loss = 0
        for input, target in dataloader:
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(input)
            loss = self.criterion(output, target) + importance * ewc.penalty(net)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        return epoch_loss / len(dataloader)

    def sgd_train(self, task: int, loss, acc):
        """Train model on a task with SGD"""
        epochs = self.hparams.epochs
        net, optimizer = self.net, self.optimizer
        trainloader = self.train_dataloaders[task]
        testloader = self.test_dataloaders[task]
        net.set_task(task)

        for _ in tqdm(range(epochs)):
            # train
            train_loss = normal_train(net, optimizer, self.criterion, trainloader, self.device)
            loss[task].append(train_loss)

            # test
            test_acc = test_model(net, testloader, self.device)
            acc[task].append(test_acc)

        # calculate parameter importance
        self.ewc.task_end(trainloader)

    def ewc_train(self, task: int, loss, acc):
        """Train model on a task with EWC"""

        epochs = self.hparams.epochs

        train_dataloaders = self.train_dataloaders
        test_dataloaders = self.test_dataloaders
        net = self.net

        # current task trainloader
        trainloader = train_dataloaders[task]

        # train on the current task for n epochs
        for _ in tqdm(range(epochs)):

            net.set_task(task)
            train_loss = self.ewc_train_epoch(trainloader)
            loss[task].append(train_loss)

            # test model on current and previous tasks
            for sub_task in range(task + 1):
                testloader = test_dataloaders[sub_task]
                net.set_task(sub_task)
                test_acc = test_model(net, testloader, self.device)
                acc[sub_task].append(test_acc)

        # update parameter importance
        self.ewc.task_end(trainloader)

    def run(self):
        """Run continual learning. Train on tasks sequentially.
        
        Returns train losses and test accuracies for all tasks at each epoch.
        """

        loss, acc = {}, {}

        for task in range(self.hparams.num_tasks):
            loss[task] = []
            acc[task] = []

            # if the first task, then do the standard sgd training
            if task == 0:
                self.sgd_train(task, loss, acc)

            # for other tasks train with ewc regularization
            else:
                net = self.net
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
                    #index = 0
                    #for child in net.modules():
                        #if (isinstance(child, nn.Conv2d)):
                            #for param in child.parameters():
                                #param.requires_grad = False
                            #print(param.requires_grad)
                            #index+=1

                self.ewc_train(task, loss, acc)
                print(net.conv2.weight)
            else:
                self.ewc_train(task, loss, acc)
                print(net.conv2.weight)
                #self.ewc_train(task, loss, acc)

        return loss, acc
