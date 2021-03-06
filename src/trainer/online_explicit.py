from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.multihead_classifier_base import MultiHeadClassifierBase
from src.regularizer.base import Regularizer
from src.trainer.test import test_model
from src.trainer.utils import normalize

alp_list: List = []
param_data_list: List= []

class OnlineExplicitTrainerHParams:
    normalize_saliency = False
    epochs = 50
    num_tasks = 3
    use_AGC = False


def AGC(net: nn.Module, optimizer):
    # TODO: why is AGC only applied to Conv2d layers?
    eta = optimizer.param_groups[0]["lr"]
    lambd = optimizer.param_groups[0]["weight_decay"]
    beta = optimizer.param_groups[0]["momentum"]
    threshold = np.sqrt(2 * lambd / (eta * (1 + beta)))

    for mod in net.modules():
        if isinstance(mod, nn.Conv2d):
            g_norms = torch.norm(mod.weight.grad.data.reshape(mod.weight.shape[0], -1), dim=1)
            p_norms = torch.norm(mod.weight.data.reshape(mod.weight.shape[0], -1), dim=1)
            ratios = torch.div(g_norms, p_norms + 1e-8)
            multiplier = (ratios < threshold) * 1
            multiplier = multiplier + (1 - multiplier) * threshold * (torch.div(p_norms, g_norms + 1e-8))
            mod.weight.grad.data = mod.weight.grad.data * multiplier.view(-1, 1, 1, 1)
    return net


def explicit_step(
    net: nn.Module, net_prev: nn.Module, imp: Dict[str, torch.Tensor], prev_imp: Dict[str, torch.Tensor],
):
    global alp_list
    global param_data_list
    net_prev_params = net_prev.state_dict()

    
    for name, param in net.named_parameters():
        if param.grad is not None:
            prev_param = net_prev_params[name]
            alp_new = imp[name] ** (1 / 2)  # alpha current
            alp_prev = prev_imp[name] ** (1 / 2)  # alpha previous
            alp = alp_new / (alp_new + alp_prev + 1e-20)  # R_j
            param.data = alp * param.data + (1 - alp) * prev_param.data  # interpolation
            # calculating average R and average interpolation:
            param_data_list.append(torch.mean(param.data))
            alp_list.append(torch.mean(alp))
            
            
class OnlineExplicitTrainer:
    """Trainer class that sequentially trains on a set of tasks with EWC regularization.
    """

    def __init__(
        self,
        hparams: OnlineExplicitTrainerHParams,
        net: MultiHeadClassifierBase,
        criterion: nn.Module,
        regularizer: Regularizer,
        train_dataloaders: List[DataLoader],
        test_dataloaders: List[DataLoader],
        device: torch.device,
    ) -> None:
        self.hparams = hparams
        self.net = net.to(device)
        self.criterion = criterion
        assert isinstance(train_dataloaders, (list, dict)), f"expect list/dict, got {type(train_dataloaders)}"
        self.train_dataloaders = train_dataloaders
        self.test_dataloaders = test_dataloaders
        self.device = device
        self.regularizer = regularizer
        self.net_prev = None

        self.optimizer = self.setup_optimizer(net)

    def setup_optimizer(self, net):
        optimizer = SGD(params=net.parameters(), lr=self.hparams.lr)
        return optimizer

    def explicit_train_epoch(
        self, task: int, net_prev: nn.Module = None, prev_imp: Dict[str, torch.Tensor] = None,
    ):
        """Train model on a task with explicit interpolation regularization for one epoch"""
        net = self.net
        optimizer = self.optimizer
        criterion = self.criterion
        device = self.device
        regularizer = self.regularizer
        #assert isinstance(self.train_dataloaders, (list, dict)), f"expect list/dict, got {type(self.train_dataloaders)}"
        dataloader = self.train_dataloaders[task]

        net.set_task(task)
        net.train()
        epoch_loss = 0
        
        # i = 0...#iter / epoch
        # #iter = size(dataset) / batch_size * epoch
        for sample in dataloader:  #  loop size(dataset) / batchsize times
            # print("inputs", inputs)
            # print("targets",targets)
            inputs, targets = sample[0], sample[1]
            inputs, targets = inputs.to(device), targets.to(device)

            ### SGD step on current minibatch
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()
            loss.backward()

            if self.hparams.use_AGC:
                net = AGC(net, optimizer)

            optimizer.step()  # task specific gradient update

            # update current task importance
            new_imp = regularizer.online_step()  # update alpha (Tn)

            ### explicit step

            if task > 0:
                if self.hparams.normalize_saliency:
                    new_imp = normalize(new_imp)

                explicit_step(net, net_prev, new_imp, prev_imp)  # explicit interpolation

        return epoch_loss / len(dataloader)

    def explicit_train(self, task: int, loss, acc):
        """Train model on a task with explicit interpolation regularization."""
        global alp_list
        alp_list = []
        global param_data_list
        param_data_list = []
        # n = task

        net = self.net

        # theta_(n-1)
        # store current network
        net_prev = deepcopy(net)

        # alpha_(n-1)
        # get parameter importance for previous tasks
        prev_imp = self.regularizer.get_parameter_importance()

        # train on the current task for n epochs
        for _ in tqdm(range(self.hparams.epochs)):
            net.set_task(task)
            train_loss = self.explicit_train_epoch(task, net_prev, prev_imp)
            #train_alp = self.explicit_train_epoch(task, net_prev, prev_imp)
            loss[task].append(train_loss)
            #alp[task].append(train_alp)

            # test model on current and previous tasks
            for sub_task in range(task + 1):
                net.set_task(sub_task)
                testloader = self.test_dataloaders[sub_task]
                test_acc = test_model(net, testloader, self.device)
                acc[sub_task].append(test_acc)
        # calculating average R and average interpolation:
        if len(alp_list) > 0:
            alp_mean_by_task = torch.mean(torch.stack(alp_list), dim=0)
            print(alp_mean_by_task)
        
        if len(param_data_list) > 0:
            param_data_mean_by_task = torch.mean(torch.stack(param_data_list), dim=0)
            print(param_data_mean_by_task)

    def run(self):
        """Run continual learning. Train on tasks sequentially.
        
        Returns train losses and test accuracies for all tasks at each epoch.
        """

        loss, acc , alp= {}, {}, {}
        net = self.net
        for task in range(self.hparams.num_tasks):
            loss[task] = []
            acc[task] = []
            #alp[task]=[]
            if task==1 : 
                #freezing the first convolutional layer after the traing and testing of the first task
                #index = 0
                #for name, child in net.named_children():
                    #if (isinstance(child, nn.Conv2d) and name=='conv1'):
                        #print(child)
                        #for param in child.parameters():
                            #param.requires_grad = False
                        #print(param.requires_grad)
                        #index+=1

                # freezing the first two convolutional layers after the traing and testing of the first task
                #index = 0
                #for child in net.modules():
                    #if (isinstance(child, nn.Conv2d)):
                        #for param in child.parameters():
                            #param.requires_grad = False
                        #print(param.requires_grad)
                        #index+=1

                self.explicit_train(task, loss, acc)
                #print(net.conv2.weight)
            else:
                self.explicit_train(task, loss, acc)
                #print(net.conv2.weight)

        return loss, acc
