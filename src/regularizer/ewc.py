from typing import Dict

import torch
from torch.utils.data import DataLoader
from torch import nn
from src.regularizer.base import Regularizer, RegularizerHParams


class EWCHparams(RegularizerHParams):
    sample_size: int = 200


class EWC(Regularizer):
    def __init__(self, hparams: EWCHparams, net: nn.Module, criterion: nn.Module, device: torch.device,) -> None:
        super().__init__(hparams, net, criterion, device)

        self.importance: Dict[str, torch.Tensor] = {name: 0.0 for name, _ in net.named_parameters()}

    @staticmethod
    def _clone_model_params(net):
        return {n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad}

    def online_step(self, batch=None):
        m = self.hparams.saliency_momentum
        for name, param in self.net.named_parameters():
            if param.requires_grad and param.grad is not None:
                imp = self.importance[name]
                new_imp = m * (param.grad ** 2) + (1 - m) * imp
                self.importance[name] = new_imp

        return self.importance

    def task_end(self, dataloader: DataLoader):
        # store model's parameters
        self.params = self._clone_model_params(self.net)

        # calculate parameter importance for the current task
        curr_imp = self.calculate_parameter_importance(
            self.net, self.criterion, dataloader, self.hparams.n_batches_for_importance_estimation, self.device
        )

        if self.importance is None:
            # this was the first task, since importance has not been assigned before
            self.importance = curr_imp
        else:
            # update parameter importance with respect to all seen tasks, including this one
            for name in self.importance:
                prev_imp = self.importance[name]

                # TODO: averaging is not the only option to mix the importances.
                # Averaging assumes that the importance of current task is equal to the importance of previous tasks combined.
                self.importance[name] = (curr_imp[name] + prev_imp) / 2

            # TODO: look at normalize_saliency argument here https://github.com/EkdeepSLubana/QRforgetting/blob/00ff0228142d685f1f4a0f463c4723baf89cbb01/reg_based.py#L176
            # do we need this option?

        return self.importance

    @staticmethod
    def calculate_parameter_importance(net: nn.Module, criterion: nn.Module, dataloader: DataLoader, n_batches, device):

        importance = {name: 0.0 for name, p in net.named_parameters() if p.requires_grad}

        N = len(dataloader)

        net.eval()
        for batch_idx, (input, target) in enumerate(dataloader):
            if batch_idx == n_batches:
                break

            input, target = input.to(device), target.to(device)

            net.zero_grad()
            output = net(input)
            loss = criterion(output, target)
            loss.backward()

            for n, p in net.named_parameters():
                if p.requires_grad and p.grad is not None:
                    importance[n] += p.grad ** 2 / N

        return importance

    def penalty(self, model: nn.Module):
        loss = 0
        for name, param in model.named_parameters():
            _loss = self.importance[name] * (param - self.params[name]) ** 2
            loss += _loss.sum()
        return loss
