# %%
# init
from argparse import ArgumentParser
from dataclasses import dataclass
from pprint import pprint

import torch
from torch import nn

from src.dataset.cifar import get_cifar10, get_cifar100
from src.dataset.dummy_dataset import get_dummy_dataloaders
from src.dataset.malware import get_malware_dataloaders
from src.dataset.mnist import get_permute_mnist
from src.model.dummy_classifier import DummyClassifier
from src.model.mnist_mlp import MnistMLP, MnistMLPHParams
from src.model.vanilla import VanillaCNNMalware, Vanilla_cnn
from src.model.vanilla_cifar import VanillaCIFAR
from src.regularizer.ewc import EWC
from src.trainer.ewc import EWCTrainer, EWCTrainerHParams
from src.trainer.online_explicit import (OnlineExplicitTrainer,
                                         OnlineExplicitTrainerHParams)
from src.trainer.sgd import standard_process
from src.utils import Logger, seed_everything
from torch.utils.data import DataLoader , TensorDataset, ConcatDataset


@dataclass
class HParams(EWCTrainerHParams, MnistMLPHParams, OnlineExplicitTrainerHParams):
    epochs: int = 50
    lr: float = 0.01
    batch_size: int = 32
    num_tasks: int = 4
    num_classes: int = 5
    seed: int = 12345
    model: str = "vanilla_cnn"
    dataset: str = "malware"
    trainer: str = "sgd"
    saliency_momentum: float = 0.8
    importance: float = 50000  # ewc importance value
    dropout: float = 0.0
    name: str = "exp"  # experiment name to append to folder name
    singlehead:bool = False
    buffer_size: int = 300



parser = ArgumentParser()
parser.add_argument("--lr", type=float, default=HParams.lr)
parser.add_argument("--dropout", type=float, default=HParams.dropout)
parser.add_argument("--model", choices=["vanilla", "vanilla_cnn", "mnist", "vanilla_cifar", "dummy"])
parser.add_argument("--dataset", choices=["malware", "mnist", "cifar10", "cifar100", "dummy"])
parser.add_argument("--trainer", default=HParams.trainer, choices=["sgd", "ewc", "online_explicit_ewc"])
parser.add_argument("--name", default=HParams.name)
parser.add_argument("--epochs", type=int, default=HParams.epochs)
parser.add_argument("--num_tasks", type=int, default=HParams.num_tasks)
parser.add_argument("--num_classes", type=int, default=HParams.num_classes)
parser.add_argument("--batch_size", type=int, default=HParams.batch_size)
parser.add_argument("--seed", type=int, default=HParams.seed)
parser.add_argument("--importance", type=float, default=HParams.importance)
parser.add_argument("--saliency_momentum", type=float, default=HParams.saliency_momentum)
#parser.add_argument("--singlehead", action="store_true")
parser.add_argument("--singlehead", type=bool)
parser.add_argument("--buffer_size", type=int, default=HParams.buffer_size)

args = parser.parse_args()


# initialize hparams and override with command line arguments
hparams = HParams(**vars(args))

print("Hyperparameters:")
pprint(vars(hparams))

# to make experiments reproducible
seed_everything(args.seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


criterion = nn.CrossEntropyLoss()


#%%
# prepare data
if hparams.dataset == "mnist":
    train_dataloaders, test_dataloaders = get_permute_mnist(hparams.num_tasks, hparams.batch_size)
elif hparams.dataset == "malware":
    datasets_root = "data/random"  # set the malware datasets directory
    train_dataloaders, test_dataloaders = get_malware_dataloaders(datasets_root, hparams.num_tasks, hparams.batch_size)
elif hparams.dataset == "cifar10":
    train_dataloaders, test_dataloaders = get_cifar10(hparams.num_tasks, hparams.num_classes, hparams.batch_size)
elif hparams.dataset == "cifar100":
    train_dataloaders, test_dataloaders = get_cifar100(hparams.num_tasks, hparams.num_classes, hparams.batch_size)
elif hparams.dataset == "dummy":
    input_shape = (28 * 28,) if hparams.model == "mnist" else (1, 64, 64)
    train_dataloaders, test_dataloaders = get_dummy_dataloaders(hparams.num_tasks, hparams.batch_size, input_shape)

#%%
# train with sgd

seed_everything(args.seed)

if hparams.model == "vanilla":
    model = VanillaCNNMalware(hparams)
elif hparams.model == "vanilla_cnn":
    model = Vanilla_cnn(hparams)
elif hparams.model == "dummy":
    model = DummyClassifier(hparams)
elif hparams.model == "mnist":
    model = MnistMLP(hparams)
elif hparams.model == "vanilla_cifar":
    model = VanillaCIFAR(hparams)

criterion = nn.CrossEntropyLoss()

#%%
# train
num_tasks=5
buffer_size=50
train_dataloaders_0_1, train_dataloaders_0_2, train_dataloaders_0_3, train_dataloaders_0_4 =[], [], [], []
train_dataloaders_1_2, train_dataloaders_1_3, train_dataloaders_1_4=[], [], []
train_dataloaders_2_3, train_dataloaders_2_4=[], []

# task1 
#iterator1 = iter(train_dataloaders[0])
for i in range(buffer_size):
    #images1, labels1 = next(iter(train_dataloaders[0]))
    #dataset1 = TensorDataset(images1 , labels1)
    samples1 = next(iter(train_dataloaders[0]))
    dataset1=TensorDataset(samples1)
    train_dataloaders_0_1.append(DataLoader(dataset1 , batch_size = 32, shuffle=True))
#train_dataloaders_0_1=DataLoader(dataset1 , batch_size = 32, shuffle=True)
# task2 
#iterator2 = iter(train_dataloaders[0])
for i in range(int(buffer_size/2)):
    #images2, labels2 = next(iter(train_dataloaders[0]))
    #dataset2 = TensorDataset(images2 , labels2)
    samples2 = next(iter(train_dataloaders[0]))
    dataset2 = TensorDataset(samples2)
train_dataloaders_0_2=DataLoader(dataset2 , batch_size = 32, shuffle=True)

#iterator3 = iter(train_dataloaders[1])
for i in range(int(buffer_size/2)):
    #images3, labels3 = next(iter(train_dataloaders[1]))
    #dataset3 = TensorDataset(images3 , labels3)
    samples3 = next(iter(train_dataloaders[1]))
    dataset3 = TensorDataset(samples3)
train_dataloaders_1_2=DataLoader(dataset3 , batch_size = 32, shuffle=True)
# task3
#iterator4 = iter(train_dataloaders[0])
for i in range(int(buffer_size/3)):
    #images4, labels4 = next(iter(train_dataloaders[0]))
    #dataset4 = TensorDataset(images4 , labels4)
    samples4 = next(iter(train_dataloaders[0]))
    dataset4 = TensorDataset(samples4)
train_dataloaders_0_3=DataLoader(dataset4 , batch_size = 32, shuffle=True)

#iterator5 = iter(train_dataloaders[1])
for i in range(int(buffer_size/3)):
    #images5, labels5 = next(iter(train_dataloaders[1]))
    #dataset5 = TensorDataset(images5 , labels5)
    samples5 = next(iter(train_dataloaders[1]))
    dataset5 = TensorDataset(samples5)
train_dataloaders_1_3=DataLoader(dataset5 , batch_size = 32, shuffle=True)

#iterator6 = iter(train_dataloaders[2])
for i in range(int(buffer_size/3)):
    #images6, labels6 = next(iter(train_dataloaders[2]))
    #dataset6 = TensorDataset(images6 , labels6)
    samples6 = next(iter(train_dataloaders[2]))
    dataset6 = TensorDataset(samples6)
train_dataloaders_2_3=DataLoader(dataset6 , batch_size = 32, shuffle=True)
# task4
#iterator7 = iter(train_dataloaders[0])
for i in range(int(buffer_size/4)):
    #images7, labels7 = next(iter(train_dataloaders[0]))
    #dataset7 = TensorDataset(images7 , labels7)
    samples7 = next(iter(train_dataloaders[0]))
    dataset7 = TensorDataset(samples7)
train_dataloaders_0_4=DataLoader(dataset7 , batch_size = 32, shuffle=True)

#iterator8 = iter(train_dataloaders[1])
for i in range(int(buffer_size/4)):
    #images8, labels8 = next(iter(train_dataloaders[1]))
    #dataset8 = TensorDataset(images8 , labels8)
    samples8 = next(iter(train_dataloaders[1]))
    dataset8 = TensorDataset(samples8)
train_dataloaders_1_4=DataLoader(dataset8 , batch_size = 32, shuffle=True)

#iterator9 = iter(train_dataloaders[2])
for i in range(int(buffer_size/4)):
    #images9, labels9 = next(iter(train_dataloaders[2]))
    #dataset9 = TensorDataset(images9 , labels9)
    samples9 = next(iter(train_dataloaders[2]))
    dataset9 = TensorDataset(samples9)
train_dataloaders_2_4=DataLoader(dataset9 , batch_size = 32, shuffle=True)

#iterator10 = iter(train_dataloaders[3])
for i in range(int(buffer_size/4)):
    #images10, labels10 = next(iter(train_dataloaders[3]))
    #dataset10= TensorDataset(images10 , labels10)
    samples10 = next(iter(train_dataloaders[3]))
    dataset10 = TensorDataset(samples10)
train_dataloaders_3_4=DataLoader(dataset10 , batch_size = 32, shuffle=True)


for task in range(num_tasks):
    if task==1:
        train_dev_sets = torch.utils.data.ConcatDataset([train_dataloaders[task], train_dataloaders_0_1])
        train_dev_loader = DataLoader(dataset=train_dev_sets)
        train_dataloaders[task] = train_dev_loader
        
    elif task==2:
        train_dev_sets = torch.utils.data.ConcatDataset([train_dataloaders[task], train_dataloaders_0_2])
        train_dev_sets = torch.utils.data.ConcatDataset([train_dev_sets, train_dataloaders_1_2])

        train_dev_loader = DataLoader(dataset=train_dev_sets)
        train_dataloaders[task] = train_dev_loader

    elif task==3:
        train_dev_sets = torch.utils.data.ConcatDataset([train_dataloaders[task], train_dataloaders_0_3])
        train_dev_sets = torch.utils.data.ConcatDataset([train_dev_sets, train_dataloaders_1_3])
        train_dev_sets = torch.utils.data.ConcatDataset([train_dev_sets, train_dataloaders_2_3])

        train_dev_loader = DataLoader(dataset=train_dev_sets)
        train_dataloaders[task] = train_dev_loader

    elif task==4:
        train_dev_sets = torch.utils.data.ConcatDataset([train_dataloaders[task], train_dataloaders_0_4])
        train_dev_sets = torch.utils.data.ConcatDataset([train_dev_sets, train_dataloaders_1_4])
        train_dev_sets = torch.utils.data.ConcatDataset([train_dev_sets, train_dataloaders_2_4])
        train_dev_sets = torch.utils.data.ConcatDataset([train_dev_sets, train_dataloaders_3_4])

        train_dev_loader = DataLoader(dataset=train_dev_sets)
        train_dataloaders[task] = train_dev_loader

train_dev_sets = torch.utils.data.ConcatDataset([train_dataloaders[0], train_dataloaders[1]])
train_dev_sets = torch.utils.data.ConcatDataset([train_dev_sets, train_dataloaders[2]])
train_dev_sets = torch.utils.data.ConcatDataset([train_dev_sets, train_dataloaders[3]])
train_dev_sets = torch.utils.data.ConcatDataset([train_dev_sets, train_dataloaders[4]])
train_dev_loader = DataLoader(dataset=train_dev_sets)
train_dataloaders = train_dev_loader


logger = Logger(hparams)

if hparams.trainer == "sgd":
    loss, acc = standard_process(hparams, model, criterion, train_dataloaders, test_dataloaders, DEVICE)
    logger.log_experiment_results(loss, acc, name="sgd")

elif hparams.trainer == "ewc":

    ewc_trainer = EWCTrainer(hparams, model, criterion, train_dataloaders, test_dataloaders, DEVICE)
    loss_ewc, acc_ewc = ewc_trainer.run()
    logger.log_experiment_results(loss_ewc, acc_ewc, name="ewc")

elif hparams.trainer == "online_explicit_ewc":

    regularizer = EWC(hparams, model, criterion, DEVICE)
    ewc_trainer = OnlineExplicitTrainer(
        hparams, model, criterion, regularizer, train_dataloaders, test_dataloaders, DEVICE
    )
    loss_explicit_ewc, acc_explicit_ewc= ewc_trainer.run()
    logger.log_experiment_results(loss_explicit_ewc, acc_explicit_ewc, name="explicit ewc")


