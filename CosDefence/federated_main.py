import os
import os.path as osp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tqdm
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from models import MLP, LeNet, AlexNet, ResNet

if __name__ == "__main__":
    here = osp.dirname(osp.abspath(__file__))
    parser = argparse.ArgumentParser(description="CosDefence: A Defence against Data Poisoning Attacks in Federated Learning")
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset benchmark to use', choices=["mnist", "cifar10"])
    parser.add_argument('--model', type=str, default='lenet', help='neural network model to use', choices=["mlp", "lenet", "alexnet", "resnet"])
    parser.add_argument('--distribution', type=str, default='dirichlet', help='label distribution', choices=["uniform", "dirichlet", "partial"])
    parser.add_argument('--train-batch-size', type=int, default=8, help='batch size of training')
    parser.add_argument('--val-batch-size', type=int, default=16, help='batch size of val')
    parser.add_argument('--test-batch-size', type=int, default=16, help='batch size of test')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    ## Parameters of federated learning settings
    parser.add_argument('--num-clients', type=int, default=10, help='number of clients')
    parser.add_argument('--num-attackers', type=int, default=1, help='number of malicious attackers')
    parser.add_argument("--num-communication-rounds", type=int, default=50, help="number of federated learning communication rounds")
    parser.add_argument("--num-local-epochs", type=int, default=1, help="number of local epochs")
    ## Parameters for CosDefense
    parser.add_argument("--num-trainers", type=int, default=1, help="number of trainers")
    parser.add_argument("--num-validators", type=int, default=1, help="number of validators")
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    experiment_dir = osp.join(here, 'results', 'results/federated_train', args.dataset, args.model)

    trans_train_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trans_val_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    transform_train_cifar10 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_val_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # load dataset and split clients
    if args.dataset == 'mnist':
        args.num_classes = 10
        args.dataset_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # label flipping attacks: malicious client flips class 1 "car" to class 9 "truck"
        args.flip_label_from = 1
        args.flip_label_to = 9
        if not args.model in ["mlp", "lenet"]:
            exit('Error: only models "mlp", "lenet", support for dataset {}'.format(args.dataset))
        dataset_train = torchvision.datasets.MNIST('../Data/mnist/', train=True, download=True, transform=trans_train_mnist)
        dataset_test = torchvision.datasets.MNIST('../Data/mnist/', train=False, download=True, transform=trans_val_mnist)
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False)
    elif args.dataset == 'cifar10':
        args.num_classes = 10
        args.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
        # label flipping attacks: malicious client flips class 1 "car" to class 9 "truck"
        args.flip_label_from = 1
        args.flip_label_to = 7
        if not args.model in ["alexnet", "resnet"]:
            exit('Error: only models "alexnet", "resnet", support for dataset {}'.format(args.dataset))
        dataset_train = torchvision.datasets.CIFAR10('../Data/cifar', train=True, download=True, transform=transform_train_cifar10)
        dataset_test = torchvision.datasets.CIFAR10('../Data/cifar', train=False, download=True, transform=transform_val_cifar10)

        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False)
    else:
        exit('Error: unrecognized dataset')