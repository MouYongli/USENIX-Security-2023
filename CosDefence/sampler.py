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

batch_size = 8
train_size = 0.75
least_samples = batch_size / (1 - train_size)


def separate_data(data, num_clients, num_classes, niid=True, real=True, partition=None, balance=False,
                  class_per_client=2):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]
    dataset_content, dataset_label = data
    if partition is None or partition == "noise":
        dataset = []
        for i in range(num_classes):
            idx = dataset_label == i
            print(idx)
            dataset.append(dataset_content[idx])
        if not niid or real:
            class_per_client = num_classes
        class_num_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_client[client] > 0:
                    selected_clients.append(client)
            if niid and not real:
                selected_clients = selected_clients[:int(num_clients / num_classes * class_per_client)]
            num_all = len(dataset[i])
            num_clients_ = len(selected_clients)
            if niid and real:
                num_clients_ = np.random.randint(1, len(selected_clients))
            num_per = num_all / num_clients_
            if balance:
                num_samples = [int(num_per) for _ in range(num_clients_ - 1)]
            else:
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
                                                num_clients_ - 1).tolist()
            num_samples.append(num_all - sum(num_samples))

            if niid:
                # each client is not sure to have all the labels
                selected_clients = list(np.random.choice(selected_clients, num_clients_, replace=False))
            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if len(X[client]) == 0:
                    X[client] = dataset[i][idx:idx + num_sample]
                    y[client] = i * np.ones(num_sample)
                else:
                    X[client] = np.append(X[client], dataset[i][idx:idx + num_sample], axis=0)
                    y[client] = np.append(y[client], i * np.ones(num_sample), axis=0)
                idx += num_sample
                statistic[client].append((i, num_sample))
                class_num_client[client] -= 1

    elif niid and partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)
        net_dataidx_map = {}
        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
        # additional codes
        for client in range(num_clients):
            idxs = net_dataidx_map[client]
            X[client] = dataset_content[idxs]
            y[client] = dataset_label[idxs]

            for i in np.unique(y[client]):
                statistic[client].append((int(i), int(sum(y[client] == i))))
    else:
        raise EOFError
    del data
    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)
    return X, y, statistic


def load_mnist_data():
    trans_train_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trans_test_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train_ds = torchvision.datasets.MNIST('../Data/mnist/', train=True, download=True,
                                                transform=trans_train_mnist)
    mnist_test_ds = torchvision.datasets.MNIST('../Data/mnist/', train=False, download=True, transform=trans_test_mnist)
    X_train, y_train = mnist_train_ds.data, mnist_train_ds.targets
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.targets
    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()
    return (X_train, y_train, X_test, y_test)


def load_cifar10_data():
    transform_train_cifar10 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar10_train_ds = torchvision.datasets.CIFAR10('../Data/cifar', train=True, download=True,
                                                    transform=transform_train_cifar10)
    cifar10_test_ds = torchvision.datasets.CIFAR10('../Data/cifar', train=False, download=True,
                                                   transform=transform_test_cifar10)
    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.targets
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.targets
    return X_train, y_train, X_test, y_test


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts


# https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/datasets.py
def partition_data(dataset, partition, n_nets, alpha=0.5):
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data()
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data()
    n_train = X_train.shape[0]
    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}
    elif partition == "hetero-dir":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        net_dataidx_map = {}
        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


if __name__ == '__main__':
    # trans_train_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # dataset_train = torchvision.datasets.MNIST('../Data/mnist/', train=True, download=True, transform=trans_train_mnist)
    # x, y, stat = separate_data((dataset_train.data, dataset_train.targets), 10, 10)

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset='mnist', partition='homo', n_nets=11, alpha=0.5)
    print(net_dataidx_map.keys())
    print(traindata_cls_counts)