import os
import os.path as osp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tqdm
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, jaccard_score, precision_score, recall_score, matthews_corrcoef, balanced_accuracy_score, roc_curve, auc

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
    parser.add_argument('--train-batch-size', type=int, default=8, help='batch size of training')
    parser.add_argument('--test-batch-size', type=int, default=16, help='batch size of test')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    experiment_dir = osp.join(here, 'results', 'centralized_train', args.dataset, args.model)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    trans_train_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trans_test_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    transform_train_cifar10 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # load dataset and split clients
    if args.dataset == 'mnist':
        args.num_classes = 10
        args.dataset_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        args.flip_label_from = 1
        args.flip_label_to = 9
        if not args.model in ["mlp", "lenet"]:
            exit('Error: models "mlp", "lenet", support for dataset {}'.format(args.dataset))
        dataset_train = torchvision.datasets.MNIST('../Data/mnist/', train=True, download=True, transform=trans_train_mnist)
        dataset_test = torchvision.datasets.MNIST('../Data/mnist/', train=False, download=True, transform=trans_test_mnist)
        # kwargs = {'num_workers': 4, 'pin_memory': True}
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True) #, **kwargs)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False) #, **kwargs)
    elif args.dataset == 'cifar10':
        args.num_classes = 10
        args.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
        if not args.model in ["alexnet", "resnet"]:
            exit('Error: models "alexnet", "resnet", support for dataset {}'.format(args.dataset))
        dataset_train = torchvision.datasets.CIFAR10('../Data/cifar', train=True, download=True, transform=transform_train_cifar10)
        dataset_test = torchvision.datasets.CIFAR10('../Data/cifar', train=False, download=True, transform=transform_test_cifar10)
        # kwargs = {'num_workers': 4, 'pin_memory': True}
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True) #, **kwargs)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False) #, **kwargs)
    else:
        exit('Error: unrecognized dataset')

    if args.model == 'mlp':
        model = MLP()
    elif args.model == 'lenet':
        model = LeNet()
    elif args.model == 'alexnet':
        model = AlexNet()
    elif args.model == 'resnet':
        model = ResNet()
    else:
        exit('Error: unrecognized model')

    if cuda:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tls_list, vls_list, cm_list, acc_list, bacc_list, prec_list, rec_list, f1_list, iou_list, mc_list = [], [], [], [], [], [], [], [], [], []
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}')
        model.train()
        train_loss = 0.0
        for i, data in tqdm.tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc='Train'):
            images, labels = data
            if cuda:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
        tls = train_loss / len(dataloader_train)
        print(f'Train loss {tls:4f}')
        tls_list.append(tls)
        model.eval()
        labels_list = []
        predictions_list = []
        test_loss = 0.0
        for i, data in tqdm.tqdm(enumerate(dataloader_test), total=len(dataloader_test), desc='Val'):
            images, labels = data
            if cuda:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            test_loss += loss.data.item()
            for label, prediction in zip(labels.data.cpu().numpy(), predictions.data.cpu().numpy()):
                labels_list.append(label)
                predictions_list.append(prediction)
        vls = test_loss / len(dataloader_test)
        cm = confusion_matrix(labels_list, predictions_list)
        acc = accuracy_score(labels_list, predictions_list)
        bacc = balanced_accuracy_score(labels_list, predictions_list)
        prec = precision_score(labels_list, predictions_list, average="macro", zero_division=0)
        rec = recall_score(labels_list, predictions_list, average="macro", zero_division=0)
        f1 = f1_score(labels_list, predictions_list, average="macro", zero_division=0)
        iou = jaccard_score(labels_list, predictions_list, average="macro", zero_division=0)
        mc = matthews_corrcoef(labels_list, predictions_list)
        print(f'Val loss {vls:4f}, Acc {acc:2f}, Balanced Acc {bacc:2f}, Precision {bacc:2f}, Recall {rec:2f}, F1-score {f1:2f}, Matthews Corrcoef {mc:2f}')
        vls_list.append(vls)
        cm_list.append(cm)
        acc_list.append(acc)
        bacc_list.append(bacc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
        iou_list.append(iou)
        mc_list.append(mc)
    data = {'train_losses': tls_list, 'test_losses': vls_list, 'confusion_matrice': cm_list, 'accuracies': acc_list,
            'balanced_accuracies': bacc_list, 'precisions': prec_list, 'recalls': rec_list, 'f1_scores': f1_list,
            'matthews_corrcoefs': mc_list}
    np.save(osp.join(experiment_dir, "result.npy"), data)
