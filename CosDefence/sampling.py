import numpy as np
import torchvision
import torchvision.transforms as transforms

'''
Reference links
# https://github.com/TsingZ0/PFL-Non-IID/blob/5c8406674d3ef4ea21fd8aca63ea530457f7bcc3/dataset/utils/dataset_utils.py
# https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/datasets.py
'''

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts

def partition_data(dataset, num_clients=10, partition="dirichlet", alpha=0.5):
    data, targets = dataset[0], dataset[1]
    num_data = len(data)
    num_classes = len(np.unique(targets))
    if partition == "uniform":
        idxs = np.random.permutation(num_data)
        batch_idxs = np.array_split(idxs, num_clients)
        client_idx_dict = {i: batch_idxs[i] for i in range(num_clients)}
    elif partition == "equal":
        client_idx_dict = {}
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.array([1.0 / num_clients for _ in range(num_clients)])
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            client_idx_dict[j] = idx_batch[j]
    elif partition == "dirichlet":
        min_size = 0
        client_idx_dict = {}
        while min_size < 10:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(num_classes):
                idx_k = np.where(targets == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < num_data / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            client_idx_dict[j] = idx_batch[j]
    elif partition == "partial":
        raise NotImplemented
    else:
        exit('Error: unrecognized distribution')

    return client_idx_dict


def get_mnist_iid(dataset, distribution="uniform", num_clients=10):
    """
    @param mode: iid mode - equal or homo
    """
    assert distribution in ["equal", "uniform"]
    data, targets = dataset.data, dataset.targets
    client_idx_dict = partition_data((data, targets), num_clients=num_clients, partition=distribution)
    client_cls_counts = record_net_data_stats(targets, client_idx_dict)
    return client_idx_dict, client_cls_counts


def get_mnist_noniid(dataset, distribution="dirichlet", num_clients=10):
    """
    @param mode: iid mode - dir or cls
    """
    assert distribution in ["dirichlet", "partial"]
    data, targets = dataset.data, dataset.targets
    client_idx_dict = partition_data((data, targets), num_clients=num_clients, partition=distribution, alpha=0.5)
    client_cls_counts = record_net_data_stats(targets, client_idx_dict)
    return client_idx_dict, client_cls_counts


def get_cifar10_iid(dataset, distribution="uniform", num_clients=100):
    """
    @param mode: iid mode - equal or uniform
    """
    assert distribution in ["equal", "uniform"]
    data, targets = dataset.data, np.array(dataset.targets)
    client_idx_dict = partition_data((data, targets), num_clients=num_clients, partition=distribution)
    client_cls_counts = record_net_data_stats(targets, client_idx_dict)
    return client_idx_dict, client_cls_counts

def get_cifar10_noniid(dataset, distribution="dirichlet", num_clients=100, alpha=0.5):
    """
    @param mode: iid mode - dir or cls
    """
    assert distribution in ["dirichlet", "partial"]
    data, targets = dataset.data, np.array(dataset.targets)
    client_idx_dict = partition_data((data, targets), num_clients=num_clients, partition=distribution, alpha=0.5)
    client_cls_counts = record_net_data_stats(targets, client_idx_dict)
    return client_idx_dict, client_cls_counts

if __name__ == '__main__':
    # transform_train_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # transform_test_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # mnist_train_ds = torchvision.datasets.MNIST('../Data/mnist/', train=True, download=True, transform=transform_train_mnist)
    # mnist_test_ds = torchvision.datasets.MNIST('../Data/mnist/', train=False, download=True, transform=transform_test_mnist)
    # # mnist_client_idx_dict, mnist_client_cls_counts = get_mnist_noniid(mnist_train_ds, mode="dir", num_clients=10)
    # mnist_client_idx_dict, mnist_client_cls_counts = get_mnist_iid(mnist_train_ds, mode="homo", num_clients=10)
    # import pprint
    # pprint.pprint(mnist_client_cls_counts)
    # counts = 0
    # for i in mnist_client_cls_counts.keys():
    #     for j in mnist_client_cls_counts[i].keys():
    #         counts = counts + mnist_client_cls_counts[i][j]

    transform_train_cifar10 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar10_train_ds = torchvision.datasets.CIFAR10('../Data/cifar', train=True, download=True, transform=transform_train_cifar10)
    cifar10_test_ds = torchvision.datasets.CIFAR10('../Data/cifar', train=False, download=True, transform=transform_test_cifar10)
    cifar10_client_idx_dict, cifar10_client_cls_counts = get_cifar10_noniid(cifar10_train_ds, mode="dir", num_clients=10)
    # cifar10_client_idx_dict, cifar10_client_cls_counts = get_cifar10_iid(cifar10_train_ds, mode="homo", num_clients=10)
    import pprint
    pprint.pprint(cifar10_client_cls_counts)