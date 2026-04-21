
"""
load_data.py - Utilities for loading and partitioning datasets.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from typing import Tuple, List, Dict, Any


def load_dataset(dataset_name: str):
    """
    Load the specified dataset (e.g., "mnist", "cifar10") and return train/test splits.
    (This is your existing function; adjust as needed.)
    """
    # Example for CIFAR10:
    from torchvision import datasets, transforms
    if dataset_name.lower() == "cifar10":
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

        transform_test = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
        
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    

    elif dataset_name.lower() == "cifar100":
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

        transform_test = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

        train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    
    elif dataset_name.lower() == "svhn":

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])

        train_dataset = datasets.SVHN(root="./data", split="train", download=True, transform=transform_train)
        test_dataset  = datasets.SVHN(root="./data", split="test",  download=True, transform=transform_test)


        train_dataset.labels = np.where(train_dataset.labels == 10, 0, train_dataset.labels)
        test_dataset.labels  = np.where(test_dataset.labels  == 10, 0, test_dataset.labels)

         
    elif dataset_name.lower() == "mnist":

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    
    elif dataset_name.lower() == "fmnist":

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_dataset, test_dataset


def partition_idx_labeldir(y: np.ndarray, n_parties: int, alpha: float, num_classes: int, min_require_size: int = 10):
    """
    Partition the dataset indices among n_parties using a Dirichlet distribution.

    Args:
        y (np.ndarray): Array of labels for each sample.
        n_parties (int): Number of clients.
        alpha (float): Dirichlet parameter controlling the skewness (smaller => more skewed).
        num_classes (int): Total number of classes.
        min_require_size (int): Minimum number of samples each client should have.

    Returns:
        dict: A dictionary mapping each client index to a list of indices.
    """
    N = y.shape[0]
    K = num_classes
    net_dataidx_map = {}
    min_size = 0

    # Repeatedly sample until every client has at least min_require_size samples
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            # Sample Dirichlet proportions for this class
            proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
            # Optional: Favor clients with fewer samples by zeroing out proportions
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            # Determine split points for indices of class k
            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # Split the indices accordingly
            idx_split = np.split(idx_k, split_points)
            # Append indices to each client's batch
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, idx_split)]
        min_size = min([len(idx_j) for idx_j in idx_batch])
    # Shuffle indices for each client and build the final mapping
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    return net_dataidx_map


def partition_data(train_dataset, config):
    """
    Partition the training dataset among clients according to the strategy specified in config.

    If the strategy is 'iid', data is split equally.
    If the strategy is 'dirichlet', use the Dirichlet-based partitioning.
    """
    num_clients = config["num_clients"]
    strategy = config["data_partition"]["strategy"].lower()

    if strategy == "iid":
        total = len(train_dataset)
        # Simple equal splitting (adjust if needed)
        split_sizes = [total // num_clients] * num_clients
        return random_split(train_dataset, split_sizes)

    elif strategy == "dirichlet":
        alpha = config["data_partition"]["dirichlet_alpha"]
        num_classes = config["data_partition"]["num_classes"]
        min_require_size = config["data_partition"].get("min_require_size", 10)
        # Extract labels efficiently without applying transforms
        if hasattr(train_dataset, 'targets'):
            labels = np.array(train_dataset.targets)
        elif hasattr(train_dataset, 'labels'):
            labels = np.array(train_dataset.labels)
        else:
            labels = np.array([label for _, label in train_dataset])
        net_dataidx_map = partition_idx_labeldir(y=labels,
                                                 n_parties=num_clients,
                                                 alpha=alpha,
                                                 num_classes=num_classes,
                                                 min_require_size=min_require_size)
        # Create subsets for each client
        client_datasets = []
        for i in range(num_clients):
            client_datasets.append(Subset(train_dataset, net_dataidx_map[i]))
        return client_datasets

    else:
        raise ValueError(f"Unsupported partitioning strategy: {strategy}")


def create_dataloaders(client_datasets: List[Subset], batch_size: int) -> List[DataLoader]:
    """
    Create PyTorch DataLoaders for each client's dataset.

    Args:
        client_datasets (List[Subset]): List of client-specific datasets.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        List[DataLoader]: A list of DataLoader objects, one per client.
    """
    dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=14) for dataset in client_datasets]
    return dataloaders
