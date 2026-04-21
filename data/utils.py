
"""
utils.py - Utility functions for data processing and analysis.
"""

from collections import Counter
from typing import List, Dict, Any

def _get_labels_fast(dataset) -> list:
    """Extract labels without applying image transforms."""
    from torch.utils.data import Subset
    if isinstance(dataset, Subset):
        src = dataset.dataset
        indices = dataset.indices
        if hasattr(src, 'targets'):
            return [src.targets[i] for i in indices]
        elif hasattr(src, 'labels'):
            return [src.labels[i] for i in indices]
    if hasattr(dataset, 'targets'):
        return list(dataset.targets)
    if hasattr(dataset, 'labels'):
        return list(dataset.labels)
    return [label for _, label in dataset]


def calculate_class_distribution(client_datasets: List[Any], num_classes: int) -> List[Dict[int, int]]:
    """
    Calculate the class distribution for each client.

    Args:
        client_datasets (List[Any]): A list of datasets (one per client). Each dataset should be iterable
                                     and yield tuples of (input, label).
        num_classes (int): Total number of classes in the dataset.

    Returns:
        List[Dict[int, int]]: A list of dictionaries. Each dictionary represents the class distribution
                              for a client, where the keys are class IDs (0 to num_classes-1) and the values
                              are the counts of samples for that class.
    """
    distributions: List[Dict[int, int]] = []
    for dataset in client_datasets:
        labels = _get_labels_fast(dataset)
        class_counts = Counter(labels)
        distribution = {class_id: class_counts.get(class_id, 0) for class_id in range(num_classes)}
        distributions.append(distribution)

    return distributions
