"""
aggregator.py - Implements aggregation strategies for Federated Learning.
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import torch
from torch import Tensor


class Aggregator(ABC):
    """
    Abstract base class for aggregation strategies.
    """

    @abstractmethod
    def aggregate(self, client_updates: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """
        Aggregate model updates from clients.

        Args:
            client_updates (List[Dict[str, Tensor]]): List of state dictionaries from clients.

        Returns:
            Dict[str, Tensor]: Aggregated model update.
        """
        pass


class FedAvgAggregator(Aggregator):
    def aggregate(self, client_updates: List[Dict[str, torch.Tensor]], weights: List[float] = None) -> Dict[str, torch.Tensor]:
        num_clients = len(client_updates)
        avg_update: Dict[str, torch.Tensor] = {}
        for key in client_updates[0]:
            # Ignore weights, do a simple average
            avg_update[key] = sum(update[key] for update in client_updates) / num_clients
        return avg_update

    
    """
    
    """


    @staticmethod
    def weighted_average(models, weights):
        """
        Compute weighted average of model parameters.
        Handles both state_dicts and nn.Module objects.

        Args:
            models: List of state_dicts OR nn.Module objects
            weights: List of aggregation weights

        Returns:
            Aggregated state_dict
        """
        if not models:
            raise ValueError("No models provided for aggregation")

        # Convert nn.Module objects to state_dicts if needed
        if isinstance(models[0], torch.nn.Module):
            models = [model.state_dict() for model in models]

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Total weight is zero")

        normalized_weights = [w / total_weight for w in weights]

        # Initialize new state
        new_state = {}
        first_model = models[0]

        # Parameters to skip aggregation (BatchNorm counters)
        skip_keys = {'num_batches_tracked'}

        for k in first_model.keys():
            # Check if this parameter should be skipped
            if any(skip_key in k for skip_key in skip_keys):
                # Just copy from first model
                new_state[k] = first_model[k].clone()
                continue

            # Initialize with first model's weighted contribution
            new_state[k] = first_model[k].float() * normalized_weights[0]

            # Accumulate remaining models in-place
            for model_dict, w in zip(models[1:], normalized_weights[1:]):
                new_state[k].add_(model_dict[k].float(), alpha=w)

            # Preserve original dtype
            new_state[k] = new_state[k].to(first_model[k].dtype)

        return new_state

class WeightedFedAvgAggregator(Aggregator):
    """
    Implements a weighted version of Federated Averaging.
    """

    def aggregate(self, client_updates: List[Dict[str, Tensor]], weights: List[float]) -> Dict[str, Tensor]:
        """
        Compute a weighted average of the client updates.

        Args:
            client_updates (List[Dict[str, Tensor]]): List of model parameter dictionaries.
            weights (List[float]): A list of weights corresponding to each client update.

        Returns:
            Dict[str, Tensor]: Dictionary containing the weighted averaged model parameters.

        Raises:
            ValueError: If the number of updates does not match the number of weights or if total weight is zero.
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation.")
        if len(client_updates) != len(weights):
            raise ValueError("The number of client updates must match the number of weights.")

        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Total weight must be non-zero.")

        inv_total = 1.0 / total_weight
        avg_update: Dict[str, Tensor] = {}
        for key in client_updates[0]:
            avg_update[key] = client_updates[0][key] * (weights[0] * inv_total)
            for update, weight in zip(client_updates[1:], weights[1:]):
                avg_update[key].add_(update[key], alpha=weight * inv_total)
        return avg_update


class FedProxAggregator(Aggregator):
    """
    Implements FedProx aggregation.

    FedProx incorporates a proximal term in the local training objective to mitigate client drift.
    In this aggregator, we assume that each client's update is computed relative to the global model,
    and the aggregated update is calculated as the global model plus the average of the deviations.
    """

    def __init__(self, global_model_state: Dict[str, Tensor]) -> None:
        """
        Initialize the FedProx aggregator with the current global model state.

        Args:
            global_model_state (Dict[str, Tensor]): The state dict of the global model.
        """
        self.global_model_state = global_model_state

    def aggregate(self, client_updates: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """
        Aggregate client updates using FedProx.

        For each parameter, compute:
            new_param = global_param + average(client_param - global_param)

        Args:
            client_updates (List[Dict[str, Tensor]]): List of model parameter dictionaries from clients.

        Returns:
            Dict[str, Tensor]: The new aggregated global model state.

        Raises:
            ValueError: If no client updates are provided.
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation.")

        num_clients = len(client_updates)
        aggregated_update: Dict[str, Tensor] = {}
        for key in self.global_model_state:
            diff_sum = sum(client_update[key] - self.global_model_state[key] for client_update in client_updates)
            aggregated_update[key] = self.global_model_state[key] + diff_sum / num_clients
        return aggregated_update
