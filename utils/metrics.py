
"""
metrics.py - Utility functions for evaluating models in Federated Learning.
"""

import torch
import logging
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader
from typing import Tuple, List, Optional
import numpy as np
from typing import Sequence, Dict
import numpy as np
from typing import Any, Sequence, Dict, Union, List



def evaluate_per_class(
        model: torch.nn.Module,
        test_loader: DataLoader,
        num_classes: int = 10,
        class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate per-class accuracy of a model on the test set.

    Returns:
        Dict with 'overall_accuracy', 'per_class' (list of dicts with
        class_id, class_name, correct, total, accuracy), and 'num_classes'.
    """
    device = next(model.parameters()).device
    model.eval()

    correct_per_class = torch.zeros(num_classes, dtype=torch.long)
    total_per_class = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for c in range(num_classes):
                mask = labels == c
                total_per_class[c] += mask.sum().item()
                correct_per_class[c] += (preds[mask] == c).sum().item()

    per_class = []
    for c in range(num_classes):
        t = total_per_class[c].item()
        cr = correct_per_class[c].item()
        acc = 100.0 * cr / t if t > 0 else 0.0
        name = class_names[c] if class_names and c < len(class_names) else str(c)
        per_class.append({"class_id": c, "class_name": name, "correct": cr, "total": t, "accuracy": acc})

    overall = 100.0 * correct_per_class.sum().item() / max(total_per_class.sum().item(), 1)
    return {"overall_accuracy": overall, "per_class": per_class, "num_classes": num_classes}


def evaluate_backdoor(
        model: torch.nn.Module,
        test_loader: DataLoader,
        target_label: int,
        trigger_function,  # Function to apply the backdoor trigger on a batch of inputs
        device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate the backdoor accuracy of a model.

    Only evaluate on samples that originally do not have the target label.
    For these samples, apply the trigger and then measure the percentage that are classified as the target label.

    Args:
        model (torch.nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        target_label (int): The backdoor target label.
        trigger_function (callable): A function that applies the backdoor trigger to a batch of inputs.
        device (torch.device): Device on which evaluation is performed.

    Returns:
        Tuple[float, float]: (backdoor accuracy percentage, average loss)
    """
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Filter out samples that already have the target label.
            mask = (labels != target_label)
            if mask.sum() == 0:
                continue  # Skip this batch if all samples are target
            filtered_inputs = inputs[mask]

            # Apply the trigger to the filtered inputs
            triggered_inputs = trigger_function(filtered_inputs)
            triggered_inputs = triggered_inputs.to(device)

            # Create a target tensor (all samples should be classified as target)
            target_tensor = torch.full((triggered_inputs.size(0),), target_label, device=device, dtype=torch.long)

            outputs = model(triggered_inputs)
            loss = criterion(outputs, target_tensor)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == target_tensor).sum().item()
            total += triggered_inputs.size(0)

    backdoor_accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else float('inf')
    return backdoor_accuracy, avg_loss


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
    """
    Evaluate a model on the test dataset and return the accuracy and average loss.

    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        Tuple[float, float]: Accuracy (in percentage) and average loss.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    # Get the device from the model's parameters
    device = next(model.parameters()).device

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Compute predictions
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else float('inf')
    return accuracy, avg_loss

def log_evaluation_results(client_id: Optional[int], accuracy: float, loss: float, is_global: bool = False) -> None:
    """
    Log the evaluation results for a client or the global model.

    Args:
        client_id (Optional[int]): The identifier for the client (None for global evaluation).
        accuracy (float): The accuracy achieved.
        loss (float): The loss value.
        is_global (bool, optional): Flag indicating if the evaluation is for the global model. Defaults to False.
    """
    if is_global:
        logging.info(f"Global Model Evaluation - Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
    else:
        logging.info(f"Client {client_id} Evaluation - Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")




def _to_int_set(xs):
    """
    Convert input to a set of integers, handling:
    - None
    - Lists of strings ['1', '2', '3']
    - Lists of integers [1, 2, 3]
    - Dictionaries (FoolsGold weights)
    """
    if xs is None:
        return set()

    # Handle dictionary input (FoolsGold weights with threshold)
    if isinstance(xs, dict):
        # This shouldn't happen here - dict should be filtered before calling
        raise ValueError("Dictionary input should be filtered to list before calling _to_int_set")

    try:
        return {int(x) for x in xs}
    except (TypeError, ValueError):
        return set(xs)


def filter_foolsgold_weights(weights_dict: Dict[str, float], threshold: float = 0.5) -> List[int]:
    """
    Convert FoolsGold weight dictionary to list of trusted client IDs.
    Clients with weight >= threshold are considered trusted.

    Args:
        weights_dict: Dictionary mapping client_id (str) -> weight (float)
        threshold: Minimum weight to be considered trusted (default: 0.5)

    Returns:
        List of trusted client IDs as integers
    """
    trusted = []
    for client_id, weight in weights_dict.items():
        if weight >= threshold:
            trusted.append(int(client_id))
    return sorted(trusted)


def normalize_trusted_clients(
        trusted_input: Union[List, Dict],
        method_name: str = "unknown",
        foolsgold_threshold: float = 0.5
) -> List[int]:
    """
    Normalize trusted clients from any defense format to list of integers.

    Args:
        trusted_input: Can be:
            - List of integers [0, 1, 2]
            - List of strings ['0', '1', '2']
            - Dictionary of weights {'0': 0.8, '1': 0.3} (FoolsGold)
        method_name: Name of defense method (for logging)
        foolsgold_threshold: Threshold for FoolsGold weights

    Returns:
        List of trusted client IDs as integers
    """
    if isinstance(trusted_input, dict):
        # FoolsGold case
        trusted = filter_foolsgold_weights(trusted_input, foolsgold_threshold)
        print(f"  [Metrics] {method_name}: Filtered {len(trusted)}/{len(trusted_input)} "
              f"clients with weight >= {foolsgold_threshold}")
        return trusted

    # List case (FedSurrogate, FedGrad, FLAME)
    return sorted([int(x) for x in trusted_input])


def print_detection_metrics(
        trusted: Union[Sequence, Dict],
        malicious_clients: Sequence,
        total_clients: int,
        round_idx: int | None = None,
        method_name: str = "Defense",
        foolsgold_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Print detection metrics for one round, return dict of computed numbers.
    Also write the same metrics to the log file via logging.info().
    """
    # Normalize input format
    if isinstance(trusted, dict):
        trusted_list = normalize_trusted_clients(trusted, method_name, foolsgold_threshold)
    else:
        trusted_list = list(trusted)

    trusted_set = _to_int_set(trusted_list)
    mal_set = _to_int_set(malicious_clients)
    all_clients = set(range(int(total_clients)))
    flagged = all_clients - trusted_set
    ben_set = all_clients - mal_set

    TP = len(flagged & mal_set)
    FP = len(flagged & ben_set)
    FN = len(trusted_set & mal_set)
    TN = len(trusted_set & ben_set)

    TPR = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    FNR = FN / (TP + FN) if (TP + FN) > 0 else float("nan")
    FPR = FP / (FP + TN) if (FP + TN) > 0 else float("nan")
    TNR = TN / (FP + TN) if (FP + TN) > 0 else float("nan")

    precision = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
    f1 = (2 * precision * TPR / (precision + TPR)) if (precision + TPR) > 0 else float("nan")

    if round_idx is not None:
        header = f"📊 {method_name} Detection Metrics — Round {round_idx + 1}"
    else:
        header = f"📊 {method_name} Detection Metrics"

    msg = (
        f"\n{header}\n"
        f"  Trusted: {sorted(trusted_set)} ({len(trusted_set)}/{total_clients})\n"
        f"  Flagged: {sorted(flagged)} ({len(flagged)}/{total_clients})\n"
        f"  TP={TP}, FP={FP}, FN={FN}, TN={TN}\n"
        f"  TPR (Recall): {TPR:.3f} | FNR: {FNR:.3f} | FPR: {FPR:.3f} | TNR: {TNR:.3f}\n"
        f"  Precision: {precision:.3f} | F1: {f1:.3f}"
    )

    print(msg, flush=True)
    logging.info(msg)

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "TPR": TPR, "FNR": FNR, "FPR": FPR, "TNR": TNR,
        "precision": precision, "f1": f1,
        "trusted_count": len(trusted_set),
        "flagged_count": len(flagged)
    }


def detection_summary_over_rounds(
        all_trusted_lists: List[Union[Sequence, Dict]],
        malicious_clients: Sequence,
        total_clients: int,
        method_name: str = "Defense",
        foolsgold_threshold: float = 0.5,
        verbose: bool = True
) -> Dict[str, float]:
    """
    Aggregate per-round detection results into an overall report.
    Treats each (round, client) decision as an independent detection event.

    Args:
        all_trusted_lists: List of trusted clients per round (can be lists or dicts)
        malicious_clients: Ground truth malicious IDs [1, 8, 9, 19]
        total_clients: Total number of clients (20)
        method_name: Name of defense method
        foolsgold_threshold: Threshold for FoolsGold weights
        verbose: Print summary if True
    """
    mal_set = _to_int_set(malicious_clients)
    agg_TP = agg_FP = agg_FN = agg_TN = 0
    per_round_stats = []

    for round_idx, trusted in enumerate(all_trusted_lists):
        # Normalize input format
        if isinstance(trusted, dict):
            trusted_list = normalize_trusted_clients(
                trusted, f"{method_name} R{round_idx + 1}", foolsgold_threshold
            )
        else:
            trusted_list = list(trusted)

        trusted_set = _to_int_set(trusted_list)
        all_clients = set(range(int(total_clients)))
        flagged = all_clients - trusted_set
        ben_set = all_clients - mal_set

        TP = len(flagged & mal_set)
        FP = len(flagged & ben_set)
        FN = len(trusted_set & mal_set)
        TN = len(trusted_set & ben_set)

        agg_TP += TP
        agg_FP += FP
        agg_FN += FN
        agg_TN += TN

        # Calculate per-round metrics with safe division
        round_TPR = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
        round_FPR = FP / (FP + TN) if (FP + TN) > 0 else float("nan")
        round_precision = TP / (TP + FP) if (TP + FP) > 0 else float("nan")

        # Safe F1 calculation: F1 = 2*TP / (2*TP + FP + FN)
        if (TP + FP) > 0 and (TP + FN) > 0:
            round_f1 = 2 * TP / (2 * TP + FP + FN)
        else:
            round_f1 = float("nan")

        per_round_stats.append({
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "TPR": round_TPR,
            "FPR": round_FPR,
            "precision": round_precision,
            "f1": round_f1
        })

    # Overall metrics
    overall_TPR = agg_TP / (agg_TP + agg_FN) if (agg_TP + agg_FN) > 0 else float("nan")
    overall_FPR = agg_FP / (agg_FP + agg_TN) if (agg_FP + agg_TN) > 0 else float("nan")
    overall_precision = agg_TP / (agg_TP + agg_FP) if (agg_TP + agg_FP) > 0 else float("nan")
    overall_f1 = (2 * overall_precision * overall_TPR / (overall_precision + overall_TPR)
                  if (overall_precision + overall_TPR) > 0 else float("nan"))

    mean_per_round = lambda k: float(np.nanmean([r[k] for r in per_round_stats if not np.isnan(r[k])])) \
        if per_round_stats else float("nan")

    summary = {
        "method": method_name,
        "rounds": len(all_trusted_lists),
        "agg_TP": agg_TP, "agg_FP": agg_FP, "agg_FN": agg_FN, "agg_TN": agg_TN,
        "overall_TPR": overall_TPR, "overall_FPR": overall_FPR,
        "overall_precision": overall_precision, "overall_f1": overall_f1,
        "mean_TPR_per_round": mean_per_round("TPR"),
        "mean_FPR_per_round": mean_per_round("FPR"),
        "mean_precision_per_round": mean_per_round("precision"),
        "mean_f1_per_round": mean_per_round("f1")
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"=== {method_name} Detection Summary Across All Rounds ===")
        print(f"{'=' * 60}")
        print(f" Rounds analyzed: {summary['rounds']}")
        print(f" Ground truth: {len(mal_set)} malicious clients = {sorted(mal_set)}")
        print(f" Aggregate counts: TP={agg_TP}, FP={agg_FP}, FN={agg_FN}, TN={agg_TN}")
        print(f" Overall TPR (Recall): {overall_TPR:.3f} | Overall FPR: {overall_FPR:.3f}")
        print(f" Overall Precision: {overall_precision:.3f} | Overall F1: {overall_f1:.3f}")
        print(f" Mean per-round TPR: {summary['mean_TPR_per_round']:.3f} | "
              f"Mean per-round FPR: {summary['mean_FPR_per_round']:.3f}")
        print(f"{'=' * 60}\n")

    return summary


