"""
FLShield: A Validation Based Federated Learning Framework to Defend
Against Poisoning Attacks. Kabir et al., IEEE S&P 2024.

Date: 2026-04-10
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from torch.utils.data import DataLoader

_log = logging.getLogger(__name__)

# Per-class sample cap used by the official validation_test_fun.
_PER_CLASS_CAP = 30


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def _flatten_delta(
    client_state: Dict[str, torch.Tensor],
    global_state: Dict[str, torch.Tensor],
) -> np.ndarray:
    """Flatten (client - global) over weight/bias parameters into a 1-D vector."""
    parts: List[np.ndarray] = []
    for name, g_param in global_state.items():
        if "weight" not in name and "bias" not in name:
            continue
        if name not in client_state:
            continue
        delta = (client_state[name].detach().cpu().float()
                 - g_param.detach().cpu().float())
        parts.append(delta.reshape(-1).numpy())
    if not parts:
        raise ValueError("No weight/bias parameters found for flattening.")
    vec = np.concatenate(parts)
    if not np.isfinite(vec).all():
        raise ValueError("Non-finite values in client delta; refusing to mask.")
    return vec


def _cluster_deltas(
    deltas: np.ndarray,
    max_k: int = 15,
) -> Tuple[np.ndarray, List[List[int]], np.ndarray]:
    """
    Cluster client deltas using AgglomerativeClustering with complete linkage
    on a precomputed cosine distance matrix. Optimal k by silhouette score.

    Returns
    -------
    labels : (n,) int array of cluster labels
    clusters : list of lists of client indices per cluster
    coses : (n, n) cosine distance matrix
    """
    n = deltas.shape[0]
    if n <= 2:
        return np.zeros(n, dtype=int), [list(range(n))], np.zeros((n, n))

    coses = cosine_distances(deltas)
    np.fill_diagonal(coses, 0.0)
    coses = np.clip(coses, 0.0, 2.0)  # numerical safety

    best_k, best_score, best_labels = 2, -np.inf, None
    upper = min(n, max_k)  # exclusive upper bound, like the official `range`
    for k in range(2, upper):
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="complete",
        ).fit(coses)
        labels = clustering.labels_
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(coses, labels, metric="precomputed")
        except ValueError:
            continue
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    if best_labels is None:
        # Degenerate fall-through: everyone in one cluster
        best_labels = np.zeros(n, dtype=int)
        best_k = 1

    clusters: List[List[int]] = [[] for _ in range(best_k)]
    for i, lab in enumerate(best_labels):
        clusters[int(lab)].append(i)
    clusters = [c for c in clusters if c]  # drop any empty (shouldn't happen)

    _log.info(
        "[FLShield] k=%d (silhouette=%.3f), cluster sizes=%s",
        len(clusters), best_score if np.isfinite(best_score) else float("nan"),
        [len(c) for c in clusters],
    )
    return best_labels, clusters, coses


def _per_class_loss_capped(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
    per_class_cap: int = _PER_CLASS_CAP,
) -> np.ndarray:
    """
    Per-class mean cross-entropy on the validation set, capped at the first
    `per_class_cap` samples per class. Mirrors validation_test_fun in the
    official repo. Returns NaN for classes with zero samples.
    """
    model.eval()
    model.to(device)

    losses_per_class: List[List[float]] = [[] for _ in range(num_classes)]
    counts = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            sample_losses = F.cross_entropy(logits, labels, reduction="none")
            for cls in range(num_classes):
                if counts[cls] >= per_class_cap:
                    continue
                mask = labels == cls
                if not mask.any():
                    continue
                cls_losses = sample_losses[mask].detach().cpu().tolist()
                room = per_class_cap - int(counts[cls])
                take = cls_losses[:room]
                losses_per_class[cls].extend(take)
                counts[cls] += len(take)
            if (counts >= per_class_cap).all():
                break

    out = np.full(num_classes, np.nan, dtype=np.float64)
    for cls in range(num_classes):
        if counts[cls] > 0:
            out[cls] = float(np.mean(losses_per_class[cls]))
    return out


def _build_cluster_state(
    cluster_indices: List[int],
    client_updates: List[Dict[str, torch.Tensor]],
    global_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Cluster representative = global + mean(delta_i for i in cluster).
    Equivalent to averaging the full client states within the cluster when all
    clients started from the same global model, but written explicitly in
    delta-space so the rest of the pipeline stays consistent.
    """
    new_state: Dict[str, torch.Tensor] = {}
    n_c = float(len(cluster_indices))
    for key, g_tensor in global_state.items():
        if g_tensor.dtype.is_floating_point:
            mean_delta = torch.zeros_like(g_tensor, dtype=torch.float32)
            for i in cluster_indices:
                mean_delta += (
                    client_updates[i][key].to(dtype=torch.float32, device=g_tensor.device)
                    - g_tensor.to(dtype=torch.float32)
                )
            mean_delta /= n_c
            new_state[key] = (g_tensor.to(torch.float32) + mean_delta).to(g_tensor.dtype)
        else:
            # Non-float buffers (e.g. num_batches_tracked): take from first client
            new_state[key] = client_updates[cluster_indices[0]][key].clone()
    return new_state


# ----------------------------------------------------------------------------- #
# Main entry point
# ----------------------------------------------------------------------------- #
def flshield(
    global_state: Dict[str, torch.Tensor],
    client_updates: List[Dict[str, torch.Tensor]],
    global_model: nn.Module,
    val_loader: DataLoader,
    num_classes: int = 10,
    device: Optional[torch.device] = None,
    weights: Optional[List[float]] = None,
    start_round: int = 0,            # kept for API compat; not used
    current_round: int = 0,          # kept for API compat; not used
) -> Tuple[Dict[str, torch.Tensor], List[int]]:
    """
    Server-side FLShield aggregation (server-validation variant).

    Parameters
    ----------
    global_state : current global model state_dict
    client_updates : list of FULL parameter state_dicts from clients
    global_model : current global nn.Module (used as a template)
    val_loader : server-held validation DataLoader
    num_classes : number of classification classes
    device : evaluation device
    weights : optional per-client base weights (e.g. dataset sizes)

    Returns
    -------
    aggregated_state : new global state_dict
    accepted_indices : indices of clients whose updates were aggregated

    Notes
    -----
    This is the server-validation variant. The paper's full method uses
    client-side validators with a 2-means majority filter across validators
    (see flshield_utils/validation_preprocessing.py::ValidationProcessor).
    Implementing that requires changes to the trainer to expose per-client
    validation partitions and is not done here.
    """
    n = len(client_updates)
    if n == 0:
        raise ValueError("No client updates provided.")
    if device is None:
        device = next(global_model.parameters()).device

    # ---------- 1. Cluster on deltas ----------
    deltas = np.stack(
        [_flatten_delta(u, global_state) for u in client_updates], axis=0
    )
    _labels, clusters, _coses = _cluster_deltas(deltas, max_k=15)
    num_clusters = len(clusters)

    # ---------- 2. Global per-class baseline loss ----------
    base_model = copy.deepcopy(global_model).to(device)
    base_model.load_state_dict(global_state, strict=True)
    global_class_loss = _per_class_loss_capped(
        base_model, val_loader, num_classes, device
    )
    _log.info(
        "[FLShield] Global per-class loss: %s",
        ["nan" if np.isnan(x) else f"{x:.4f}" for x in global_class_loss],
    )

    # ---------- 3. Per-cluster representative + LIPC ----------
    cluster_scores: List[float] = []
    for cidx, cluster in enumerate(clusters):
        cluster_state = _build_cluster_state(cluster, client_updates, global_state)
        rep_model = copy.deepcopy(global_model).to(device)
        rep_model.load_state_dict(cluster_state, strict=True)

        cluster_class_loss = _per_class_loss_capped(
            rep_model, val_loader, num_classes, device
        )

        # LIPC: positive = cluster better than global. NaN classes ignored.
        lipc = global_class_loss - cluster_class_loss
        valid_lipc = lipc[np.isfinite(lipc)]
        if valid_lipc.size == 0:
            score = -np.inf
        else:
            score = float(np.min(valid_lipc))
        cluster_scores.append(score)

        _log.info(
            "[FLShield] Cluster %d (size=%d) min-LIPC=%.4f, per-class LIPC=%s",
            cidx, len(cluster), score,
            ["nan" if not np.isfinite(x) else f"{x:.4f}" for x in lipc],
        )

        del rep_model

    # ---------- 4. Accept the top half of clusters by min-LIPC ----------
    order = np.argsort(cluster_scores)  # ascending: worst first
    n_accept = max(1, (num_clusters + 1) // 2)  # top half (ceiling)
    accepted_cluster_idx = set(order[-n_accept:].tolist())

    accepted_clients: List[int] = []
    for cidx in sorted(accepted_cluster_idx):
        accepted_clients.extend(clusters[cidx])
    accepted_clients.sort()

    rejected_clients = sorted(set(range(n)) - set(accepted_clients))
    _log.info(
        "[FLShield] Accepted %d/%d clients: %s | Rejected: %s",
        len(accepted_clients), n, accepted_clients, rejected_clients,
    )

    if not accepted_clients:
        # Defensive: keep the global model unchanged rather than collapse
        _log.warning("[FLShield] No clients accepted; returning unchanged global.")
        return {k: v.clone() for k, v in global_state.items()}, []

    # ---------- 5. Weighted FedAvg over accepted deltas ----------
    if weights is None:
        base_w = np.ones(len(accepted_clients), dtype=np.float64)
    else:
        base_w = np.array([float(weights[i]) for i in accepted_clients],
                          dtype=np.float64)
    if base_w.sum() <= 0:
        base_w = np.ones_like(base_w)
    base_w = base_w / base_w.sum()

    new_state: Dict[str, torch.Tensor] = {}
    for key, g_tensor in global_state.items():
        if g_tensor.dtype.is_floating_point:
            agg_delta = torch.zeros_like(g_tensor, dtype=torch.float32)
            for w_i, idx in zip(base_w, accepted_clients):
                client_param = client_updates[idx][key].to(
                    dtype=torch.float32, device=g_tensor.device
                )
                agg_delta += float(w_i) * (
                    client_param - g_tensor.to(torch.float32)
                )
            new_state[key] = (g_tensor.to(torch.float32) + agg_delta).to(g_tensor.dtype)
        else:
            # Integer/bool buffers: copy from the first accepted client
            new_state[key] = client_updates[accepted_clients[0]][key].clone()

    return new_state, accepted_clients