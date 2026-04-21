from __future__ import annotations

import copy
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import nn

_log = logging.getLogger(__name__)

ModelState = Dict[str, torch.Tensor]


# ----------------------------- helpers ----------------------------------------

def _is_bn_buffer(key: str) -> bool:
    """BatchNorm non-trainable buffers that often exist in state_dict()."""
    return any(s in key for s in ["running_mean", "running_var", "num_batches_tracked"])


def _flatten_trainable(state_dict: ModelState) -> np.ndarray:
    """
    Flatten trainable parameters only (exclude BN running stats) to a 1-D numpy vector.
    This is used ONLY for similarity computation.
    """
    parts: List[torch.Tensor] = []
    for key, p in state_dict.items():
        if not _is_bn_buffer(key):
            parts.append(p.reshape(-1).float())
    if not parts:
        return np.array([], dtype=np.float32)
    return torch.cat(parts).detach().cpu().numpy()


def _normalize_np(w: np.ndarray) -> np.ndarray:
    s = float(w.sum())
    if s <= 0.0:
        raise ValueError("Weights must sum to > 0.")
    return w / s


def _weighted_average_state_dicts(
    states: List[ModelState],
    weights: np.ndarray,
) -> ModelState:
    """Weighted average over FULL model state_dicts (absolute weights, not deltas)."""
    out: ModelState = {}
    for k in states[0].keys():
        acc = None
        for i in range(len(states)):
            term = states[i][k].float() * float(weights[i])
            acc = term if acc is None else (acc + term)
        out[k] = acc.to(states[0][k].dtype)
    return out


# ----------------------- coalition model building -----------------------------

def compute_coalition_models(
    client_updates: List[ModelState],
    global_model: nn.Module,
    weights: Optional[List[float]] = None,
) -> List[nn.Module]:
    """
    Build leave-one-out coalition models using the Eq.5 shortcut:

        IS_i = (w_g - a_i * I_i) / (1 - a_i)

    IMPORTANT:
      This shortcut is only correct if w_g is the weighted average of the SAME
      `client_updates` with the SAME `weights` (same round). To avoid subtle bugs,
      we compute w_g directly from `client_updates` here (do NOT assume global_model
      already equals that weighted average at this point in your pipeline).

    Returns:
      One coalition nn.Module per client (same length as client_updates).
    """
    n = len(client_updates)
    if n == 0:
        return []
    if n == 1:
        return [copy.deepcopy(global_model)]

    if weights is None:
        alpha = np.ones(n, dtype=np.float64) / n
    else:
        alpha = _normalize_np(np.asarray(weights, dtype=np.float64))

    # Build w_g from THIS round's client_updates
    w_g = _weighted_average_state_dicts(client_updates, alpha)

    coalition_models: List[nn.Module] = []
    for i in range(n):
        ai = float(alpha[i])

        # Edge case: one client has (almost) all weight -> coalition undefined
        if ai >= 1.0 - 1e-12:
            mdl = copy.deepcopy(global_model)
            mdl.load_state_dict(w_g, strict=False)
            coalition_models.append(mdl)
            continue

        coalition_sd: ModelState = {}
        for k in w_g.keys():
            coalition_sd[k] = (w_g[k].float() - ai * client_updates[i][k].float()) / (1.0 - ai)
            coalition_sd[k] = coalition_sd[k].to(client_updates[i][k].dtype)

        mdl = copy.deepcopy(global_model)
        mdl.load_state_dict(coalition_sd, strict=False)
        coalition_models.append(mdl)

    return coalition_models


# ----------------------- aligned-gradient local training -----------------------

def spmc_aligned_train(
    model: nn.Module,
    coalition_model: nn.Module,
    dataloader,
    optimizer,
    local_epochs: int,
    device: torch.device,
    lamda: float = 1.0,
    temperature: float = 1.0,
) -> None:
    """
    Client-side aligned-gradient training (ProGrad-style).

    Note: This function isn't required for the server-side SPMC aggregation,
    but is commonly imported alongside it.
    """
    import torch.nn.functional as F  # keep local to avoid unused import warnings

    model.train()
    coalition_model.to(device)
    coalition_model.eval()

    for _ in range(local_epochs):
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            with torch.no_grad():
                tea_output = coalition_model(images)

            xe_loss = F.cross_entropy(outputs, labels)

            tea_prob = F.softmax(tea_output / temperature, dim=-1)
            log_stu_prob = F.log_softmax(outputs / temperature, dim=-1)
            kl_loss = F.kl_div(log_stu_prob, tea_prob, reduction="batchmean") * (temperature ** 2)

            # Step A: grads for KL
            optimizer.zero_grad()
            kl_loss.backward(retain_graph=True)
            kl_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

            # Step B: grads for CE
            optimizer.zero_grad()
            xe_loss.backward()

            # Step C: projection if opposing
            for name, p in model.named_parameters():
                if p.grad is None or name not in kl_grads:
                    continue
                g_d = p.grad
                g_g = kl_grads[name]

                g_g_norm = torch.linalg.norm(g_g)
                if g_g_norm < 1e-12:
                    continue

                g_g_hat = g_g / g_g_norm
                dot = torch.dot(g_d.flatten(), g_g_hat.flatten())
                if dot < 0:
                    p.grad = g_d - lamda * dot * g_g_hat

            optimizer.step()


# ------------------------------ server-side -----------------------------------

def spmc(
    global_state: ModelState,                 # kept for API compatibility (not used)
    client_updates: List[ModelState],
    weights: Optional[List[float]] = None,    # FedAvg base weights (dataset sizes)
    *,
    eps: float = 1e-5,
    preserve_fedavg: bool = True,
) -> Tuple[ModelState, np.ndarray]:
    """
    SPMC/SPMD-like server aggregation (fixed to match the OFFICIAL repo's weight direction).

    Steps:
      1) Build leave-one-out coalition vectors using Eq.5 shortcut (weighted).
      2) Compute cosine similarity phi_i between client i and coalition(i).
      3) Compute SPMC scores s_i = exp(phi_i - max(phi) + eps)
         (higher similarity => higher score). This matches RobustFederation/Server/SPMD.py.
      4) Final elastic weights:
            - if preserve_fedavg: normalize(s_i * base_w_i)
            - else:              normalize(s_i)
      5) Aggregate FULL state_dicts using elastic weights.

    Returns:
      aggregated_state_dict, elastic_weights
    """
    n = len(client_updates)
    if n == 0:
        raise ValueError("No client updates provided.")
    if n == 1:
        return client_updates[0], np.array([1.0], dtype=np.float64)

    # Base weights (FedAvg)
    if weights is None:
        base_w = np.ones(n, dtype=np.float64) / n
    else:
        base_w = _normalize_np(np.asarray(weights, dtype=np.float64))

    # Flatten trainable params only for similarity computation
    flat_clients = [_flatten_trainable(u) for u in client_updates]

    # w_g (flat) = Σ base_w[i] * I_i
    flat_global = sum(float(base_w[i]) * flat_clients[i] for i in range(n))

    # Leave-one-out coalition vectors via Eq.5 shortcut
    coalition_vectors: List[np.ndarray] = []
    for i in range(n):
        wi = float(base_w[i])
        if wi >= 1.0 - 1e-12:
            coalition_vectors.append(flat_global)
        else:
            coalition_vectors.append((flat_global - wi * flat_clients[i]) / (1.0 - wi))

    # Cosine similarity phi_i = cos(client_i, coalition_i)
    phi = np.empty(n, dtype=np.float64)
    for i in range(n):
        c = coalition_vectors[i]
        x = flat_clients[i]
        denom = (np.linalg.norm(c) * np.linalg.norm(x)) + eps
        phi[i] = float(np.dot(c, x) / denom)

    _log.info("[SPMC] cosine similarities: %s", [f"{p:.6f}" for p in phi])

    # Official-repo-like: higher cosine => higher weight
    spmc_scores = np.exp(phi - phi.max() + eps)

    if preserve_fedavg:
        elastic_weights = _normalize_np(spmc_scores * base_w)
    else:
        elastic_weights = _normalize_np(spmc_scores)

    _log.info("[SPMC] elastic weights: %s", [f"{w:.6f}" for w in elastic_weights])

    aggregated = _weighted_average_state_dicts(client_updates, elastic_weights)
    return aggregated, elastic_weights
