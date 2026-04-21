##FLAME Defense Implementation ; Nguyen et al. 2022
import torch
import math
import numpy as np
import hdbscan
from copy import deepcopy
from typing import List, Dict, Tuple


def _get_float_keys(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    """Return keys whose tensors are floating-point (excludes int buffers like num_batches_tracked)."""
    return [k for k, v in state_dict.items() if v.is_floating_point()]


def _is_bn_running_stat(key: str) -> bool:
    """Check if a key is a BatchNorm running statistic (not a learned parameter)."""
    return 'running_mean' in key or 'running_var' in key


def flatten_float_params(state_dict: Dict[str, torch.Tensor], float_keys: List[str]) -> torch.Tensor:
    """Flatten only the floating-point entries of a state dict into a single 1D tensor."""
    return torch.cat([state_dict[k].view(-1) for k in float_keys])


def reconstruct_float_params(
    flat_vec: torch.Tensor,
    reference_state_dict: Dict[str, torch.Tensor],
    float_keys: List[str],
) -> Dict[str, torch.Tensor]:
    """Reconstruct float entries from a flat vector; non-float keys are copied from reference."""
    new_state_dict = {}
    pointer = 0
    for key in float_keys:
        param = reference_state_dict[key]
        numel = param.numel()
        new_state_dict[key] = flat_vec[pointer:pointer + numel].view(param.shape).clone()
        pointer += numel
    # Copy non-float buffers (e.g. num_batches_tracked) unchanged
    for key, val in reference_state_dict.items():
        if key not in new_state_dict:
            new_state_dict[key] = val.clone()
    return new_state_dict


def euclidean_distance(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """Compute Euclidean (L2) distance between two vectors."""
    return torch.norm(vec1 - vec2).item()

def flame(
        global_state: Dict[str, torch.Tensor],
        client_updates: List[Dict[str, torch.Tensor]],
        epsilon: float,
        delta: float,
        lamda: float = None,
        nan_handling: str = "exclude"  # How to handle NaN: "exclude", "zero", or "global"
) -> Tuple[Dict[str, torch.Tensor], List[int]]:
    """


    """
    n = len(client_updates)
    print(f"\n[FLAME] Processing {n} client updates", flush=True)

    # Identify float keys once (excludes int buffers like num_batches_tracked)
    float_keys = _get_float_keys(global_state)
    # Separate learned params from BN running stats — noise should NOT be added to running stats
    param_keys = [k for k in float_keys if not _is_bn_running_stat(k)]
    bn_stat_keys = [k for k in float_keys if _is_bn_running_stat(k)]

    # Flatten with NaN detection (robustness only, no algorithmic change)
    global_vec = flatten_float_params(global_state, param_keys).cpu()
    flat_updates = []
    valid_clients = []

    # Check global model for NaN — if corrupted, sanitise before proceeding
    global_has_nan = False
    for key in float_keys:
        v = global_state[key]
        if torch.isnan(v).any() or torch.isinf(v).any():
            global_has_nan = True
            break
    if global_has_nan:
        print("  ⚠ GLOBAL MODEL has NaN/Inf — sanitising (replacing NaN/Inf with 0)", flush=True)
        for key in float_keys:
            global_state[key] = torch.nan_to_num(global_state[key], nan=0.0, posinf=0.0, neginf=0.0)
        global_vec = flatten_float_params(global_state, param_keys).cpu()

    for i, update in enumerate(client_updates):
        flat = flatten_float_params(update, param_keys).cpu()

        # Check both learned params and BN stats for NaN/Inf
        bn_has_nan = any(
            torch.isnan(update[k]).any() or torch.isinf(update[k]).any()
            for k in bn_stat_keys if k in update
        )
        if torch.isnan(flat).any() or torch.isinf(flat).any() or bn_has_nan:
            # Debug: identify which keys have NaN
            nan_keys = [k for k in float_keys
                        if k in update and (torch.isnan(update[k]).any() or torch.isinf(update[k]).any())]
            print(f"  ⚠ Client {i}: NaN/Inf detected in keys: {nan_keys[:5]}", flush=True)

            if nan_handling == "exclude":
                print(f"    → Excluding client {i}", flush=True)
                continue
            elif nan_handling == "zero":
                flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
            elif nan_handling == "global":
                nan_mask = torch.isnan(flat) | torch.isinf(flat)
                flat[nan_mask] = global_vec[nan_mask]

        flat_updates.append(flat)
        valid_clients.append(i)

    if len(flat_updates) == 0:
        return global_state, []



    # Convert to numpy for sklearn
    update_matrix = np.stack([upd.numpy().astype(np.float64) for upd in flat_updates])

    # Precompute cosine distance matrix (Algorithm 1, Step 1)
    from sklearn.metrics.pairwise import cosine_distances
    cosine_distance_matrix = cosine_distances(update_matrix).astype(np.float64)

    # HDBSCAN clustering (Algorithm 1, Step 2)
    min_cluster_size = int(len(valid_clients) / 2) + 1
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        allow_single_cluster=True,
        metric='precomputed'
    )
    cluster_labels = clusterer.fit_predict(cosine_distance_matrix)
    print(f"  Cluster labels: {cluster_labels}", flush=True)

    # Identify largest cluster (Algorithm 1, Step 3)
    valid_labels = cluster_labels[cluster_labels != -1]
    if len(valid_labels) == 0:
        admitted_local = list(range(len(valid_clients)))
    else:
        unique, counts = np.unique(valid_labels, return_counts=True)
        largest_cluster = unique[np.argmax(counts)]
        admitted_local = [i for i in range(len(valid_clients)) if cluster_labels[i] == largest_cluster]

    admitted_indices = [valid_clients[i] for i in admitted_local]
    print(f"  ✓ Admitted: {admitted_indices}", flush=True)

    # Convert to deltas (update - global) for clipping and noise steps.
    # FLAME's clipping bound and DP noise are defined over deltas, not full weights.
    flat_deltas = [upd - global_vec for upd in flat_updates]

    # Compute adaptive clipping bound S_t (Algorithm 1, Step 4)
    delta_norms = [torch.norm(flat_deltas[i]).item() for i in range(len(flat_deltas))]
    S_t = np.median(delta_norms)
    print(f"  Adaptive clipping bound S_t: {S_t:.6f}", flush=True)

    # Clip admitted deltas (Algorithm 1, Step 5)
    clipped_deltas = []
    for local_idx in admitted_local:
        norm_i = delta_norms[local_idx]
        gamma = min(1.0, S_t / norm_i) if norm_i > 0 else 1.0
        clipped_deltas.append(gamma * flat_deltas[local_idx])

    # Average clipped deltas (Algorithm 1, Step 6)
    avg_delta = sum(clipped_deltas) / len(clipped_deltas) if clipped_deltas else torch.zeros_like(global_vec)

    # ✅ ADD DIFFERENTIAL PRIVACY NOISE (Algorithm 1, Step 7)
    if lamda is not None:
        lam = lamda
    else:
        lam = (1.0 / float(epsilon)) * math.sqrt(2 * math.log(1.25 / float(delta)))
    sigma = lam * S_t
    print(f"  DP noise: lambda={lam:.6f}, sigma={sigma:.6f}", flush=True)
    noise = torch.normal(mean=0.0, std=max(sigma, 1e-8), size=avg_delta.size())
    new_global_vec = global_vec + avg_delta + noise

    # ✅ RECONSTRUCT WITH DTYPE PRESERVATION (learned params only)
    new_global_state = reconstruct_float_params(new_global_vec, global_state, param_keys)

    # Average BN running stats from admitted clients WITHOUT noise
    if bn_stat_keys and clipped_deltas:
        for key in bn_stat_keys:
            avg_stat = torch.zeros_like(global_state[key], dtype=torch.float32).cpu()
            for local_idx in admitted_local:
                avg_stat += client_updates[valid_clients[local_idx]][key].float().cpu()
            avg_stat /= len(admitted_local)
            new_global_state[key] = avg_stat.to(global_state[key].dtype)

    # Clamp running_var to stay positive (prevents NaN from sqrt in BatchNorm)
    for key in new_global_state:
        if 'running_var' in key:
            new_global_state[key] = torch.clamp(new_global_state[key], min=1e-7)

    return new_global_state, admitted_indices

