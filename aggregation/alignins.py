"""
AlignIns Defense Implementation
Paper: "Detecting Backdoor Attacks in Federated Learning via
        Direction Alignment Inspection" (Xu et al.)

Implements Algorithm 1 exactly, including:
  - TDA (Eq. (1))
  - MPSA with Top-k and principal sign (Eq. (2))
  - MZ-score filtering (Def. 3, Eq. (3))
  - Post-filtering median-norm clipping

"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


class StateDict(dict):
    """
    Extended dict that allows attribute assignment for metadata WITHOUT
    polluting the dict keys (which would break load_state_dict).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store metadata in a separate namespace
        object.__setattr__(self, '_metadata', {})

    def __setattr__(self, name, value):
        # Store attributes in separate _metadata dict, not in the main dict
        if name.startswith('_'):
            object.__getattribute__(self, '_metadata')[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        # Retrieve from _metadata if it exists
        if name.startswith('_'):
            metadata = object.__getattribute__(self, '_metadata')
            if name in metadata:
                return metadata[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __hasattr__(self, name):
        if name.startswith('_'):
            metadata = object.__getattribute__(self, '_metadata')
            return name in metadata
        return super().__hasattr__(name)


def flatten_model(model_state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten model parameters into a single 1-D vector."""
    return torch.cat([p.flatten() for p in model_state_dict.values()])


def unflatten_model(
    flat_params: torch.Tensor,
    reference_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Unflatten a 1-D parameter tensor back into a state_dict structure."""
    state_dict = {}
    offset = 0

    for key, param in reference_state_dict.items():
        num_params = param.numel()
        state_dict[key] = flat_params[offset:offset + num_params].reshape(param.shape)
        offset += num_params

    return state_dict


def _compute_mz_scores(values: List[float], eps: float = 1e-12) -> np.ndarray:
    """
    MZ score as defined in the paper (Definition 3):

        λ_i = (x_i - med(X)) / σ

    where med(X) is the median and σ is the standard deviation of X.

    If σ is (near) zero, all scores are set to 0 (everyone at the median).
    """
    arr = np.asarray(values, dtype=float)
    med = np.median(arr)
    std = np.std(arr)

    if std < eps:
        return np.zeros_like(arr)

    return (arr - med) / std


def alignins_defense(
    local_models: List[Tuple[int, torch.nn.Module]],
    global_model: torch.nn.Module,
    *,
    client_sample_sizes: Optional[Dict[int, int]] = None,  # not used in AlignIns
    lambda_s: float = 1.0,   # MPSA MZ-score radius λ_s (paper default)
    lambda_c: float = 1.0,   # TDA MZ-score radius λ_c (paper default)
    sparsity: float = 0.3,   # fraction of coordinates for Top-k (≈ 0.3·d)
    device: str = "cpu",
    verbose: bool = True,
    malicious_clients: Optional[List[int]] = None,  # optional, for debugging only
):
    """
    AlignIns defense (Algorithm 1) for Byzantine/backdoor-robust aggregation.

    Args:
        local_models: list of (client_id, model) tuples (models after local training).
        global_model: current global model θ^t.
        client_sample_sizes: optional dict client_id -> num samples (unused).
        lambda_s: filtering radius for MPSA MZ-scores (λ_s).
        lambda_c: filtering radius for TDA MZ-scores (λ_c).
        sparsity: fraction of parameters used for MPSA (k = sparsity * d).
        device: torch device ("cpu" or "cuda").
        verbose: whether to print diagnostics.
        malicious_clients: optional list of known malicious client IDs (for eval/debug).

    Returns:
        aggregated_state: StateDict (θ^{t+1}) with attached metadata:
            - aggregated_state.selected_clients
            - aggregated_state.detection_metrics
        selected_clients: list of client IDs in the benign set S.
        detection_metrics: dict with TDA/MPSA/MZ scores and selected indices.
    """
    device = torch.device(device)

    # ===== 1. Extract client IDs and models =====
    client_ids: List[int] = []
    models: List[torch.nn.Module] = []

    for item in local_models:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"Expected (client_id, model) tuple, got: {item}")
        cid, model = item
        client_ids.append(cid)
        # Some frameworks pass (model, extra) as model; use first element if tuple.
        if isinstance(model, tuple):
            models.append(model[0])
        else:
            models.append(model)

    n_clients = len(models)
    if n_clients == 0:
        raise ValueError("alignins_defense received no local models.")

    if verbose:
        print(f"\n{'=' * 60}")
        print("AlignIns Defense (paper-faithful implementation with StateDict)")
        print(f"{'=' * 60}")
        print(f"Total clients: {n_clients}")
        print(f"Lambda_s (MPSA radius): {lambda_s}")
        print(f"Lambda_c (TDA radius):  {lambda_c}")
        print(f"Sparsity (Top-k fraction): {sparsity}")

    # Clamp sparsity into (0, 1]
    if sparsity <= 0.0 or sparsity > 1.0:
        raise ValueError(f"'sparsity' must be in (0, 1], got {sparsity}.")

    # ===== 2. Global model flat parameters θ^t =====
    global_state = global_model.state_dict()
    flat_global_model = flatten_model(global_state).to(device)  # θ^t
    d = flat_global_model.numel()

    # ===== 3. Local updates Δ_i = θ_i - θ =====
    local_updates: List[torch.Tensor] = []
    for model in models:
        local_state = model.state_dict()
        flat_local = flatten_model(local_state).to(device)
        update = flat_local - flat_global_model
        local_updates.append(update)

    inter_model_updates = torch.stack(local_updates, dim=0)  # shape: [n_clients, d]

    # ===== 4. TDA (Temporal Direction Alignment) =====
    # ω_i = <Δ_i, θ> / (||Δ_i|| ||θ||)  (Eq. (1))
    tda_list: List[float] = []
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    for i in range(n_clients):
        tda = cos(inter_model_updates[i], flat_global_model).item()
        tda_list.append(tda)

    if verbose:
        print("\nTDA (cosine similarity Δ_i vs θ):")
        print([round(x, 4) for x in tda_list])

    # ===== 5. MPSA (Masked Principal Sign Alignment) =====
    # Principal sign: p = sgn(∑_i sgn(Δ_i))
    major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))

    # k = floor(sparsity * d), 1 ≤ k ≤ d
    k = int(d * sparsity)
    k = max(1, min(k, d))

    mpsa_list: List[float] = []

    for i in range(n_clients):
        update = inter_model_updates[i]

        # Top-k by magnitude
        _, top_indices = torch.topk(torch.abs(update), k, largest=True, sorted=False)

        # sgn(Δ_i) and diff with principal sign
        sgn_i = torch.sign(update)
        diff = (sgn_i - major_sign)

        # Mask out non-top-k coordinates
        mask = torch.zeros_like(update, dtype=torch.bool)
        mask[top_indices] = True
        diff_masked = diff[mask]

        # L0 norm: number of non-zero entries (misaligned signs)
        num_not_aligned = torch.count_nonzero(diff_masked)
        # ρ_i = 1 - (# not-aligned) / k   (Eq. (2))
        rho_i = 1.0 - (num_not_aligned.item() / float(k))
        mpsa_list.append(rho_i)

    if verbose:
        print("\nMPSA (masked principal sign alignment ratio ρ_i):")
        print([round(x, 4) for x in mpsa_list])

    # ===== 6. MZ-score (median-based Z-score) =====
    # λ_i = (x_i - med(X)) / σ, then filter by |λ_i| ≤ λ.
    mzscore_tda = _compute_mz_scores(tda_list)
    mzscore_mpsa = _compute_mz_scores(mpsa_list)

    if verbose:
        print("\nMZ-score (TDA):")
        print([round(x, 4) for x in mzscore_tda])
        print("MZ-score (MPSA):")
        print([round(x, 4) for x in mzscore_mpsa])

    # ===== 7. Filtering: build benign set S =====
    benign_idx_tda = {
        i for i in range(n_clients)
        if abs(mzscore_tda[i]) <= lambda_c
    }
    benign_idx_mpsa = {
        i for i in range(n_clients)
        if abs(mzscore_mpsa[i]) <= lambda_s
    }
    benign_idx = sorted(list(benign_idx_tda.intersection(benign_idx_mpsa)))

    if verbose:
        print("\nFiltering Results:")
        print(f"  MPSA filter: {len(benign_idx_mpsa)} clients passed")
        print(f"  TDA filter:  {len(benign_idx_tda)} clients passed")
        print(f"  Both filters (benign set S): {len(benign_idx)} clients passed")

        if malicious_clients is not None:
            detected = [
                cid for j, cid in enumerate(client_ids)
                if j not in benign_idx and cid in malicious_clients
            ]
            missed = [cid for cid in malicious_clients if cid not in detected]
            print(f"  Malicious clients (provided): {malicious_clients}")
            print(f"  Detected malicious (filtered out): {detected}")
            print(f"  Missed malicious (passed): {missed}")

    # Edge case: no clients pass filtering.
    # Paper does not specify this; we use a safe fallback: no update.
    if len(benign_idx) == 0:
        if verbose:
            print("⚠ WARNING: No clients passed AlignIns filtering. Returning zero update.")
        zero_update = torch.zeros_like(flat_global_model)
        base_state = unflatten_model(flat_global_model + zero_update, global_state)
        aggregated_state = StateDict(base_state)
        selected_clients: List[int] = []

        detection_metrics = {
            "selected_count": 0,
            "tda_scores": tda_list,
            "mpsa_scores": mpsa_list,
            "mz_tda": mzscore_tda.tolist(),
            "mz_mpsa": mzscore_mpsa.tolist(),
            "benign_indices": benign_idx,
        }

        # Attach metadata for server-side hooks
        aggregated_state.selected_clients = selected_clients
        aggregated_state.detection_metrics = detection_metrics

        return aggregated_state, selected_clients, detection_metrics

    # ===== 8. Post-filtering model clipping (Alg. 1 lines 12–13) =====
    # S = benign_idx; compute c = median ||Δ_i|| over i ∈ S.
    benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)
    benign_norms = torch.norm(benign_updates, dim=1, keepdim=True)  # |S| x 1
    norm_clip = benign_norms.median(dim=0)[0].item()

    if verbose:
        print("\nNorm clipping:")
        print(f"  Median norm c over benign set S: {norm_clip:.4f}")

    # Clip only benign updates:
    #   Δ_i <- Δ_i * min{1, c / ||Δ_i||} for i ∈ S.
    # Safeguard against zero norms.
    safe_norms = benign_norms.clone()
    safe_norms[safe_norms == 0] = 1.0
    scale = torch.clamp(norm_clip / safe_norms, max=1.0)  # |S| x 1
    clipped_benign_updates = benign_updates * scale  # |S| x d

    # ===== 9. Aggregate clipped benign updates =====
    aggregated_update = clipped_benign_updates.mean(dim=0)  # 1 x d
    new_flat_model = flat_global_model + aggregated_update

    # Wrap in StateDict and attach metadata
    base_state = unflatten_model(new_flat_model, global_state)
    aggregated_state = StateDict(base_state)

    selected_clients = [client_ids[i] for i in benign_idx]

    detection_metrics = {
        "selected_count": len(benign_idx),
        "tda_scores": tda_list,
        "mpsa_scores": mpsa_list,
        "mz_tda": mzscore_tda.tolist(),
        "mz_mpsa": mzscore_mpsa.tolist(),
        "benign_indices": benign_idx,
    }

    aggregated_state.selected_clients = selected_clients
    aggregated_state.detection_metrics = detection_metrics

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Selected Clients (benign set S): {selected_clients}")
        print(f"{'=' * 60}\n")

    return aggregated_state, selected_clients, detection_metrics


def AlignIns(
    local_models: List[Tuple[int, torch.nn.Module]],
    global_model: torch.nn.Module,
    **kwargs,
):
    """
    Wrapper for compatibility with evaluation frameworks.

    Example usage:
        aggregated_state, selected_clients, metrics = AlignIns(
            local_models, global_model, lambda_c=1.0, lambda_s=1.0, sparsity=0.3
        )
        # Server can do:
        #   global_model.load_state_dict(aggregated_state)
        #   trusted = aggregated_state.selected_clients
        #   stats   = aggregated_state.detection_metrics
    """
    return alignins_defense(local_models, global_model, **kwargs)
