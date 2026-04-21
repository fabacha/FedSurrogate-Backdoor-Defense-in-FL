"""
FedSurrogate Defense
"""

from __future__ import annotations
import logging
import numpy as np, torch, hdbscan
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

_log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Fast Cosine Distance (pure torch — avoids float64 numpy copy)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def _cosine_distance_matrix(X: torch.Tensor) -> np.ndarray:
    """Pairwise cosine distance matrix, returned as float64 numpy for HDBSCAN.

    """
    norms = X.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    X_n = X / norms
    sim = X_n @ X_n.T                     # [N, N] cosine similarity
    D = (1.0 - sim).clamp(min=0.0)        # numerical safety
    return D.double().numpy()


# ══════════════════════════════════════════════════════════════════
# Architecture Detection
# ══════════════════════════════════════════════════════════════════

def detect_architecture(state_dict) -> str:
    keys = set(state_dict.keys())
    if any(k.startswith('layer') for k in keys) and 'bn1.weight' in keys:
        return 'resnet'
    elif 'fc2.weight' in keys and 'fc1.weight' in keys:
        return 'simple_cnn'
    return 'unknown'


def get_final_fc_layers(state_dict) -> Tuple[str, Optional[str]]:
    keys = list(state_dict.keys())
    fc_w = [k for k in keys if 'fc' in k and 'weight' in k and 'bn' not in k]
    if not fc_w:
        raise ValueError(f"No FC layer found. Keys: {keys[:10]}...")
    w_name = fc_w[-1]
    b_name = w_name.replace('.weight', '.bias')
    return w_name, b_name if b_name in keys else None


def get_scoring_layers(state_dict, mode: str = "conv_fc") -> List[str]:
    """Get candidate layers for scoring (used as fallback if LCA disabled)."""
    keys = list(state_dict.keys())
    arch = detect_architecture(state_dict)

    if mode == "all":
        return [k for k in keys if 'weight' in k or 'bias' in k]
    elif mode == "fc_only":
        return [k for k in keys
                if 'fc' in k and ('weight' in k or 'bias' in k) and 'bn' not in k]
    elif mode == "conv_fc":
        return [k for k in keys
                if ('conv' in k or 'fc' in k)
                and ('weight' in k or 'bias' in k) and 'bn' not in k]
    elif mode == "mid_deep":
        # Intermediate-to-deep conv layers + FC head
        # Best for gradient alignment (rescue/screening) on ResNet-18:
        #   layer2.1 + layer3 + fc2 — captures trigger feature encoding
        #   and classification-level backdoor signal
        if arch == 'resnet':
            return [k for k in keys
                    if (k.startswith('layer2.1') or
                        k.startswith('layer3') or
                        k.startswith('fc'))
                    and ('weight' in k)
                    and 'bn' not in k
                    and 'shortcut' not in k]
        elif arch == 'simple_cnn':
            # Fallback: fc1 + fc2 for simple CNN
            return [k for k in keys
                    if ('fc1' in k or 'fc2' in k)
                    and ('weight' in k)]
        else:
            wb = [k for k in keys if 'weight' in k and 'bn' not in k]
            return wb[-max(2, len(wb) // 3):]
    elif mode == "last_block":
        if arch == 'resnet':
            return [k for k in keys
                    if (k.startswith('layer4') or k.startswith('fc'))
                    and ('weight' in k or 'bias' in k)]
        elif arch == 'simple_cnn':
            return [k for k in keys
                    if ('conv2' in k or 'fc1' in k or 'fc2' in k)
                    and ('weight' in k or 'bias' in k) and 'bn' not in k]
        else:
            wb = [k for k in keys if 'weight' in k or 'bias' in k]
            return wb[-max(1, len(wb) // 3):]
    raise ValueError(f"Unknown mode: {mode}")


# ══════════════════════════════════════════════════════════════════
# Layer Criticality Analysis (LCA)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def layer_criticality_analysis(
    local_models: List[Tuple],
    global_model: torch.nn.Module,
    *,
    candidate_layers: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    sigma_threshold: float = 1.5,
    min_params: int = 500,
    min_critical_params: int = 50000,
    device: str = "cpu",
) -> Tuple[List[str], Dict[str, float]]:
    """
    Select critical layers by cross-client directional divergence.
    """
    device = torch.device(device)
    global_sd = global_model.state_dict()

    # Determine candidate layers
    if candidate_layers is None:
        candidate_layers = [
            k for k in global_sd.keys()
            if ('weight' in k or 'bias' in k)
            and 'bn' not in k
            and 'running' not in k
            and 'num_batches' not in k
            and global_sd[k].numel() >= min_params
        ]

    n_clients = len(local_models)

    # Pre-cache all client state_dicts (avoids calling .state_dict() per layer)
    client_sds = [mdl.state_dict() for _, mdl in local_models]

    # Pre-compute upper-triangle mask (reused across all layers)
    triu_mask = torch.triu(torch.ones(n_clients, n_clients, device=device), diagonal=1).bool()

    # ── Compute per-layer directional divergence ──
    layer_scores = {}

    for layer_name in candidate_layers:
        if layer_name not in global_sd:
            continue

        g_param = global_sd[layer_name].float().to(device).flatten()

        # Collect per-client update vectors for this layer
        update_vecs = []
        skip = False
        for sd in client_sds:
            if layer_name not in sd:
                skip = True
                break
            delta = sd[layer_name].float().to(device).flatten() - g_param
            update_vecs.append(delta)

        if skip or len(update_vecs) != n_clients:
            continue

        # Stack into matrix [N x params] and compute pairwise cosine distances
        U = torch.stack(update_vecs)  # [N, params]

        # Normalise rows to unit vectors for cosine distance
        norms = U.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        U_normed = U / norms

        # Pairwise cosine similarity matrix → distance = 1 - similarity
        cos_sim = U_normed @ U_normed.T  # [N, N]
        cos_dist = 1.0 - cos_sim

        # Extract upper triangle (exclude diagonal)
        pairwise_dists = cos_dist[triu_mask]

        # Score: mean pairwise cosine distance
        mean_dist = pairwise_dists.mean().item()

        layer_scores[layer_name] = mean_dist

    if not layer_scores:
        return candidate_layers, {}

    # ── Normalise and select critical layers ──
    all_scores = np.array(list(layer_scores.values()))
    all_names = list(layer_scores.keys())
    median_score = np.median(all_scores)

    # Normalise: ratio to median (layers at 1.0 = typical, >1.0 = anomalous)
    if median_score > 1e-12:
        normalised = all_scores / median_score
    else:
        normalised = np.ones_like(all_scores)

    # MAD of normalised scores
    med_norm = np.median(normalised)
    mad = np.median(np.abs(normalised - med_norm))
    mad = max(mad, 1e-8)

    # Update scores dict with normalised values for reporting
    norm_scores = {name: ns for name, ns in zip(all_names, normalised)}

    if top_k is not None:
        sorted_layers = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
        critical_layers = [name for name, _ in sorted_layers[:top_k]]
    else:
        threshold = med_norm + sigma_threshold * mad
        critical_layers = [
            name for name, ns in norm_scores.items()
            if ns > threshold
        ]

        # ── Ensure minimum parameter count for effective clustering ──
        critical_params = sum(global_sd[l].numel() for l in critical_layers)

        if critical_params < min_critical_params:
            # Expand by adding next-highest-scored layers until we hit min
            sorted_layers = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
            for name, _ in sorted_layers:
                if name not in critical_layers:
                    critical_layers.append(name)
                    critical_params += global_sd[name].numel()
                    if critical_params >= min_critical_params:
                        break

        # Guarantee at least 3 layers
        if len(critical_layers) < 3:
            sorted_layers = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
            critical_layers = [name for name, _ in sorted_layers[:max(3, len(sorted_layers) // 3)]]

    return critical_layers, norm_scores


@torch.no_grad()
def l2_norm_layer_selection(
    local_models: List[Tuple],
    global_model: torch.nn.Module,
    *,
    candidate_layers: Optional[List[str]] = None,
    top_k: int = 5,
    min_params: int = 500,
    always_include_fc: bool = True,
    device: str = "cpu",
) -> Tuple[List[str], Dict[str, float]]:
    """

    Returns:
        selected_layers: List of selected layer names
        scores: Dict mapping layer_name -> median L2 norm
    """
    device_t = torch.device(device)
    global_sd = global_model.state_dict()

    if candidate_layers is None:
        candidate_layers = [
            k for k in global_sd.keys()
            if ('weight' in k or 'bias' in k)
            and 'bn' not in k
            and 'running' not in k
            and 'num_batches' not in k
            and global_sd[k].numel() >= min_params
        ]

    client_sds = [mdl.state_dict() for _, mdl in local_models]
    n_clients = len(local_models)

    layer_scores = {}

    for layer_name in candidate_layers:
        if layer_name not in global_sd:
            continue

        g_param = global_sd[layer_name].float().to(device_t).flatten()

        l2_norms = []
        skip = False
        for sd in client_sds:
            if layer_name not in sd:
                skip = True
                break
            delta = sd[layer_name].float().to(device_t).flatten() - g_param
            l2_norms.append(delta.norm(p=2).item())

        if skip or len(l2_norms) != n_clients:
            continue

        # Use median L2 norm (robust to outlier malicious clients)
        layer_scores[layer_name] = float(np.median(l2_norms))

    if not layer_scores:
        return candidate_layers, {}

    # Rank by median L2 norm (descending) and pick top-k
    sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
    selected = [name for name, _ in sorted_layers[:top_k]]

    # Always include the final FC weight layer
    if always_include_fc:
        fc_weight, _ = get_final_fc_layers(global_sd)
        if fc_weight and fc_weight not in selected:
            selected.append(fc_weight)

    _log.info("L2-norm layer selection: top-%d → %s", top_k, selected)
    _log.info("  Scores: %s", {n: f"{s:.4f}" for n, s in sorted_layers[:top_k + 3]})

    return selected, layer_scores


# ══════════════════════════════════════════════════════════════════
# Core Computation Helpers
# ══════════════════════════════════════════════════════════════════

_MEM = {"round": 0, "comp": defaultdict(float)}


def _compute_loo_scores(target_cids, cid_list, W_all, G_all, weights, device, use_loo=True):
    """LOO compromise scores: cs_i = [(W_i - W*_{-i})^T · ∇*_{-i}] / [‖...‖ · ‖...‖]"""
    W_star = torch.zeros_like(W_all[0])
    G_star = torch.zeros_like(G_all[0])
    for cid, w, g in zip(cid_list, W_all, G_all):
        W_star += weights[cid] * w
        G_star += weights[cid] * g
    total_w = sum(weights[cid] for cid in cid_list)

    cid_idx = {c: i for i, c in enumerate(cid_list)}
    scores = []
    for cid in target_cids:
        idx = cid_idx[cid]
        Wi, Gi, wi = W_all[idx], G_all[idx], weights[cid]

        if use_loo:
            rem = total_w - wi
            if rem < 1e-12:
                scores.append(0.0); continue
            Wr = (W_star - wi * Wi) / rem
            Gr = (G_star - wi * Gi) / rem
        else:
            Wr, Gr = W_star, G_star

        diff = Wi - Wr
        num = torch.dot(diff, Gr)
        den = diff.norm(p=2) * Gr.norm(p=2) + 1e-12
        scores.append((num / den).item())
    return scores


def _extract_layer_vectors(cid_list, cid2sd, global_sd, layers, device):
    """Extract concatenated weight + gradient vectors for given layers.

    Args:
        cid2sd: dict mapping client_id -> state_dict (pre-cached).
    """
    W_all, G_all = [], []
    for cid in cid_list:
        st = cid2sd[cid]
        wp, gp = [], []
        for ln in layers:
            if ln in st and ln in global_sd:
                w = st[ln].float().to(device).flatten()
                wp.append(w)
                gp.append(w - global_sd[ln].float().to(device).flatten())
        W_all.append(torch.cat(wp) if wp else torch.zeros(1, device=device))
        G_all.append(torch.cat(gp) if gp else torch.zeros(1, device=device))
    return W_all, G_all


def _compute_weights(css, cid_list):
    """Normalised sample-size weights."""
    if css is None:
        return {c: 1.0 / len(cid_list) for c in cid_list}
    if isinstance(css, dict):
        t = sum(css.values())
        return {c: css.get(c, 1) / t for c in cid_list}
    t = sum(css)
    return {c: css[i] / t for i, c in enumerate(cid_list)}


def blend_models(mal_model, donor_model, benign_ratio=0.7):
    """θ_blend = α·θ_donor + (1-α)·θ_malicious"""
    mal_sd = mal_model.state_dict()
    donor_sd = donor_model.state_dict()
    return {k: benign_ratio * donor_sd[k].float() + (1 - benign_ratio) * mal_sd[k].float()
            for k in mal_sd if k in donor_sd}


# ══════════════════════════════════════════════════════════════════
# Main Defence
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def FedSurrogate(
        local_models, global_model,
        *,
        client_sample_sizes=None,
        selected_layers=None,
        ultimate_weight: Optional[str] = None,
        ultimate_bias: Optional[str] = None,
        min_cluster_frac=0.5,
        # --- Layer Criticality Analysis (Stage 0) ---
        enable_lca: bool = False,
        lca_mode: str = "directional",  # "directional" (cosine divergence) | "l2_norm" (update magnitude)
        lca_sigma_threshold: float = 1.5,
        lca_top_k: Optional[int] = None,
        lca_min_params: int = 1000,
        lca_min_critical_params: int = 200000,
        lca_candidate_mode: str = "conv_fc",
        # --- Clustering (Stage 1) ---
        cluster_on_updates: bool = False,
        # --- Rescue (Stage 2) ---
        enable_rescue: bool = True,
        loo_rescue: bool = False,
        rescue_layer_mode: str = "mid_deep",  # "lca" | "ultimate" | "mid_deep" | "fc_only" | "conv_fc" | "all"
        zeta: float = 0.4,
        shrink_soft: float = 0.7,
        # --- Replacement (Stage 3) ---
        enable_replace: bool = True,
        replace_mode: str = "full",
        blend_benign_ratio: float = 0.7,
        shrink_replace: float = 0.3,
        # --- Donor selection ---
        donor_distance_metric: str = "cosine",
        donor_pool: str = "trusted",
        device: str = "cpu",
):
    """

    """
    if replace_mode not in ["full", "blend", "surgical", "drop"]:
        raise ValueError(f"Invalid replace_mode: '{replace_mode}'")
    if not 0.0 <= blend_benign_ratio <= 1.0:
        raise ValueError(f"blend_benign_ratio must be in [0, 1]")

    device_t = torch.device(device)
    cid_list = [cid for cid, _ in local_models]
    cid2mdl = {cid: mdl for cid, mdl in local_models}
    global_sd = global_model.state_dict()

    # ── Expand selected_layers if YAML passed a mode string ──
    # Accepts: None, a mode-string ("conv_fc", "mid_deep", ...), a 1-element
    # list containing a mode-string, or an explicit list of layer names.
    _MODE_NAMES = {"all", "fc_only", "conv_fc", "mid_deep", "last_block"}
    if isinstance(selected_layers, str):
        selected_layers = get_scoring_layers(global_sd, mode=selected_layers)
    elif (isinstance(selected_layers, (list, tuple))
          and len(selected_layers) == 1
          and selected_layers[0] in _MODE_NAMES):
        selected_layers = get_scoring_layers(global_sd, mode=selected_layers[0])

    print(f"[FedS] selected_layers ({len(selected_layers) if selected_layers else 0}): {selected_layers}")

    # ── Cache all client state_dicts ONCE (avoid repeated .state_dict() calls) ──
    cid2sd = {cid: mdl.state_dict() for cid, mdl in local_models}

    # ── Architecture detection ──
    arch = detect_architecture(global_sd)
    if ultimate_weight is None or ultimate_bias is None:
        aw, ab = get_final_fc_layers(global_sd)
        ultimate_weight = ultimate_weight or aw
        ultimate_bias = ultimate_bias or ab

    _log.info("FedSurrogate | arch=%s | FC=%s | LCA=%s (mode=%s) | rescue=%s | replace=%s",
              arch, ultimate_weight, enable_lca, lca_mode, rescue_layer_mode, replace_mode)

    # ══════════════════════════════════════════════════════════════
    # STAGE 0: Layer Selection
    # ══════════════════════════════════════════════════════════════
    if enable_lca:
        candidate_layers = get_scoring_layers(global_sd, mode=lca_candidate_mode)

        if lca_mode == "l2_norm":
            # L2-norm based: pick layers with largest median update norms
            critical_layers, lca_scores = l2_norm_layer_selection(
                local_models, global_model,
                candidate_layers=candidate_layers,
                top_k=lca_top_k or 5,
                min_params=lca_min_params,
                always_include_fc=True,
                device=device,
            )
        else:
            # "directional": original LCA (cosine divergence)
            critical_layers, lca_scores = layer_criticality_analysis(
                local_models, global_model,
                candidate_layers=candidate_layers,
                top_k=lca_top_k,
                sigma_threshold=lca_sigma_threshold,
                min_params=lca_min_params,
                min_critical_params=lca_min_critical_params,
                device=device,
            )

        critical_params = sum(global_sd[l].numel() for l in critical_layers)
        total_params = sum(global_sd[l].numel() for l in candidate_layers)
        _log.info("Layer selection (%s): %d/%d layers (%d/%d params = %.1f%%)",
                  lca_mode, len(critical_layers), len(candidate_layers),
                  critical_params, total_params,
                  100 * critical_params / max(total_params, 1))

        # Use selected layers for clustering
        clustering_layers = critical_layers

        # Rescue layers: selected by rescue_layer_mode
        if rescue_layer_mode == "lca":
            rescue_layers = list(critical_layers)
            _log.info("Rescue layers: using %d selected layers", len(rescue_layers))
        elif rescue_layer_mode == "ultimate":
            rescue_layers = [ultimate_weight]
            if ultimate_bias:
                rescue_layers.append(ultimate_bias)
        else:
            rescue_layers = get_scoring_layers(global_sd, mode=rescue_layer_mode)
    else:
        clustering_layers = None  # will use selected_layers or all

        if rescue_layer_mode == "lca":
            rescue_layers = [ultimate_weight]
            if ultimate_bias:
                rescue_layers.append(ultimate_bias)
            _log.info("Rescue layers: LCA disabled, falling back to ultimate")
        elif rescue_layer_mode == "ultimate":
            rescue_layers = [ultimate_weight]
            if ultimate_bias:
                rescue_layers.append(ultimate_bias)
        else:
            rescue_layers = get_scoring_layers(global_sd, mode=rescue_layer_mode)

    # ══════════════════════════════════════════════════════════════
    # STAGE 1: HDBSCAN Clustering on LCA-Selected Layers
    # ══════════════════════════════════════════════════════════════
    # Determine which layers to cluster on
    if clustering_layers is not None:
        effective_layers = set(clustering_layers)
    elif selected_layers is not None:
        effective_layers = set(selected_layers)
    else:
        effective_layers = None  # all layers

    vecs = []
    if cluster_on_updates:
        for cid in cid_list:
            sd = cid2sd[cid]
            parts = [
                (sd[k] - global_sd[k]).reshape(-1)
                for k in sd
                if effective_layers is None or k in effective_layers
            ]
            vecs.append(torch.cat(parts).cpu() if parts else torch.zeros(1))
    else:
        for cid in cid_list:
            sd = cid2sd[cid]
            parts = [
                sd[k].reshape(-1)
                for k in sd
                if effective_layers is None or k in effective_layers
            ]
            vecs.append(torch.cat(parts).cpu() if parts else torch.zeros(1))

    X = torch.stack(vecs).float()          # stay float32
    D = _cosine_distance_matrix(X)         # fast torch → small float64 numpy
    X_np = X.numpy()                       # for euclidean lookups in Stage 3

    mcs = max(2, int(len(cid_list) * min_cluster_frac) + 1)
    hdb = hdbscan.HDBSCAN(metric="precomputed",
                          min_cluster_size=mcs, min_samples=1,
                          allow_single_cluster=True).fit(D)
    lab = hdb.labels_
    benign_lab = max((l for l in lab if l != -1),
                     key=lambda l: (lab == l).sum(), default=None)

    coarse = [cid for cid, l in zip(cid_list, lab) if l == benign_lab]
    suspects = [cid for cid in cid_list if cid not in coarse]

    _MEM["round"] += 1
    r = _MEM["round"]
    _log.info("Round %d | trusted=%s | suspects=%s", r, coarse, suspects)

    if not suspects:
        shrink = {cid: 1.0 for cid in coarse}
        return coarse, [], {}, shrink

    # ══════════════════════════════════════════════════════════════
    # STAGE 2: Two-Way Gradient Alignment Filter (Screen + Rescue)
    # ══════════════════════════════════════════════════════════════
    weights = _compute_weights(client_sample_sizes, cid_list)
    rescues, malicious = [], []
    demoted = []

    if enable_rescue:
        # Extract LCA-layer vectors for ALL clients (using cached state_dicts)
        W_all, G_all = _extract_layer_vectors(
            cid_list, cid2sd, global_sd, rescue_layers, device_t)

        # ── 2a) SCREEN coarse-trusted clients ──────────────────
        if len(coarse) >= 4:
            screen_scores_raw = _compute_loo_scores(
                coarse, cid_list, W_all, G_all, weights, device_t, loo_rescue)

            screen_t = torch.tensor(screen_scores_raw, device=device_t)

            # Normalise to [0, 1]
            if screen_t.max() > screen_t.min():
                screen_norm = (screen_t - screen_t.min()) / (screen_t.max() - screen_t.min())
            else:
                screen_norm = torch.zeros_like(screen_t)

            # Cumulative screening scores across rounds
            for s, cid in zip(screen_norm.cpu().tolist(), coarse):
                key = f"screen_{cid}"
                m = r if key in _MEM["comp"] else 1
                _MEM["comp"][key] = (m - 1) / m * _MEM["comp"].get(key, 0.0) + s / m

            screen_cumul = torch.tensor(
                [_MEM["comp"].get(f"screen_{cid}", 0.0) for cid in coarse],
                device=device_t)

            # IQR-based outlier detection within the trusted set
            if len(screen_cumul) >= 4:
                q1 = torch.quantile(screen_cumul, 0.25).item()
                q3 = torch.quantile(screen_cumul, 0.75).item()
                iqr = q3 - q1
                screen_thresh = q3 + 1.5 * iqr
            else:
                screen_thresh = float('inf')

            # Demote outliers from trusted → suspects
            for cid, sc in zip(list(coarse), screen_cumul.cpu().tolist()):
                if sc > screen_thresh:
                    demoted.append(cid)

            for cid in demoted:
                coarse.remove(cid)
                suspects.append(cid)

            if demoted:
                _log.info("Screen: demoted %s to suspects", demoted)

        # ── 2b) RESCUE false positives from suspects ──────────
        if suspects:
            rescue_scores_raw = _compute_loo_scores(
                suspects, cid_list, W_all, G_all, weights, device_t, loo_rescue)

            cst = torch.tensor(rescue_scores_raw, device=device_t)

            # Normalise to [0, 1]
            if cst.max() > cst.min():
                cst = (cst - cst.min()) / (cst.max() - cst.min())
            else:
                cst = torch.zeros_like(cst)

            # Cumulative compromise scores
            for s, cid in zip(cst.cpu().tolist(), suspects):
                m = r if cid in _MEM["comp"] else 1
                _MEM["comp"][cid] = (m - 1) / m * _MEM["comp"].get(cid, 0.0) + s / m

            csi = torch.tensor([_MEM["comp"].get(cid, 0.0) for cid in suspects],
                              device=device_t)
            med = torch.median(csi).item()
            eps1 = min(zeta, med)

            rescues = [c for c, s in zip(suspects, csi) if s <= eps1]
            malicious = [c for c in suspects if c not in rescues]

            _log.info("Rescue: rescued=%s flagged=%s (ε=%.3f)", rescues, malicious, eps1)
    else:
        malicious = suspects[:]

    if not suspects and not malicious:
        shrink = {cid: 1.0 for cid in coarse}
        if enable_rescue:
            shrink.update({c: shrink_soft for c in rescues})
        return coarse + rescues, [], {}, shrink

    # ══════════════════════════════════════════════════════════════
    # STAGE 3: Replacement (Full / Blend / Drop)
    # ══════════════════════════════════════════════════════════════
    repl_map = {}

    if not enable_replace:
        _log.info("Replacement disabled — flagged clients will be dropped")
    elif replace_mode == "drop":
        pass  # nothing to do
    elif replace_mode in ["full", "blend", "surgical"] and malicious:
        available = coarse + rescues if donor_pool == "trusted" else coarse

        if available:
            idx_map = {c: i for i, c in enumerate(cid_list)}

            for fid in malicious:
                if donor_distance_metric == "euclidean":
                    donor = min(available,
                                key=lambda t: np.linalg.norm(
                                    X_np[idx_map[fid]] - X_np[idx_map[t]]))
                else:
                    donor = min(available,
                                key=lambda t: D[idx_map[fid], idx_map[t]])
                repl_map[fid] = donor

                if replace_mode == "blend":
                    blended = blend_models(cid2mdl[fid], cid2mdl[donor], blend_benign_ratio)
                    cid2mdl[fid].load_state_dict(blended)
                elif replace_mode == "surgical":
                    # Only replace LCA-critical layers; keep the rest
                    surgical_layers = set(critical_layers) if enable_lca else set()
                    # Also include associated bias layers
                    for lname in list(surgical_layers):
                        b_name = lname.replace('.weight', '.bias')
                        if b_name in global_sd and b_name != lname:
                            surgical_layers.add(b_name)
                    donor_sd = cid2mdl[donor].state_dict()
                    patched_sd = {
                        k: donor_sd[k].clone() if k in surgical_layers else v.clone()
                        for k, v in cid2mdl[fid].state_dict().items()
                    }
                    cid2mdl[fid].load_state_dict(patched_sd)
                    _log.info("Surgical replace client %d: %d/%d layers from donor %d",
                              fid, len(surgical_layers), len(patched_sd), donor)
                else:
                    # Full: fast clone instead of copy.deepcopy
                    cid2mdl[fid].load_state_dict(
                        {k: v.clone() for k, v in cid2mdl[donor].state_dict().items()})

            _log.info("Replacement [%s]: %s", replace_mode,
                      {fid: repl_map[fid] for fid in malicious if fid in repl_map})

    # ══════════════════════════════════════════════════════════════
    # STAGE 4: Shrink Factors
    # ══════════════════════════════════════════════════════════════
    shrink = {cid: 1.0 for cid in coarse}
    if enable_rescue:
        shrink.update({c: shrink_soft for c in rescues})
    if replace_mode == "drop":
        shrink.update({c: 0.0 for c in malicious})
    else:
        shrink.update({c: shrink_replace for c in repl_map})


    trusted = coarse + rescues
    return trusted, malicious, repl_map, shrink