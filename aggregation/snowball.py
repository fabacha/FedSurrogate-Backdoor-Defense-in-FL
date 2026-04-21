"""
Snowball Defense Implementation (Qin et al., AAAI 2024)
"Resisting Backdoor Attacks in Federated Learning via
 Bidirectional Elections and Individual Perspective"

Two-phase defense:
  Phase 1 (Bottom-Up): Per-layer K-Means clustering with Calinski-Harabasz
           scoring → top 10% by accumulated votes become "selectees"
  Phase 2 (Top-Down):  VAE trained on pairwise diffs of selectees;
           progressively expands the benign set by reconstruction error
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def _flatten_selected_layers(
    state_dict: Dict[str, torch.Tensor],
    layer_filters: List[str],
) -> torch.Tensor:
    """Flatten only the layers whose keys contain any of the filter strings."""
    parts = []
    for k, v in state_dict.items():
        if any(f in k for f in layer_filters):
            parts.append(v.detach().float().cpu().flatten())
    if not parts:
        raise ValueError(
            f"No layers matched filters {layer_filters}. "
            f"Available keys: {list(state_dict.keys())}"
        )
    return torch.cat(parts)


def _cluster(init_ids: List[int], data: np.ndarray) -> np.ndarray:
    """K-Means with fixed initial centroids (one per init_id)."""
    if np.isnan(data).any():
        data = np.nan_to_num(data, nan=0.0)
    init_centroids = np.array([data[i] for i in init_ids])
    km = KMeans(n_clusters=len(init_ids), init=init_centroids, n_init=1)
    return km.fit_predict(data)


class _DiffDataset(Dataset):
    def __init__(self, data: List[torch.Tensor]):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def _build_dif_set(data: List[torch.Tensor]) -> List[torch.Tensor]:
    """All ordered pairwise differences (i≠j)."""
    difs = []
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                difs.append(data[i] - data[j])
    return difs


def _obtain_dif(
    base: List[torch.Tensor], target: torch.Tensor,
) -> List[torch.Tensor]:
    """Asymmetric diffs: for each item in base, both (item-target) and (target-item)."""
    difs = []
    for item in base:
        if torch.sum(item - target) != 0.0:
            difs.append(item - target)
            difs.append(target - item)
    return difs


# ═══════════════════════════════════════════════════════════════
#  VAE
# ═══════════════════════════════════════════════════════════════

def _init_weights(model: nn.Module):
    for m in model.modules():
        classname = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
            kaiming_normal_(m.weight.data, nonlinearity="relu")


class _VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim

        # Encoder
        self.fc_e1 = nn.Linear(input_dim, hidden_dim)
        self.fc_e2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc_d1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_d2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_d3 = nn.Linear(hidden_dim, input_dim)

    def encoder(self, x):
        x = F.relu(self.fc_e1(x.view(-1, self.input_dim)))
        x = F.relu(self.fc_e2(x))
        return self.fc_mean(x), F.softplus(self.fc_logvar(x))

    def decoder(self, z):
        z = F.relu(self.fc_d1(z))
        z = F.relu(self.fc_d2(z))
        return torch.sigmoid(self.fc_d3(z)).view(-1, self.input_dim)

    def sample_normal(self, mean, logvar):
        sd = torch.exp(logvar * 0.5)
        return Variable(torch.randn_like(sd)).mul(sd).add_(mean)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.sample_normal(mean, logvar)
        return self.decoder(z), mean, logvar

    @torch.no_grad()
    def recon_prob(self, x_in, L: int = 10) -> float:
        x_in = torch.sigmoid(x_in.unsqueeze(0))
        mean, logvar = self.encoder(x_in)
        total = 0.0
        for _ in range(L):
            z = self.sample_normal(mean, logvar)
            x_out = self.decoder(z)
            total += F.mse_loss(x_out, x_in, reduction="sum").item()
        return total / L


_kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
_recon_loss = nn.MSELoss(reduction="sum")


def _train_vae(
    vae: Optional[_VAE],
    data: List[torch.Tensor],
    num_epochs: int,
    device: torch.device,
    latent_dim: int = 64,
    hidden_dim: int = 256,
) -> _VAE:
    stacked = torch.sigmoid(torch.stack(data, dim=0))
    if vae is None:
        vae = _VAE(input_dim=stacked.size(1), latent_dim=latent_dim, hidden_dim=hidden_dim)
        _init_weights(vae)
    vae = vae.to(device).train()

    loader = DataLoader(_DiffDataset(stacked), batch_size=8, shuffle=True)
    opt = torch.optim.Adam(vae.parameters())

    for _ in range(num_epochs):
        for x in loader:
            x = x.to(device)
            recon_x, mu, logvar = vae(x)
            loss = _recon_loss(recon_x, x) + torch.mean(_kl_loss(mu, logvar))
            opt.zero_grad()
            loss.backward()
            opt.step()

    return vae.cpu()


# ═══════════════════════════════════════════════════════════════
#  Main Snowball function
# ═══════════════════════════════════════════════════════════════

def snowball(
    global_state: Dict[str, torch.Tensor],
    local_models: List[Tuple[int, torch.nn.Module]],
    cur_round: int,
    *,
    weights: Optional[List[float]] = None,
    layer_filters: Optional[List[str]] = None,
    ct: int = 10,
    vt: float = 0.5,
    v_step: float = 0.05,
    vae_initial: int = 270,
    vae_tuning: int = 30,
    vae_hidden: int = 256,
    vae_latent: int = 64,
    warmup_rounds: int = 100,
    device: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], List[int]]:
    """
    Snowball defense (Algorithm 1 from Qin et al., AAAI 2024).

    Args:
        global_state:   current global model state_dict
        local_models:   list of (client_id, nn.Module)
        cur_round:      current FL round number
        weights:        per-client sample counts (for weighted avg); None → equal
        layer_filters:  layer name substrings for bottom-up election
        ct:             number of suspicious clusters (K-means uses ct+1)
        vt:             target fraction of clients after top-down election
        v_step:         fraction of clients added per expansion step
        vae_initial:    VAE initial training epochs
        vae_tuning:     VAE fine-tuning epochs per expansion step
        vae_hidden:     VAE hidden dimension
        vae_latent:     VAE latent dimension
        warmup_rounds:  rounds before top-down election activates
        device:         torch device for VAE training

    Returns:
        new_global_state, selected_client_ids
    """
    if layer_filters is None:
        layer_filters = ["conv1", "fc2"]

    n = len(local_models)
    client_ids = [cid for cid, _ in local_models]

    # --- Compute deltas (state_dict level) ---
    model_deltas = []
    for _, mdl in local_models:
        delta = {}
        st = mdl.state_dict()
        for k in global_state:
            delta[k] = (st[k].float() - global_state[k].float()).cpu()
        model_deltas.append(delta)

    # ═════════════════════════════════════════════════════════
    #  Phase 1: Bottom-Up Election
    # ═════════════════════════════════════════════════════════

    # Per-layer kernels (flattened deltas)
    all_keys = list(global_state.keys())
    kernels_by_layer = {}
    for k in all_keys:
        kernels_by_layer[k] = [model_deltas[i][k].flatten().numpy() for i in range(n)]

    cnt = np.zeros(n, dtype=np.float64)

    for layer_name in all_keys:
        # Layer filter
        if not any(f in layer_name for f in layer_filters):
            continue

        updates = kernels_by_layer[layer_name]
        benign_list = []
        score_list = []

        for idx_client in range(n):
            # Pairwise differences from this client's perspective
            ddif = np.array([updates[idx_client] - updates[j] for j in range(n)])
            norms = np.linalg.norm(ddif, axis=1)
            norm_rank = np.argsort(norms)

            # ct most distant clients are "suspicious"
            suspicious_idx = norm_rank[-ct:]
            centroid_ids = [idx_client] + list(suspicious_idx)

            cluster_result = _cluster(centroid_ids, ddif)

            try:
                score = calinski_harabasz_score(ddif, cluster_result)
            except ValueError:
                score = 0.0

            benign_ids = np.argwhere(
                cluster_result == cluster_result[idx_client]
            ).flatten()

            benign_list.append(benign_ids)
            score_list.append(score)

        score_arr = np.array(score_list)

        # Effective clients: those with CH score > 0, min 10%
        effective_ids = np.argwhere(score_arr > 0).flatten()
        if len(effective_ids) < int(n * 0.1):
            effective_ids = np.argsort(-score_arr)[:max(1, int(n * 0.1))]

        # Min-max normalize scores
        smin, smax = score_arr.min(), score_arr.max()
        if smax - smin > 0:
            score_norm = (score_arr - smin) / (smax - smin)
        else:
            score_norm = np.zeros_like(score_arr)

        # Accumulate votes
        for idx_client in effective_ids:
            for idx_b in benign_list[idx_client]:
                cnt[idx_b] += score_norm[idx_client]

    # Select top 10% by vote count
    cnt_rank = np.argsort(-cnt)
    n_select = max(1, math.ceil(n * 0.1))
    selected_ids = cnt_rank[:n_select].tolist()

    print(f"[Snowball] Bottom-up selected {len(selected_ids)}/{n}: "
          f"{[client_ids[i] for i in selected_ids]}", flush=True)

    # ═════════════════════════════════════════════════════════
    #  Early return if before warmup (bottom-up only)
    # ═════════════════════════════════════════════════════════
    if cur_round < warmup_rounds:
        return _aggregate(global_state, model_deltas, selected_ids, weights), \
               [client_ids[i] for i in selected_ids]

    # ═════════════════════════════════════════════════════════
    #  Phase 2: Top-Down Election (VAE-based expansion)
    # ═════════════════════════════════════════════════════════
    dev = torch.device(device)

    # Flatten selected layers for VAE
    flatten_list = [
        _flatten_selected_layers(model_deltas[i], layer_filters) for i in range(n)
    ]

    # Train VAE on pairwise diffs of selectees
    initial_difs = _build_dif_set([flatten_list[i] for i in selected_ids])
    vae = _train_vae(None, initial_difs, vae_initial, dev, vae_latent, vae_hidden)

    target_count = int(n * vt)

    while len(selected_ids) < target_count:
        # Fine-tune VAE
        cur_difs = _build_dif_set([flatten_list[i] for i in selected_ids])
        vae = _train_vae(vae, cur_difs, vae_tuning, dev, vae_latent, vae_hidden)
        vae.eval()

        # Score remaining clients by reconstruction error
        rest_ids = [i for i in range(n) if i not in selected_ids]
        losses = []
        with torch.no_grad():
            for idx in rest_ids:
                difs = _obtain_dif(
                    [flatten_list[i] for i in selected_ids],
                    flatten_list[idx],
                )
                if len(difs) == 0:
                    losses.append(float("inf"))
                    continue
                m_loss = sum(vae.recon_prob(d) for d in difs) / len(difs)
                losses.append(m_loss)

        # Add best candidates
        rank = np.argsort(losses)
        n_add = min(
            math.ceil(n * v_step),
            target_count - len(selected_ids),
        )
        selected_ids.extend(np.array(rest_ids)[rank[:n_add]].tolist())

    print(f"[Snowball] Top-down expanded to {len(selected_ids)}/{n}: "
          f"{sorted([client_ids[i] for i in selected_ids])}", flush=True)

    return _aggregate(global_state, model_deltas, selected_ids, weights), \
           [client_ids[i] for i in selected_ids]


def _aggregate(
    global_state: Dict[str, torch.Tensor],
    model_deltas: List[Dict[str, torch.Tensor]],
    selected_ids: List[int],
    weights: Optional[List[float]],
) -> Dict[str, torch.Tensor]:
    """Weighted average of selected deltas, applied to global state."""
    if weights is not None:
        w = np.array([weights[i] for i in selected_ids], dtype=np.float64)
        w = w / w.sum()
    else:
        w = np.full(len(selected_ids), 1.0 / len(selected_ids))

    new_state = {}
    for k, g in global_state.items():
        original_dtype = g.dtype
        agg_delta = sum(
            model_deltas[i][k] * w[j] for j, i in enumerate(selected_ids)
        )
        result = g.float().cpu() + agg_delta
        if original_dtype in (torch.long, torch.int64, torch.int32, torch.int):
            new_state[k] = result.round().to(original_dtype).to(g.device)
        else:
            new_state[k] = result.to(original_dtype).to(g.device)

    return new_state