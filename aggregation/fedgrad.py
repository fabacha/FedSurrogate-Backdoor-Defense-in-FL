import copy
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


# ============================================================
#  FedGrad Aggregator (Nguyen et.al., 2023)
#  Soft‑ & hard‑filter implementation with running trust scores
# ============================================================

class FedGradAggregator:
    """Server‑side FedGrad defence (Algorithm1 in the paper)."""

    # ─────────────── default hyper‑parameters (paper) ─────────────── #
    ζ: float = 0.5  # upper bound for adaptive ε₁ (soft filter)
    γ: float = 0.75  # trust‑score threshold
    λ1: float = 0.25  # instant score if a client **is** flagged this round
    λ2: float = 1.0  # instant score if a client is **not** flagged
    hard_start: int = 10  # start hard filter after this many rounds

    # ───────────────────────── constructor ────────────────────────── #
    def __init__(
            self,
            model_template: torch.nn.Module,
            ultimate_weight: str,
            ultimate_bias: str | None = None,
            device: torch.device | str = "cpu",
    ) -> None:
        self.uw_name = ultimate_weight
        self.ub_name = ultimate_bias
        self.device = torch.device(device)

        # persistent running state
        self.round: int = 0
        self.comp_score: Dict[str, float] = {}  # cumulative compromise
        self.trust_score: Dict[str, float] = {}  # cumulative trust
        self.hist_sim: Dict[Tuple[str, str], torch.Tensor] = {}  # sim matrix

        # template copy (handy, but not strictly required)
        self.model_template = copy.deepcopy(model_template).to(self.device)

    # ================================================================= #
    # =========================== PUBLIC API =========================== #
    # ================================================================= #
    @torch.no_grad()
    def aggregate(
            self,
            global_state: Dict[str, torch.Tensor],
            local_models: List[Tuple[str, torch.nn.Module]],
    ) -> Tuple[
        Dict[str, torch.Tensor],  # new global parameters
        Dict[str, List[str]],  # accepted clients per stage
        List[str],  # blocked (malicious) clients
    ]:
        """Return new averaged parameters + metadata after one FedGrad round."""

        self.round += 1
        cids: List[str] = []

        # -------------------------------------------------------------- #
        # 0)  Extract ultimate‑layer weights and approximate gradients   #
        # -------------------------------------------------------------- #
        W, G = [], []
        for cid, model in local_models:
            st = model.state_dict()
            w = st[self.uw_name].float().to(self.device)
            gw = w - global_state[self.uw_name].float().to(self.device)

            if self.ub_name is not None:
                b = st[self.ub_name].float().to(self.device)
                gb = b - global_state[self.ub_name].float().to(self.device)
                w = torch.cat([w.flatten(), b.flatten()])
                gw = torch.cat([gw.flatten(), gb.flatten()])
            else:
                w = w.flatten()
                gw = gw.flatten()

            W.append(w)
            G.append(gw)
            cids.append(cid)

        W = torch.stack(W)  # (K, D)
        G = torch.stack(G)  # (K, D)
        K = W.size(0)

        # -------------------------------------------------------------- #
        # 1)  Soft filter (compromise score threshold, Eq.‑4)            #
        # -------------------------------------------------------------- #
        W_star = W.mean(0)
        G_star = G.mean(0)

        num = ((W - W_star) * G_star).sum(1)  # ⟨Δw_i, ḡ⟩
        denom = (W - W_star).norm(p=1, dim=1) * (G_star.norm() + 1e-12)  # ‖Δw_i‖₁ ‖ḡ‖
        cst = num / (denom + 1e-12)  # raw compromise
        cst = (cst - cst.min()) / (cst.max() - cst.min() + 1e-12)  # min‑max ⇒ [0,1]

        # running (round‑by‑round) compromise score  ĉ_i,t  (Eq.‑8)
        for s, cid in zip(cst, cids):
            m = self.round if cid in self.comp_score else 1
            prev = self.comp_score.get(cid, 0.0)
            self.comp_score[cid] = (m - 1) / m * prev + s.item() / m

        csi_vec = torch.tensor([self.comp_score[cid] for cid in cids], device=self.device)
        eps1 = min(self.ζ, torch.median(csi_vec).item())
        S1 = {cid for cid, s in zip(cids, csi_vec) if s > eps1}  # flagged by soft filter
        accepted_s1 = set(cids) - S1

        # -------------------------------------------------------------- #
        # 2)  Hard filter (after `hard_start` rounds)                    #
        # -------------------------------------------------------------- #
        S2: set[str] = set()
        accepted_s2: set[str] = set(cids)  # everyone accepted until activated

        if self.round >= self.hard_start:
            # attribute vectors  a_i   (row‑sum of ΔW (+Δb))
            attr = []
            for cid, model in local_models:
                dw = (model.state_dict()[self.uw_name] - global_state[self.uw_name]).float().to(self.device)
                rowsum_w = dw.sum(1)  # (C,)  – OK for Linear; adapt if Conv
                if self.ub_name is not None:
                    db = (model.state_dict()[self.ub_name] - global_state[self.ub_name]).float().to(self.device)
                    vec = torch.cat([rowsum_w, db])
                else:
                    vec = rowsum_w
                attr.append(vec)
            attr = torch.stack(attr)  # (K, D_attr)
            if torch.isnan(attr).any():
                logging.warning("NaN in FedGrad attr vectors — replacing with 0")
                attr = torch.nan_to_num(attr, nan=0.0)

            # cumulative cosine similarity matrix
            # clamp norms to avoid NaN from zero-vector clients
            norms = attr.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            attr_n = attr / norms
            sim_now = attr_n @ attr_n.T
            sim_now = torch.nan_to_num(sim_now, nan=0.0)
            for i in range(K):
                for j in range(i + 1, K):
                    key = (cids[i], cids[j])
                    m = self.round if key in self.hist_sim else 1
                    prev = self.hist_sim.get(key, torch.tensor(0.0, device=self.device))
                    self.hist_sim[key] = (m - 1) / m * prev + sim_now[i, j] / m

            sim_mat = torch.zeros((K, K), device=self.device)
            for i in range(K):
                for j in range(i + 1, K):
                    val = self.hist_sim[(cids[i], cids[j])]
                    if torch.isnan(val):
                        val = torch.tensor(0.0, device=self.device)
                        self.hist_sim[(cids[i], cids[j])] = val
                    sim_mat[i, j] = sim_mat[j, i] = val

            # K‑Means (k=2) on similarity rows
            sim_np = sim_mat.cpu().numpy()
            if np.isnan(sim_np).any():
                logging.warning("NaN in FedGrad sim_mat after hist_sim lookup — replacing with 0")
                sim_np = np.nan_to_num(sim_np, nan=0.0)
            labels = KMeans(n_clusters=2, n_init="auto", max_iter=200).fit_predict(sim_np)

            # estimate benign cluster by closeness score (use low-dim attr
            # vectors instead of raw weights to avoid curse of dimensionality)
            dists = torch.cdist(attr, attr, p=2)
            sizes = [(labels == c).sum() for c in (0, 1)]
            f_est = min(sizes)  # est. # malicious
            k_nn = max(1, K - f_est - 2)
            close = torch.topk(dists, k=k_nn, largest=False, dim=1).values.mean(1)
            benign_label = labels[int(torch.argmin(close))]

            S2 = {cid for cid, lab in zip(cids, labels) if lab != benign_label}
            accepted_s2 = set(cids) - S2

        # -------------------------------------------------------------- #
        # 3)  Combine filters & update trust scores                      #
        # -------------------------------------------------------------- #
        flagged = S1.union(S2)
        malicious_final: set[str] = set()

        for cid in cids:
            inst = self.λ1 if cid in flagged else self.λ2
            m = self.round if cid in self.trust_score else 1
            prev = self.trust_score.get(cid, self.λ2)
            self.trust_score[cid] = (m - 1) / m * prev + inst / m
            if cid in flagged and self.trust_score[cid] < self.γ:
                malicious_final.add(cid)

        # -------------------------------------------------------------- #
        # 4)  FedAvg on benign clients (FIXED DTYPE HANDLING)            #
        # -------------------------------------------------------------- #
        benign = [(cid, mdl) for cid, mdl in local_models if cid not in malicious_final]
        if not benign:  # fall back: no benign detected
            benign = local_models
            malicious_final.clear()

        # ✅ FIXED: Initialize with proper dtype preservation
        new_state = {}
        for k, p in global_state.items():
            # Store original dtype
            original_dtype = p.dtype

            # Initialize accumulator in float32 for averaging
            new_state[k] = torch.zeros_like(p, dtype=torch.float32)

            # Accumulate benign client parameters
            for _, mdl in benign:
                new_state[k] += mdl.state_dict()[k].float().to(self.device)

            # Average
            new_state[k] = new_state[k] / len(benign)

            # ✅ Cast back to original dtype
            if original_dtype in [torch.long, torch.int64, torch.int32, torch.int]:
                # Round for integer types to prevent precision loss
                new_state[k] = new_state[k].round().to(original_dtype)
            else:
                # Direct cast for float types (float16, float32, bfloat16, etc.)
                new_state[k] = new_state[k].to(original_dtype)

        return new_state, {
            "soft": sorted(accepted_s1),
            "hard": sorted(accepted_s2),
            "final": sorted([cid for cid, _ in benign]),
        }, sorted(malicious_final)
