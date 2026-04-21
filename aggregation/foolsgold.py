"""
foolsgold.py – An implementation of Algorithm 1 from
"FoolsGold: (Fung et al., 2020).


"""

from __future__ import annotations
import copy
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F


class FoolsGoldAggregator:
    # ----------------------------- ctor ---------------------------------- #
    def __init__(
            self,
            model_template: torch.nn.Module,
            selected_layers: List[str] | None = None,
            recompute_mask: bool = True,
            device: torch.device | str = "cpu",
    ):
        self.selected_layers = selected_layers
        self.recompute_mask = recompute_mask
        self.device = torch.device(device)
        self.model_template = copy.deepcopy(model_template).to(self.device)

        # persistent state
        self.histories: Dict[str, torch.Tensor] = {}  # running H_i vectors
        self.mask: torch.Tensor | None = None  # feature-importance mask

    # -------------------------- public API -------------------------------- #
    @torch.no_grad()
    def aggregate(
            self,
            global_state: Dict[str, torch.Tensor],
            local_models: List[Tuple[str, torch.nn.Module]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Args
        ----
            global_state : θ_g  (state_dict)
            local_models : list of (cid, nn.Module) after local training

        Returns
        -------
            new_global_state : FedAVG weighted by FoolsGold w_i
            fg_weights       : {cid: w_i}  (for logging / debugging)
        """
        # 0) build / refresh feature mask
        if self.recompute_mask or self.mask is None:
            self.mask = self._feature_mask(global_state)

        # 1) update client histories
        cids, H = [], []
        for cid, model in local_models:
            Δ = self._flatten_update(model.state_dict(), global_state)
            Δ.mul_(self.mask)

            self.histories[cid] = self.histories.get(cid, torch.zeros_like(Δ)) + Δ

            cids.append(cid)
            H.append(self.histories[cid])

        H = torch.stack(H, 0)  # shape (n, D)
        n = H.size(0)

        # 2) pair-wise cosine similarities
        S = F.normalize(H, dim=1) @ F.normalize(H, dim=1).t()
        S.fill_diagonal_(0.0)  # ignore self-similarity

        # 3) pardoning
        maxS = S.max(1).values
        for i in range(n):
            for j in range(n):
                if maxS[j] > maxS[i] and maxS[i] > 0:
                    S[i, j] *= maxS[i] / (maxS[j] + 1e-12)

        # 4) draft weights v = 1 – max S(i)
        v = 1.0 - S.max(1).values

        # 5) logit re-scale & clip
        v = v / (v.max() + 1e-12)
        v = torch.log(v / (1.0 - v + 1e-12) + 1e-12) + 0.5
        w = torch.clamp(v, 0.0, 1.0)

        fg_weights = {cid: float(wi) for cid, wi in zip(cids, w)}

        # 6) weighted update aggregation (paper Alg.1 line 26):
        #    θ_{t+1} = θ_t + Σ_i (α_i / Σα) · Δ_{i,t}
        #    Normalize weights so update magnitude doesn't scale with n.
        if w.sum().item() == 0:
            w = torch.ones_like(w) / n
        else:
            w = w / w.sum()

        new_state: Dict[str, torch.Tensor] = {}
        for key in global_state.keys():
            original_dtype = global_state[key].dtype
            g = global_state[key].float().to(self.device)

            weighted_delta = torch.stack([
                (local_models[i][1].state_dict()[key].float().to(self.device) - g) * w[i]
                for i in range(n)
            ], dim=0).sum(0)

            aggregated = g + weighted_delta

            # Cast back to original dtype
            if original_dtype in [torch.long, torch.int64, torch.int32, torch.int]:
                new_state[key] = aggregated.round().to(original_dtype).to(global_state[key].device)
            else:
                new_state[key] = aggregated.to(original_dtype).to(global_state[key].device)

        return new_state, fg_weights

    # --------------------------- helpers ---------------------------------- #
    def _flatten_update(
            self,
            local_state: Dict[str, torch.Tensor],
            global_state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Flatten Δ_i over selected layers with dtype handling.

        Args:
            local_state: Client model state dict
            global_state: Global model state dict

        Returns:
            Flattened update vector (excluding integer parameters)
        """
        parts = []
        for k, g in global_state.items():
            if self.selected_layers is None or k in self.selected_layers:
                # ✅ Skip integer parameters (they don't have gradients)
                if g.dtype in [torch.long, torch.int64, torch.int32, torch.int,
                               torch.int16, torch.int8, torch.uint8, torch.bool]:
                    # Add zeros for integer parameters to maintain shape consistency
                    parts.append(torch.zeros(g.numel(), device=self.device))
                else:
                    # Compute difference for float parameters
                    delta = (local_state[k].float() - g.float()).flatten().to(self.device)
                    parts.append(delta)

        return torch.cat(parts)

    def _feature_mask(self, global_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Simple L2 importance per parameter with dtype handling.

        Args:
            global_state: Global model state dict

        Returns:
            Feature importance mask (normalized to max=1)
        """
        norms = []
        for k, p in global_state.items():
            if self.selected_layers is None or k in self.selected_layers:
                # ✅ Handle integer parameters
                if p.dtype in [torch.long, torch.int64, torch.int32, torch.int,
                               torch.int16, torch.int8, torch.uint8, torch.bool]:
                    # Use constant small value for integer parameters
                    l2 = 1e-12
                else:
                    # Compute L2 norm for float parameters
                    l2 = p.float().norm().item() + 1e-12

                norms.append(torch.full((p.numel(),), l2, device=self.device))

        mask = torch.cat(norms)
        return mask / (mask.max() + 1e-12)  # normalize ≤ 1
