
from __future__ import annotations
import copy
import logging
from typing import Dict, List, Tuple, Any, Optional, Sequence

import torch
from torch import nn

# --- Aggregators / defenses ---------------------------------------------------
from aggregation.aggregator import FedAvgAggregator, WeightedFedAvgAggregator
from aggregation.flame import flame
from aggregation.spmc import spmc
from aggregation.flshield import flshield
from aggregation.foolsgold import FoolsGoldAggregator
from aggregation.alignins import AlignIns
from aggregation.fedgrad import FedGradAggregator
from aggregation.fedsurrogate import FedSurrogate
from aggregation.snowball import snowball
from utils.metrics import print_detection_metrics

# --- Types -------------------------------------------------------------------
ModelState = Dict[str, torch.Tensor]
LocalModels = List[Tuple[int, nn.Module]]  # enforce int IDs for consistency


def _normalize_weights(weights: Sequence[float]) -> List[float]:
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("Sum of client weights must be > 0.")
    return [float(w) / total for w in weights]


def _to_device_like(model: nn.Module, other: nn.Module) -> nn.Module:
    return model.to(next(other.parameters()).device)


class Server:
    """
    Federated-learning server with pluggable defenses.


    """

    def __init__(
            self,
            model: nn.Module,
            *,
            aggregator: Optional[object] = None,
            defense: Optional[str] = None,
            defense_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.global_model = model
        self.defense = (defense or "none").lower()
        self.defense_params = defense_params or {}

        self.last_trusted: List[int] = []
        self.last_anomalous: List[int] = []

        device = next(model.parameters()).device

        # Initialize defense helpers (only what's needed)
        if self.defense == "foolsgold":
            self.fg = FoolsGoldAggregator(
                model_template=model,
                selected_layers=self.defense_params.get("selected_layers"),
                recompute_mask=self.defense_params.get("recompute_mask", True),
                device=device,
            )
        elif self.defense == "fedgrad":
            # Passed in from builder to keep state across rounds
            self.fg: FedGradAggregator = self.defense_params["aggregator"]

        # Default aggregator (plain FedAvg over full parameter states)
        self.aggregator = aggregator or FedAvgAggregator()

    # ------------------------------- API --------------------------------------
    def get_global_model(self) -> nn.Module:
        return self.global_model

    def distribute_global_model(self, clients: list) -> None:
        """
        Push the current global model weights to all clients and snapshot
        it on MaliciousClients for Neurotoxin / MR delta computation.

        MUST be called at the START of each round, BEFORE client.train().

        This is the fix for Neurotoxin being skipped — without this call,
        `initial_global_model` is always None on MaliciousClient and
        Neurotoxin masking is never applied.

        Example round loop:
            server.distribute_global_model(selected_clients)
            for client in selected_clients:
                client.train()
            updates = [c.get_model_update() for c in selected_clients]
            server.aggregate(updates, weights=..., client_ids=...)
        """
        global_state = self.global_model.state_dict()
        for client in clients:
            # Push global weights into each client's local model
            client.model.load_state_dict(
                {k: v.clone() for k, v in global_state.items()}, strict=False
            )
            # Snapshot for Neurotoxin / MR delta (only used by MaliciousClient)
            if hasattr(client, "set_initial_global_model"):
                client.set_initial_global_model(global_state)

        logging.debug(
            "distribute_global_model → %d clients (%d malicious snapshots)",
            len(clients),
            sum(1 for c in clients if hasattr(c, "set_initial_global_model")),
        )

    def aggregate(
            self,
            client_updates: List[ModelState],
            weights: Optional[List[float]] = None,
            *,
            client_ids: Optional[List[int]] = None,
            local_models: Optional[List[Tuple[int, nn.Module]]] = None,
    ) -> None:
        """
        Aggregate client FULL PARAMETER states and update the global model.

        Parameters
        ----------
        client_updates : list[state_dict]
            Each update is a FULL parameter `state_dict` (absolute weights).
        weights : list[float], optional
            Per-client FedAvg weights (e.g., dataset sizes). If None, unweighted.
        client_ids : list[int], optional
            Stable IDs for this round (required by some defenses).
        local_models : list[(int, nn.Module)], optional
            Real models (needed by defenses like FoolsGold/FedSurrogate/Snowball).
            If provided, IDs MUST correspond to `client_ids`.
        """

        # Ensure we have client_ids for tracking
        if client_ids is None:
            client_ids = list(range(len(client_updates)))

        # --------------------------- FLAME -------------------------------------
        if self.defense == "flame":
            aggregated, admitted_indices = flame(
                global_state=self.global_model.state_dict(),
                client_updates=client_updates,
                epsilon=self.defense_params.get("epsilon", 3705),
                delta=self.defense_params.get("delta", 1e-5),
            )

            # ⭐ Track trusted clients
            self.last_trusted = [client_ids[i] for i in admitted_indices]
            all_clients = set(client_ids)
            self.last_anomalous = list(all_clients - set(self.last_trusted))

            self._copy_into_global(aggregated)

            # ⭐ Logging and metrics
            logging.info("🔥 FLAME: Admitted %d/%d clients",
                         len(self.last_trusted), len(client_ids))
            logging.info("  ✅ Admitted: %s", sorted(self.last_trusted))
            if self.last_anomalous:
                logging.info("  🚫 Rejected: %s", sorted(self.last_anomalous))

            if "malicious_clients" in self.defense_params:
                print_detection_metrics(
                    trusted=self.last_trusted,
                    malicious_clients=self.defense_params["malicious_clients"],
                    total_clients=len(client_ids),
                    method_name="FLAME"
                )
            return

        # ------------------------ AlignIns ------------------------------------
        if self.defense == "alignins":
            if local_models is None or client_ids is None:
                raise ValueError("AlignIns requires local_models and client_ids.")

            # Enforce deterministic order by ID
            sorted_models = list(local_models)
            sorted_models.sort(key=lambda t: int(t[0]))
            sorted_ids = [int(cid) for cid, _ in sorted_models]

            # Prepare sample sizes (if needed by your setup)
            if weights is None:
                client_sample_sizes = None
            else:
                id_to_weight = {int(cid): w for cid, w in zip(client_ids, weights)}
                client_sample_sizes = {cid: id_to_weight[cid] for cid in sorted_ids}

            # Run AlignIns defense - NOW RETURNS 3 VALUES
            aggregated_state, selected_clients, detection_metrics = AlignIns(
                local_models=sorted_models,
                global_model=self.global_model,
                client_sample_sizes=client_sample_sizes,
                **self.defense_params
            )

            # Update global model
            self._copy_into_global(aggregated_state)

            # ⭐ Track selected clients for metrics
            self.last_trusted = selected_clients
            all_clients = set(sorted_ids)
            self.last_anomalous = list(all_clients - set(selected_clients))

            logging.info("🔵 AlignIns: Selected %d/%d clients",
                         len(selected_clients), len(sorted_ids))
            logging.info("  ✅ Selected: %s", sorted(selected_clients))
            if self.last_anomalous:
                logging.info("  🚫 Rejected: %s", sorted(self.last_anomalous))

            # Print detection metrics if malicious clients are known
            if "malicious_clients" in self.defense_params:
                print_detection_metrics(
                    trusted=selected_clients,
                    malicious_clients=self.defense_params["malicious_clients"],
                    total_clients=len(sorted_ids),
                    method_name="AlignIns"
                )
            return

        # ------------------------ FedSurrogate ---------------------------------
        if self.defense == "fedsurrogate":
            if local_models is None or client_ids is None or weights is None:
                raise ValueError("FedSurrogate requires local_models, client_ids, and weights.")

            # Enforce deterministic order by ID
            zipped = list(zip(client_ids, local_models, weights))
            zipped.sort(key=lambda t: int(t[0]))
            client_ids, local_models, weights = (
                [int(cid) for cid, _, _ in zipped],
                [lm for _, lm, _ in zipped],
                [float(w) for _, _, w in zipped],
            )

            helper = self.defense_params["helper"]

            trusted, flagged, repl_map, shrink = helper(
                local_models=local_models,
                global_model=self.global_model,
                client_sample_sizes=weights,
                selected_layers=self.defense_params.get("selected_layers"),
                ultimate_weight=self.defense_params["ultimate_weight"],
                ultimate_bias=self.defense_params["ultimate_bias"],
                min_cluster_frac=self.defense_params.get("min_cluster_frac", 0.5),
                zeta=self.defense_params.get("zeta", 0.5),
                shrink_soft=self.defense_params.get("shrink_soft", 0.7),
                shrink_replace=self.defense_params.get("shrink_replace", 0.3),
                enable_lca=self.defense_params.get("enable_lca", True),
                lca_mode=self.defense_params.get("lca_mode", "directional"),
                lca_top_k=self.defense_params.get("lca_top_k", 5),
                enable_rescue=self.defense_params.get("enable_rescue", True),
                enable_replace=self.defense_params.get("enable_replace", True),
                replace_mode=self.defense_params.get("replace_mode", "full"),
                rescue_layer_mode=self.defense_params.get("rescue_layer_mode", "lca"),
                device=next(self.global_model.parameters()).device,
            )

            # Build final state_dict list + weights (avoid passing nn.Module objects)
            cid2w = {cid: w for cid, w in zip(client_ids, weights)}
            id2mdl = {cid: mdl for cid, mdl in local_models}
            trusted_set = set(trusted)

            final_sds: List[Dict[str, torch.Tensor]] = []
            final_weights: List[float] = []

            for cid in client_ids:
                if cid in trusted_set:
                    final_sds.append(id2mdl[cid].state_dict())
                    final_weights.append(cid2w[cid] * float(shrink.get(cid, 1.0)))
                elif cid in repl_map:
                    replacement_info = repl_map[cid]
                    if isinstance(replacement_info, list):
                        # Multi-donor: [(donor_id, weight), ...]
                        for donor_id, donor_weight in replacement_info:
                            if donor_id not in id2mdl:
                                continue
                            final_sds.append(id2mdl[donor_id].state_dict())
                            final_weights.append(cid2w[cid] * float(shrink.get(cid, 0.4)) * donor_weight)
                    else:
                        # Single-donor: just donor_id
                        donor = int(replacement_info)
                        if donor not in id2mdl:
                            continue
                        final_sds.append(id2mdl[donor].state_dict())
                        final_weights.append(cid2w[cid] * float(shrink.get(cid, 0.4)))

            final_weights = _normalize_weights(final_weights)
            new_state = FedAvgAggregator.weighted_average(final_sds, final_weights)
            self.global_model.load_state_dict(new_state, strict=False)

            # ⭐ Track trusted clients
            self.last_trusted = list(map(int, trusted))
            return

        # ------------------------------ SPMC -----------------------------------
        if self.defense == "spmc":
            aggregated, elastic_weights = spmc(
                global_state=self.global_model.state_dict(),
                client_updates=client_updates,
                weights=weights,
            )

            # Threshold-based detection for TPR/FPR metrics
            # Clients with elastic weight < threshold * mean_weight are flagged
            spmc_threshold = float(self.defense_params.get("spmc_threshold", 0.5))
            mean_w = float(elastic_weights.mean())
            weight_threshold = spmc_threshold * mean_w

            self.last_spmc_weights = {
                cid: float(w) for cid, w in zip(client_ids, elastic_weights)
            }
            self.last_trusted = [
                int(cid) for cid, w in zip(client_ids, elastic_weights)
                if w >= weight_threshold
            ]
            self.last_anomalous = [
                int(cid) for cid, w in zip(client_ids, elastic_weights)
                if w < weight_threshold
            ]

            self._copy_into_global(aggregated)

            logging.info("SPMC: Aggregated %d clients", len(client_ids))
            logging.info("  Weights: %s", self.last_spmc_weights)
            logging.info("  Threshold: %.6f (%.2f * mean %.6f)",
                         weight_threshold, spmc_threshold, mean_w)
            logging.info("  Trusted (>= threshold): %s (%d/%d)",
                         sorted(self.last_trusted),
                         len(self.last_trusted), len(client_ids))
            if self.last_anomalous:
                logging.info("  Flagged (< threshold): %s (%d/%d)",
                             sorted(self.last_anomalous),
                             len(self.last_anomalous), len(client_ids))

            if "malicious_clients" in self.defense_params:
                print_detection_metrics(
                    trusted=self.last_trusted,
                    malicious_clients=self.defense_params["malicious_clients"],
                    total_clients=len(client_ids),
                    method_name="SPMC",
                )
            return

        # ----------------------------- FLShield --------------------------------
        if self.defense == "flshield":
            val_loader = self.defense_params.get("val_loader")
            if val_loader is None:
                raise ValueError("FLShield requires `val_loader` in defense_params.")

            device = next(self.global_model.parameters()).device
            self._flshield_round = getattr(self, "_flshield_round", 0)

            aggregated, accepted_indices = flshield(
                global_state=self.global_model.state_dict(),
                client_updates=client_updates,
                global_model=self.global_model,
                val_loader=val_loader,
                num_classes=int(self.defense_params.get("num_classes", 10)),
                device=device,
                weights=weights,
                start_round=int(self.defense_params.get("start_round", 1)),
                current_round=self._flshield_round,
            )
            self._flshield_round += 1

            self.last_trusted = [client_ids[i] for i in accepted_indices]
            all_clients = set(client_ids)
            self.last_anomalous = list(all_clients - set(self.last_trusted))

            self._copy_into_global(aggregated)

            logging.info("FLShield: Accepted %d/%d clients",
                         len(self.last_trusted), len(client_ids))
            logging.info("  Accepted: %s", sorted(self.last_trusted))
            if self.last_anomalous:
                logging.info("  Rejected: %s", sorted(self.last_anomalous))

            if "malicious_clients" in self.defense_params:
                print_detection_metrics(
                    trusted=self.last_trusted,
                    malicious_clients=self.defense_params["malicious_clients"],
                    total_clients=len(client_ids),
                    method_name="FLShield",
                )
            return

        # -------------------------- FoolsGold ----------------------------------
        if self.defense == "foolsgold":
            # Build local model objects from full states (deterministic order)
            local_models_fg: List[Tuple[str, nn.Module]] = []

            # Zip client_ids with updates for proper tracking
            id_update_pairs = list(zip(client_ids, client_updates))
            id_update_pairs.sort(key=lambda x: int(x[0]))

            for cid, upd in id_update_pairs:
                mdl = copy.deepcopy(self.global_model)
                mdl.load_state_dict(upd, strict=False)
                local_models_fg.append((str(int(cid)), _to_device_like(mdl, self.global_model)))

            # Call FoolsGold aggregator
            new_state, fg_weights = self.fg.aggregate(
                global_state=self.global_model.state_dict(),
                local_models=local_models_fg,
            )

            # ⭐ Convert FoolsGold weights to trusted clients list
            threshold = 0.5
            self.last_trusted = [
                int(cid) for cid, weight in fg_weights.items()
                if weight >= threshold
            ]
            self.last_anomalous = [
                int(cid) for cid, weight in fg_weights.items()
                if weight < threshold
            ]

            # Store raw weights for detailed analysis
            self.last_foolsgold_weights = fg_weights

            # Update global model
            self._copy_into_global(new_state)

            # ⭐ Detailed logging
            logging.info("🟢 FoolsGold weights: %s", fg_weights)
            logging.info("  Trusted (≥%.2f): %s (%d/%d)",
                         threshold, sorted(self.last_trusted),
                         len(self.last_trusted), len(fg_weights))
            if self.last_anomalous:
                logging.info("  Rejected (<%.2f): %s (%d/%d)",
                             threshold, sorted(self.last_anomalous),
                             len(self.last_anomalous), len(fg_weights))

            if "malicious_clients" in self.defense_params:
                print_detection_metrics(
                    trusted=self.last_trusted,
                    malicious_clients=self.defense_params["malicious_clients"],
                    total_clients=len(fg_weights),
                    method_name="FoolsGold"
                )
            return

        # --------------------------- FedGrad -----------------------------------
        if self.defense == "fedgrad":
            if len(client_ids) != len(client_updates):
                raise ValueError("`client_ids` length must match `client_updates`.")

            # Build local models with persistent IDs, deterministic order
            local_models_fg: List[Tuple[str, nn.Module]] = []
            for cid, upd in zip(client_ids, client_updates):
                mdl = copy.deepcopy(self.global_model)
                mdl.load_state_dict(upd, strict=False)
                local_models_fg.append((str(int(cid)), _to_device_like(mdl, self.global_model)))

            local_models_fg.sort(key=lambda x: x[0])

            new_state, accepted, blocked = self.fg.aggregate(
                global_state=self.global_model.state_dict(),
                local_models=local_models_fg,
            )

            # ⭐ Track trusted clients
            round_nr = self.fg.round
            logging.info("🛡️ FedGrad round %d", round_nr)
            logging.info("  ✅ soft  : %s", accepted["soft"])
            logging.info("  ✅ hard  : %s", accepted["hard"])
            logging.info("  ✅ final : %s", accepted["final"])
            logging.info("  🚫 blocked: %s", blocked)

            self.last_trusted = [int(cid) for cid in accepted["final"] if str(cid).isdigit()]
            self.last_anomalous = [int(cid) for cid in blocked if str(cid).isdigit()]

            self._copy_into_global(new_state)

            if "malicious_clients" in self.defense_params:
                print_detection_metrics(
                    trusted=self.last_trusted,
                    malicious_clients=self.defense_params["malicious_clients"],
                    total_clients=len(client_updates),
                    method_name="FedGrad"
                )
            return

        # ----------------------------- Snowball ---------------------------------
        if self.defense == "snowball":
            if local_models is None or client_ids is None:
                raise ValueError("Snowball requires local_models and client_ids.")

            sorted_models = sorted(local_models, key=lambda t: int(t[0]))

            self._snowball_round = getattr(self, "_snowball_round", 0)

            aggregated, accepted_ids = snowball(
                global_state=self.global_model.state_dict(),
                local_models=sorted_models,
                cur_round=self._snowball_round,
                weights=weights,
                layer_filters=self.defense_params.get("layer_filters"),
                ct=int(self.defense_params.get("ct", 10)),
                vt=float(self.defense_params.get("vt", 0.5)),
                v_step=float(self.defense_params.get("v_step", 0.05)),
                vae_initial=int(self.defense_params.get("vae_initial", 270)),
                vae_tuning=int(self.defense_params.get("vae_tuning", 30)),
                vae_hidden=int(self.defense_params.get("vae_hidden", 256)),
                vae_latent=int(self.defense_params.get("vae_latent", 64)),
                warmup_rounds=int(self.defense_params.get("warmup_rounds", 100)),
                device=str(next(self.global_model.parameters()).device),
            )
            self._snowball_round += 1

            # snowball() returns actual client IDs, not indices
            self.last_trusted = list(accepted_ids)
            all_clients = set(client_ids)
            self.last_anomalous = list(all_clients - set(self.last_trusted))

            self._copy_into_global(aggregated)

            logging.info("Snowball: Accepted %d/%d clients",
                         len(self.last_trusted), len(client_ids))
            logging.info("  Accepted: %s", sorted(self.last_trusted))
            if self.last_anomalous:
                logging.info("  Rejected: %s", sorted(self.last_anomalous))

            if "malicious_clients" in self.defense_params:
                print_detection_metrics(
                    trusted=self.last_trusted,
                    malicious_clients=self.defense_params["malicious_clients"],
                    total_clients=len(client_ids),
                    method_name="Snowball",
                )
            return

        # --------------------------- Oracle ------------------------------------
        if self.defense == "oracle":
            mal_set = set(self.defense_params.get("malicious_clients", []))
            benign_idx = [i for i, cid in enumerate(client_ids) if cid not in mal_set]

            self.last_trusted = [client_ids[i] for i in benign_idx]
            self.last_anomalous = [cid for cid in client_ids if cid in mal_set]

            benign_updates = [client_updates[i] for i in benign_idx]
            benign_weights = [weights[i] for i in benign_idx] if weights else None

            if benign_updates:
                aggregated = WeightedFedAvgAggregator().aggregate(benign_updates, benign_weights)
            else:
                aggregated = self.global_model.state_dict()

            self._copy_into_global(aggregated)

            logging.info("🔮 Oracle: Removed %d malicious, aggregated %d benign clients",
                         len(self.last_anomalous), len(self.last_trusted))

            if mal_set:
                print_detection_metrics(
                    trusted=self.last_trusted,
                    malicious_clients=list(mal_set),
                    total_clients=len(client_ids),
                    method_name="Oracle"
                )
            return

        # ------------------------- No defense (FedAvg) -------------------------
        # Track all clients as "trusted" (no defense)
        self.last_trusted = list(client_ids)
        self.last_anomalous = []

        if weights is None:
            aggregated = self.aggregator.aggregate(client_updates, None)
        else:
            aggregated = WeightedFedAvgAggregator().aggregate(client_updates, weights)

        self._copy_into_global(aggregated)

        # ⭐ Logging
        logging.info("📊 FedAvg (No Defense): Aggregating all %d clients", len(client_ids))

        # Optional: Print metrics for comparison
        if "malicious_clients" in self.defense_params:
            print_detection_metrics(
                trusted=self.last_trusted,
                malicious_clients=self.defense_params["malicious_clients"],
                total_clients=len(client_ids),
                method_name="FedAvg (No Defense)"
            )

    # -------------------------- internals -------------------------------------
    def _copy_into_global(self, aggregated_params: ModelState) -> None:
        """Copy FULL parameters into the global model (handles device moves)."""
        with torch.no_grad():
            gstate = self.global_model.state_dict()
            missing = [k for k in gstate.keys() if k not in aggregated_params]
            if missing:
                raise KeyError(f"Missing keys in aggregated params: {missing[:5]} ...")
            for name, tensor in aggregated_params.items():
                # move incoming tensor to the param's device before copy
                gstate[name].copy_(tensor.to(gstate[name].device))

