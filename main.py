# main.py  –  Clean & modular entry-point for YAML-based FL experiments
# =========================================================================

from __future__ import annotations

import argparse
import sys
import copy
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from utils.metrics import detection_summary_over_rounds
from aggregation.alignins import AlignIns



import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


# ─────────────────── project-local imports ──────────────────── #
from aggregation.aggregator import FedAvgAggregator, WeightedFedAvgAggregator
from client.client import BaseClient, BenignClient, MaliciousClient   # ← include BaseClient
from data.load_data import load_dataset, partition_data
from data.utils import calculate_class_distribution
from models.cifarnet import CIFAR10Model, ResNet8, ResNet18
from models.simple_cnn import SimpleCNN
from server import Server
from utils.backdoor_utils import apply_trigger_batch, add_semantic_trigger, add_a3fl_trigger, add_pgd_trigger
from utils.metrics import (
    evaluate_backdoor,
    evaluate_model,
    log_evaluation_results,
    print_detection_metrics,
    evaluate_per_class,
)
from utils.visualization import (
    plot_and_save_class_distribution,
    plot_evaluation_metrics,
    plot_individual_distributions,
    # save_backdoored_images,
)

python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
print(f"\nPython: {python_version}")

print(f"PyTorch Version: {torch.__version__}")

print(f"CUDA Version: {torch.version.cuda}")

# ╭──────────────────────────── CLI & SEED ───────────────────────────╮

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run FL experiment from YAML")
    p.add_argument("--config", default="config/default_config.yaml",
                   help="Path to YAML config file")
    p.add_argument("--seed", type=int, default=69, help="Random seed")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True is safe when input sizes are fixed (e.g. CIFAR 32x32)
    # and gives a large speedup on convolutions
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    logging.info(f"Random seed set to {seed}")


# ╭───────────────────────── CONFIG HELPERS ─────────────────────────╮

def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def initialise_model(name: str, dataset_name: str) -> torch.nn.Module:
    num_classes = 100 if dataset_name.lower() == "cifar100" else 10
    if name.lower() == "simple_cnn":
        return SimpleCNN()
    if name.lower() == "cifar10model":
        return CIFAR10Model()
    if name.lower() == "resnet8":
        return ResNet8(num_classes=num_classes)
    if name.lower() == "resnet18":
        return ResNet18(num_classes=num_classes)

    raise ValueError(f"Unknown model '{name}'")


# ╭──────────────────────── CLIENT FACTORY ──────────────────────────╮

def build_clients(
    datasets: List[torch.utils.data.Dataset],
    base_model: torch.nn.Module,
    cfg: Dict[str, Any],
) -> List[BaseClient]:
    """
    Instantiate Benign/Malicious clients based on YAML config.

    Notes:
    - Do NOT bake MR scale into client_settings; we set per-round fields later.
    - You can toggle MR and scale mode per-client via YAML fields read inside client.py:
        client_settings:
          model_replacement: true
          mr_scale_mode: "weight"  # or "count"
    """
    # cs = cfg["client_settings"]
    
    cs = copy.deepcopy(cfg["client_settings"])
    cs["dataset_name"] = cfg["dataset_name"]
    cs["model"] = cfg["model"]
    n_mal = int(cfg.get("num_malicious", 0))
    explicit = cfg.get("malicious_clients") or []

    if not explicit:
        explicit = random.sample(range(len(datasets)), n_mal)
        logging.info(f"[AUTO] Malicious clients = {explicit}")
    else:
        logging.info(f"[CFG]  Malicious clients = {explicit}")

    # Persist resolved list so downstream metrics/logging can find it
    cfg["malicious_clients"] = explicit

    clients: List[BaseClient] = []
    for cid, ds in enumerate(datasets):
        model_copy = copy.deepcopy(base_model)
        if cid in explicit:
            clients.append(MaliciousClient(cid, ds, model_copy, cs,
                                           attack_type=cfg.get("attack_type", "backdoor")))
        else:
            clients.append(BenignClient(cid, ds, model_copy, cs))
    return clients


class TrustedClientsTracker:
    """Track trusted clients across all rounds for cumulative metrics"""

    def __init__(self):
        self.all_trusted_lists = []

    def add_round(self, trusted_clients):
        """Store trusted clients for current round"""
        self.all_trusted_lists.append(list(trusted_clients))

    def get_all_trusted_lists(self):
        """Return all tracked trusted client lists"""
        return self.all_trusted_lists


# Initialize tracker before training loop
trusted_tracker = TrustedClientsTracker()


# ╭──────────────────────── CLIENT SELECTION ───────────────────────╮

def select_clients(
    clients: List[BaseClient],
    cfg: Dict[str, Any],
    rnd: int,
) -> List[BaseClient]:
    """
    Cross-device client selection: sample a subset of clients each round.

    Modes for malicious_participation:
      - "proportional": malicious clients are sampled at the same rate as
        benign. E.g. 10/20 selected → ~2/4 malicious expected.
      - "always": all malicious clients always participate (worst-case).
        Remaining slots filled by random benign clients.
      - "random": all clients (benign + malicious) sampled uniformly.
    """
    sel_cfg = cfg.get("client_selection", {})
    if not sel_cfg.get("enabled", False):
        return clients  # full participation

    k = int(sel_cfg.get("clients_per_round", len(clients)))
    k = min(k, len(clients))
    mode = sel_cfg.get("malicious_participation", "proportional").lower()

    mal_ids = set(cfg.get("malicious_clients", []))
    benign = [c for c in clients if c.client_id not in mal_ids]
    malicious = [c for c in clients if c.client_id in mal_ids]

    if mode == "always":
        # All malicious participate; fill remaining slots with random benign
        selected = list(malicious)
        remaining = k - len(selected)
        if remaining > 0:
            selected += random.sample(benign, min(remaining, len(benign)))
    elif mode == "random":
        # Uniform random from all clients
        selected = random.sample(clients, k)
    else:
        # "proportional" — sample benign and malicious at the same rate
        rate = k / len(clients)
        n_mal = max(0, round(len(malicious) * rate))
        n_ben = k - n_mal
        n_mal = min(n_mal, len(malicious))
        n_ben = min(n_ben, len(benign))
        selected = random.sample(malicious, n_mal) + random.sample(benign, n_ben)

    random.shuffle(selected)
    sel_ids = sorted([c.client_id for c in selected])
    mal_in_round = [c.client_id for c in selected if c.client_id in mal_ids]
    logging.info("Client selection: %d/%d clients | malicious in round: %s",
                 len(selected), len(clients), mal_in_round)
    return selected


# ╭──────────────────────── DEFENCE / SERVER ────────────────────────╮

"""
Fixed build_server() - Add malicious_clients to all defenses
Modified: November 14, 2025, 12:15 PM PST
Lines ~147-201
"""


def build_server(model: torch.nn.Module, cfg: Dict[str, Any],
                 n_clients: int) -> Server:
    def_cfg = cfg.get("defense", {})

    # Get malicious clients list for all defenses
    malicious_clients = cfg.get("malicious_clients", [])

    if not def_cfg.get("enabled", False) or def_cfg.get("type", "none") == "none":
        agg_name = cfg.get("aggregator", {}).get("type", "fedavg").lower()
        aggregator = WeightedFedAvgAggregator() if agg_name == "weighted" else FedAvgAggregator()
        return Server(
            model=model,
            aggregator=aggregator,
            defense_params={"malicious_clients": malicious_clients}  # ⭐ Add this
        )

    defence_type = def_cfg["type"].lower()
    device = next(model.parameters()).device

    # In your main training file (e.g., train_fedsurrogate.py)


    # During federated round
    # ═══════════════════════════════════════════════════════
    # ⭐ ADD ALIGNINS HERE (after device assignment)
    # ═══════════════════════════════════════════════════════
    if defence_type == "alignins":
        return Server(
            model=model,
            defense="alignins",
            defense_params={
                "lambda_s": def_cfg.get("lambda_s", 1.0),
                "lambda_c": def_cfg.get("lambda_c", 1.0),
                "sparsity": def_cfg.get("sparsity", 0.3),
                "device": str(device),
                "verbose": def_cfg.get("verbose", True),
                "malicious_clients": malicious_clients,
            }
        )

    if defence_type == "foolsgold":
        from aggregation.foolsgold import FoolsGoldAggregator
        fg = FoolsGoldAggregator(model, def_cfg.get("selected_layers"),
                                 def_cfg.get("recompute_mask", True), device)
        return Server(model=model, defense="foolsgold",
                      defense_params={
                          "aggregator": fg,
                          "malicious_clients": malicious_clients  # ✅ Already there
                      })

    if defence_type == "fedgrad":
        from aggregation.fedgrad import FedGradAggregator
        fg = FedGradAggregator(model, def_cfg["ultimate_weight"],
                               def_cfg.get("ultimate_bias"), device)
        for k in ("ζ", "gamma", "λ1", "λ2", "hard_start"):
            if k in def_cfg:
                setattr(fg, k, def_cfg[k])
        return Server(model=model, defense="fedgrad",
                      defense_params={
                          "aggregator": fg,
                          "malicious_clients": malicious_clients  # ⭐ Add this
                      })

    if defence_type == "fedsurrogate":
        from aggregation.fedsurrogate import FedSurrogate
        return Server(
            model=model,
            defense="fedsurrogate",
            defense_params={
                "helper": FedSurrogate,
                "selected_layers": def_cfg.get("selected_layers"),
                "ultimate_weight": def_cfg["ultimate_weight"],
                "ultimate_bias": def_cfg.get("ultimate_bias"),
                "min_cluster_frac": def_cfg.get("min_cluster_frac", .5),
                "shrink_soft": def_cfg.get("shrink_soft", 0.7),
                "shrink_replace": def_cfg.get("shrink_replace", 0.3),
                "zeta": def_cfg.get("zeta", .4),
                "enable_lca": def_cfg.get("enable_lca", True),
                "lca_mode": def_cfg.get("lca_mode", "l2_norm"),
                "lca_top_k": def_cfg.get("lca_top_k", 5),
                "enable_rescue": def_cfg.get("enable_rescue", True),
                "enable_replace": def_cfg.get("enable_replace", True),
                "replace_mode": def_cfg.get("replace_mode", "full"),
                "rescue_layer_mode": def_cfg.get("rescue_layer_mode", "lca"),
                "hard_start": def_cfg.get("hard_start", 10),
                "malicious_clients": malicious_clients,  # ✅ Already there
            }
        )

    if defence_type == "flame":
        return Server(model=model, defense="flame",
                      defense_params={
                          "epsilon": def_cfg.get("epsilon", 3705),
                          "delta": def_cfg.get("delta", 1e-5),
                          "malicious_clients": malicious_clients
                      })




    if defence_type == "oracle":
        return Server(
            model=model,
            defense="oracle",
            defense_params={
                "malicious_clients": malicious_clients,
            }
        )

    if defence_type == "spmc":
        return Server(
            model=model,
            defense="spmc",
            defense_params={
                "malicious_clients": malicious_clients,
            }
        )

    if defence_type == "flshield":
        import os
        import random as _rnd
        from torch.utils.data import DataLoader, Subset
        test_ds = cfg.get("test_data")
        if test_ds is None:
            raise ValueError("FLShield requires cfg['test_data'] to be set before build_server().")
        val_size = def_cfg.get("val_dataset_size", 200)
        indices = _rnd.sample(range(len(test_ds)), min(val_size, len(test_ds)))
        val_loader = DataLoader(
            Subset(test_ds, indices),
            batch_size=def_cfg.get("val_batch_size", 64),
            shuffle=False,
            num_workers=0 if os.name == "nt" else 2,
        )
        return Server(
            model=model,
            defense="flshield",
            defense_params={
                "val_loader": val_loader,
                "num_classes": def_cfg.get("num_classes",
                                           cfg.get("data_partition", {}).get("num_classes", 10)),
                "start_round": def_cfg.get("start_round", 1),
                "malicious_clients": malicious_clients,
            }
        )

    if defence_type == "snowball":
        return Server(
            model=model,
            defense="snowball",
            defense_params={
                "layer_filters": def_cfg.get("layer_filters"),
                "ct": def_cfg.get("ct", 10),
                "vt": def_cfg.get("vt", 0.5),
                "v_step": def_cfg.get("v_step", 0.05),
                "vae_initial": def_cfg.get("vae_initial", 270),
                "vae_tuning": def_cfg.get("vae_tuning", 30),
                "vae_hidden": def_cfg.get("vae_hidden", 256),
                "vae_latent": def_cfg.get("vae_latent", 64),
                "warmup_rounds": def_cfg.get("warmup_rounds", 100),
                "malicious_clients": malicious_clients,
            }
        )

    raise ValueError(f"Unknown defence type '{defence_type}'")

# ╭──────────────────────────  FL LOOP  ────────────────────────────╮

"""
Fixed run_federated_learning() - Add method_name to metrics
Modified: November 14, 2025, 12:15 PM PST
Lines ~330-340 and ~360-368
"""


def run_federated_learning(clients: List[BaseClient], server: Server, cfg: Dict[str, Any]):
    device = next(server.global_model.parameters()).device
    import os
    _nw = 0 if os.name == 'nt' else 2
    test_loader = DataLoader(cfg["test_data"],
                             batch_size=256,
                             shuffle=False,
                             num_workers=_nw,
                             pin_memory=(device.type == "cuda"),
                             persistent_workers=(_nw > 0))
    rounds = cfg.get("num_rounds", 10)
    hist_acc, hist_ba = [], []
    hist_det = []

    # ⭐ Get defense name for metrics
    defense_name = cfg.get("defense", {}).get("type", "FedAvg").upper()
    if defense_name == "NONE":
        defense_name = "FedAvg (No Defense)"

    for rnd in range(rounds):
        logging.info(f"──────── Round {rnd + 1}/{rounds} ────────")

        # Store for cumulative calculation
        if hasattr(server, "last_trusted"):
            trusted_tracker.add_round(server.last_trusted)

        # 0) client selection (cross-device partial participation)
        selected = select_clients(clients, cfg, rnd)

        # 1) broadcast global params to selected clients only
        global_state = server.get_global_model().state_dict()
        for c in selected:
            c.model.load_state_dict(global_state)
            c.reset_optimizer()

        # 1b) Per-round setup for attacks that need initial global model
        #     (Model Replacement, Constrain-and-Scale, Neurotoxin, A3FL)
        mal_clients = [
            c for c in selected if isinstance(c, MaliciousClient)
        ]
        if mal_clients:
            total_weight = float(sum(len(c.dataset) for c in selected))
            adv_weight = float(sum(len(c.dataset) for c in mal_clients))
            num_clients = int(len(selected))
            num_adversaries = int(len(mal_clients))

            for c in mal_clients:
                c.set_initial_global_model(global_state)
                c.total_weight = total_weight
                c.adv_weight = adv_weight
                c.num_clients = num_clients
                c.num_adversaries = num_adversaries

        # 1c) SPMC: compute and distribute coalition models for aligned gradient
        defense_type_lower = cfg.get("defense", {}).get("type", "none").lower()
        if defense_type_lower == "spmc":
            from aggregation.spmc import compute_coalition_models
            def_cfg = cfg.get("defense", {})
            prev_updates = getattr(server, '_spmc_prev_updates', [])
            prev_weights = getattr(server, '_spmc_prev_weights', None)
            coalition_models = compute_coalition_models(
                client_updates=prev_updates,
                global_model=server.global_model,
                weights=prev_weights,
            )
            for idx, c in enumerate(selected):
                if idx < len(coalition_models):
                    c.coalition_model = coalition_models[idx].to(c.device)
                else:
                    c.coalition_model = copy.deepcopy(server.global_model).to(c.device)
                c._spmc_lamda = float(def_cfg.get("spmc_lamda", 1.0))
                c._spmc_temperature = float(def_cfg.get("spmc_temperature", 1.0))
        else:
            for c in selected:
                c.coalition_model = None

        # 2) local training + collect full states (selected clients only)
        updates, weights = [], []
        per_client_eval = cfg.get("per_client_eval", False)
        for c in selected:
            c.train()

            if per_client_eval:
                acc, loss = evaluate_model(c.model, test_loader)
                local_ba, _ = evaluate_backdoor(
                    c.model,
                    test_loader,
                    cfg["client_settings"].get("backdoor_target", 1),
                    trigger_function=lambda imgs: apply_trigger_batch(
                        imgs,
                        trigger_type=cfg["client_settings"].get("trigger_type", "3x3"),
                        color=tuple(cfg["client_settings"].get("backdoor_color", [1.0, 0.0, 0.0])),
                        position=cfg["client_settings"].get("backdoor_location", "bottom_right"),
                        use_dba=False,
                        client_id=c.client_id,
                    ),
                    device=next(c.model.parameters()).device,
                )
                print(f"Client {c.client_id} – Local Backdoor Acc = {local_ba:.2f}%", flush=True)
                log_evaluation_results(c.client_id, acc, loss)

            updates.append(c.get_model_update())  # FULL parameter states
            weights.append(len(c.dataset))  # FedAvg weights

        # 3) aggregate + defend (only selected clients' updates)
        server.aggregate(
            updates, weights,
            client_ids=[int(c.client_id) for c in selected],
            local_models=[(int(c.client_id), c.model) for c in selected],
        )

        # 3b) SPMC: store this round's updates for next round's coalition computation
        if defense_type_lower == "spmc":
            server._spmc_prev_updates = updates
            server._spmc_prev_weights = weights

        # 4) evaluate global
        g_acc, g_loss = evaluate_model(server.global_model, test_loader)
        log_evaluation_results(None, g_acc, g_loss, is_global=True)
        hist_acc.append(g_acc)

        # 5) backdoor eval — select trigger function based on attack config
        tgt = cfg["client_settings"].get("backdoor_target", 1)
        cs = cfg["client_settings"]
        if cs.get("use_semantic_backdoor", False):
            _trigger_fn = lambda imgs: add_semantic_trigger(
                imgs,
                trigger_type=cs.get("semantic_trigger_type", "green_car"),
                source_class=cs.get("semantic_source_class", 0),
                intensity=cs.get("semantic_intensity", 0.3),
            )
        elif cs.get("use_a3fl", False):
            # A3FL: mask-based blending from any malicious client with an optimised trigger
            _a3fl_client = next(
                (c for c in clients
                 if isinstance(c, MaliciousClient) and getattr(c, "_a3fl_trigger", None) is not None),
                None,
            )
            if _a3fl_client is not None:
                _cached_trigger = _a3fl_client._a3fl_trigger.clone()
                _cached_mask = _a3fl_client._a3fl_mask.clone()
                _trigger_fn = lambda imgs, _t=_cached_trigger, _m=_cached_mask: (
                    add_a3fl_trigger(imgs, _t.to(imgs.device), _m.to(imgs.device))
                )
            else:
                _trigger_fn = lambda imgs: apply_trigger_batch(
                    imgs,
                    trigger_type=cs.get("trigger_type", "3x3"),
                    position=cs.get("backdoor_location", "bottom_right"),
                    color=tuple(cs.get("backdoor_color", [1.0, 0.0, 0.0])),
                    use_dba=False, client_id=0,
                )
        elif cs.get("use_pgd_trigger", False):
            # PGD universal perturbation trigger
            _pgd_client = next(
                (c for c in clients
                 if isinstance(c, MaliciousClient) and getattr(c, "_pgd_trigger", None) is not None),
                None,
            )
            if _pgd_client is not None:
                _cached_delta = _pgd_client._pgd_trigger.clone()
                _trigger_fn = lambda imgs, _d=_cached_delta: (
                    add_pgd_trigger(imgs, _d.to(imgs.device))
                )
            else:
                _trigger_fn = lambda imgs: apply_trigger_batch(
                    imgs,
                    trigger_type=cs.get("trigger_type", "3x3"),
                    position=cs.get("backdoor_location", "bottom_right"),
                    color=tuple(cs.get("backdoor_color", [1.0, 0.0, 0.0])),
                    use_dba=False, client_id=0,
                )
        else:
            _trigger_fn = lambda imgs: apply_trigger_batch(
                imgs,
                trigger_type=cs.get("trigger_type", "3x3"),
                position=cs.get("backdoor_location", "bottom_right"),
                color=tuple(cs.get("backdoor_color", [1.0, 0.0, 0.0])),
                use_dba=False, client_id=0,
            )

        b_acc, _ = evaluate_backdoor(
            server.global_model,
            test_loader,
            tgt,
            trigger_function=_trigger_fn,
            device=next(server.global_model.parameters()).device,
        )
        hist_ba.append(b_acc)
        logging.info(f"Backdoor Acc = {b_acc:.2f}%")

        # ⭐ UPDATED: Print per-round metrics with method name
        if hasattr(server, "last_trusted"):
            det_metrics = print_detection_metrics(
                trusted=server.last_trusted,
                malicious_clients=cfg.get("malicious_clients", []),
                total_clients=cfg.get("num_clients", len(clients)),
                round_idx=rnd,
                method_name=defense_name
            )
            hist_det.append(det_metrics)

       # ── final summary ──
    final_acc = hist_acc[-1] if hist_acc else None
    final_ba = hist_ba[-1] if hist_ba else None

    logging.info("=" * 60)
    logging.info("Final summary")

    if final_acc is not None:
        logging.info("Final Main Accuracy: %.4f", final_acc)
    else:
        logging.info("Final Main Accuracy: N/A")

    if final_ba is not None:
        logging.info("Final Backdoor Accuracy: %.4f", final_ba)
    else:
        logging.info("Final Backdoor Accuracy: N/A")

    if len(hist_det) > 0:
        avg_tpr = sum(m["TPR"] for m in hist_det) / len(hist_det)
        avg_fnr = sum(m["FNR"] for m in hist_det) / len(hist_det)
        avg_fpr = sum(m["FPR"] for m in hist_det) / len(hist_det)
        avg_tnr = sum(m["TNR"] for m in hist_det) / len(hist_det)

        logging.info(
            "Average Detection Metrics over all %d rounds: "
            "TPR=%.4f | FNR=%.4f | FPR=%.4f | TNR=%.4f",
            len(hist_det), avg_tpr, avg_fnr, avg_fpr, avg_tnr
        )
    else:
        logging.info("No detection metrics available for final summary.")

    logging.info("=" * 60)
    # ── per-class evaluation after final round ──
    import json
    DATASET_CLASS_NAMES = {
        "cifar10": ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"],
        "cifar100": None,  # 100 classes, use numeric IDs
        "mnist": [str(i) for i in range(10)],
        "fmnist": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
        "svhn": [str(i) for i in range(10)],
    }
    ds_name = cfg.get("dataset_name", "cifar10").lower()
    num_classes = cfg.get("data_partition", {}).get("num_classes", 10)
    class_names = DATASET_CLASS_NAMES.get(ds_name)

    per_class_results = evaluate_per_class(
        server.global_model, test_loader,
        num_classes=num_classes, class_names=class_names,
    )

    # Build experiment summary
    n_total = cfg.get("num_clients", len(clients))
    n_mal = cfg.get("num_malicious", 0)
    mal_ratio = n_mal / max(n_total, 1)
    pdr = cfg.get("client_settings", {}).get("poison_data_ratio", 0.0)
    atk = cfg.get("attack_type", cfg.get("client_settings", {}).get("trigger_type", "badnet"))

    experiment_summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "defense": defense_name,
        "model": cfg.get("model", "unknown"),
        "dataset": ds_name,
        "num_rounds": rounds,
        "num_clients": n_total,
        "num_malicious": n_mal,
        "malicious_clients": cfg.get("malicious_clients", []),
        "malicious_ratio": mal_ratio,
        "attack_type": atk,
        "poison_data_ratio": pdr,
        "trigger_type": cfg.get("client_settings", {}).get("trigger_type", "3x3"),
        "use_neurotoxin": cfg.get("client_settings", {}).get("use_neurotoxin", False),
        "model_replacement": cfg.get("client_settings", {}).get("model_replacement", False),
        "constrain_and_scale": cfg.get("client_settings", {}).get("constrain_and_scale", False),
        "distributed_backdoor": cfg.get("client_settings", {}).get("distributed_backdoor", False),
        "use_semantic_backdoor": cfg.get("client_settings", {}).get("use_semantic_backdoor", False),
        "use_a3fl": cfg.get("client_settings", {}).get("use_a3fl", False),
        "use_pgd_trigger": cfg.get("client_settings", {}).get("use_pgd_trigger", False),
        "use_lp_attack": cfg.get("client_settings", {}).get("use_lp_attack", False),
        "dirichlet_alpha": cfg.get("data_partition", {}).get("dirichlet_alpha", None),
        "fedsurrogate": {
            "replace_mode": cfg.get("defense", {}).get("replace_mode", "full"),
            "enable_rescue": cfg.get("defense", {}).get("enable_rescue", True),
            "rescue_layer_mode": cfg.get("defense", {}).get("rescue_layer_mode", "lca"),
            "enable_replace": cfg.get("defense", {}).get("enable_replace", True),
        } if defense_name.lower() == "fedsurrogate" else None,
        "final_main_accuracy": hist_acc[-1] if hist_acc else None,
        "final_backdoor_accuracy": hist_ba[-1] if hist_ba else None,
        "per_class_accuracy": per_class_results,
        "accuracy_history": hist_acc,
        "backdoor_history": hist_ba,
    }

    # Save to results directory
    results_dir = Path("results") / "evaluations"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build concise attack tag
    cs_ = cfg.get("client_settings", {})
    if cs_.get("use_semantic_backdoor", False):
        atk_tag = "Semantic"
    elif cs_.get("distributed_backdoor", False):
        atk_tag = "DBA"
    elif cs_.get("use_neurotoxin", False):
        atk_tag = "Neuro"
    elif cs_.get("use_lp_attack", False):
        atk_tag = "LPA"
    elif cs_.get("use_a3fl", False):
        atk_tag = "A3FL"
    elif cs_.get("use_pgd_trigger", False):
        atk_tag = "PGD"
    elif cs_.get("model_replacement", False):
        atk_tag = "MR"
    elif cs_.get("constrain_and_scale", False):
        atk_tag = "CaS"
    else:
        atk_tag = "CBA"

    # Build concise defense tag
    def_cfg_ = cfg.get("defense", {})
    if defense_name.lower() == "fedsurrogate":
        rep_on = def_cfg_.get("enable_replace", True)
        if rep_on:
            mode = def_cfg_.get("replace_mode", "full").capitalize()
            rep = f"{mode}"
        else:
            rep = "NoRep"
        def_tag = f"FedS_{rep}"
    else:
        def_tag = defense_name

    pdr_val = cs_.get("poison_data_ratio", 0.0)
    tag = f"{def_tag}_{atk_tag}_pdr{pdr_val}_{ds_name}_{cfg.get('model', 'model')}"
    tag = tag.replace(" ", "_").replace("(", "").replace(")", "")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = results_dir / f"{tag}_{ts}.json"
    with open(result_path, "w") as f:
        json.dump(experiment_summary, f, indent=2)
    logging.info(f"Per-class evaluation saved to {result_path}")

    # Print per-class summary
    print(f"\n{'=' * 60}")
    print(f"  Per-Class Accuracy — {defense_name} | {ds_name} | Round {rounds}")
    print(f"{'=' * 60}")
    for entry in per_class_results["per_class"]:
        bar = "#" * int(entry["accuracy"] / 2)
        print(f"  Class {entry['class_id']:>2} ({entry['class_name']:>12}): "
              f"{entry['accuracy']:6.2f}%  {entry['correct']:>5}/{entry['total']:<5}  {bar}")
    print(f"  {'─' * 50}")
    print(f"  Overall: {per_class_results['overall_accuracy']:.2f}%")
    print(f"{'=' * 60}\n")

    # ── plots ──
    n_total   = cfg.get("num_clients", len(clients))
    n_mal     = cfg.get("num_malicious", 0)
    mal_ratio = n_mal / max(n_total, 1)
    pdr       = cfg.get("client_settings", {}).get("poison_data_ratio", 0.0)
    atk       = cfg.get("attack_type", cfg.get("client_settings", {}).get("trigger_type", "badnet"))

    plot_evaluation_metrics(
        hist_acc, hist_ba,
        defense_name=defense_name,
        dataset_name=cfg.get("dataset_name", "cifar10"),
        model_name=cfg.get("model", "resnet18"),
        attack_type=atk,
        poison_data_ratio=pdr,
        malicious_ratio=mal_ratio,
        num_clients=n_total,
        num_malicious=n_mal,
    )

    # ⭐ UPDATED: Cumulative metrics with method name
    print("\nTRAINING COMPLETE - Computing Cumulative Detection Metrics")
    print("=" * 60)

    if trusted_tracker.get_all_trusted_lists():
        cumulative_summary = detection_summary_over_rounds(
            all_trusted_lists=trusted_tracker.get_all_trusted_lists(),
            malicious_clients=cfg.get("malicious_clients", []),
            total_clients=cfg.get("num_clients", len(clients)),
            method_name=defense_name,  # ⭐ Add this parameter
            verbose=True
        )

# ╭───────────────────────────── MAIN ─────────────────────────────╮

def main() -> None:
    args = parse_args()
    Path("logs").mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / f"experiment_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,  # 防止之前的 logging 配置残留
    )

    start = datetime.now()
    logging.info(f"Log file: {log_path}")
    logging.info(f"🚀 Experiment started: {start:%F %T}")

    cfg = load_config(args.config)
    set_seed(args.seed)
    logging.info("Config loaded:\n" + yaml.dump(cfg))

    # dataset + partition
    train_ds, test_ds = load_dataset(cfg["dataset_name"])
    client_datasets = partition_data(train_ds, cfg)

    # optional distribution plots
    num_classes = cfg["data_partition"].get("num_classes", 10)
    distributions = calculate_class_distribution(client_datasets, num_classes)
    plot_and_save_class_distribution(
        client_datasets, num_classes,
        filename="results/class_distribution.png",
        title="Class Distribution Across Clients",
    )
    plot_individual_distributions(distributions, num_classes)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialise_model(cfg["model"], cfg["dataset_name"]).to(device)

    # attach test data (must be set before build_server for FLTrust root loader)
    cfg["test_data"] = test_ds

    # clients & server
    clients = build_clients(client_datasets, model, cfg)
    server = build_server(model, cfg, len(clients))

    run_federated_learning(clients, server, cfg)

    logging.info(f"🏁 Finished. Total wall-clock: {(datetime.now()-start).total_seconds():.1f}s")


if __name__ == "__main__":
    main()
