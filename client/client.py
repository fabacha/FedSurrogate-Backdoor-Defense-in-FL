# client.py
# Clean client implementations for federated experiments
# Supports benign clients, backdoor attacks (standard and DBA), and more sophisticated
# Date: 2025-03-09

from __future__ import annotations
import copy
import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from utils.backdoor_utils import (
    add_trigger, add_a3fl_trigger, optimise_a3fl_trigger,
    add_pgd_trigger, optimise_pgd_trigger,
)

ModelState = Dict[str, torch.Tensor]


class BaseClient:
    """
    Base client wrapper.
    Clients return full parameter state_dicts via get_model_update().
    """

    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        model: nn.Module,
        client_settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.client_id = int(client_id)
        self.dataset = dataset
        self._client_settings: Dict[str, Any] = client_settings or {}

        self.device = self._client_settings.get("device", torch.device("cpu"))
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        self.model: nn.Module = copy.deepcopy(model).to(self.device)

        self.lr = float(self._client_settings.get("learning_rate", 0.01))
        self.local_epochs = int(self._client_settings.get("local_epochs", 1))
        self.batch_size = int(self._client_settings.get("batch_size", 32))
        momentum = float(self._client_settings.get("momentum", 0.0))
        weight_decay = float(self._client_settings.get("weight_decay", 0.0))

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()

        # Persistent DataLoader — created once, reused every round
        pin = self.device.type == "cuda"
        import os
        _nw = 0 if os.name == 'nt' else 2  # Windows: 0 workers (fork unsupported)
        self._loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=_nw,
            pin_memory=pin,
            persistent_workers=(_nw > 0),
        )

        self.total_weight: Optional[float] = None
        self.adv_weight: Optional[float] = None
        self.num_clients: Optional[int] = None
        self.num_adversaries: Optional[int] = None

        # SPMC aligned gradient: coalition model (teacher) for self-purification
        self.coalition_model: Optional[nn.Module] = None
        self._spmc_lamda: float = float(self._client_settings.get("spmc_lamda", 1.0))
        self._spmc_temperature: float = float(self._client_settings.get("spmc_temperature", 1.0))

    @property
    def client_settings(self) -> Dict[str, Any]:
        return self._client_settings

    def reset_optimizer(self) -> None:
        """Recreate optimizer."""
        momentum = float(self._client_settings.get("momentum", 0.0))
        weight_decay = float(self._client_settings.get("weight_decay", 0.0))
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    def train(self) -> None:
        """Standard local training loop, or SPMC aligned gradient if coalition_model is set."""
        if self.coalition_model is not None:
            from aggregation.spmc import spmc_aligned_train
            spmc_aligned_train(
                model=self.model,
                coalition_model=self.coalition_model,
                dataloader=self._loader,
                optimizer=self.optimizer,
                local_epochs=self.local_epochs,
                device=self.device,
                lamda=self._spmc_lamda,
                temperature=self._spmc_temperature,
            )
            return

        self.model.train()

        for _ in range(self.local_epochs):
            for x, y in self._loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # ← ADD
                self.optimizer.step()


    def get_model_update(self) -> ModelState:
        """
        Return a copy of the full parameter state_dict.
        Tensors are moved to CPU to avoid device mismatches during aggregation.
        """
        return {
            name: param.detach().clone().cpu()
            for name, param in self.model.state_dict().items()
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.client_id})"


class BenignClient(BaseClient):
    """Honest client with no modifications."""
    pass


class MaliciousClient(BaseClient):
    """
    Malicious client supporting:
      - Data backdoor poisoning (standard and DBA)
      - Semantic backdoor attacks
      - Model replacement attack
      - Separate learning rate / local epochs from benign clients
    """

    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        model: nn.Module,
        client_settings: Optional[Dict[str, Any]] = None,
        attack_type: str = "backdoor",
    ) -> None:
        super().__init__(client_id, dataset, model, client_settings)

        self.attack_type = attack_type
        self.poison_ratio = float(self._client_settings.get("poison_data_ratio", 0.3))
        self.backdoor_target = int(self._client_settings.get("backdoor_target", 0))
        self.trigger_type = self._client_settings.get("trigger_type", "3x3")
        self.trigger_position = self._client_settings.get("backdoor_location", "bottom_right")
        self.backdoor_color = tuple(self._client_settings.get("backdoor_color", (1.0, 0.0, 0.0)))
        self.use_dba = bool(self._client_settings.get("distributed_backdoor", False))
        self.pixels_per_client = int(self._client_settings.get("pixels_per_client", 3))
        self.num_dba_clients = int(self._client_settings.get("num_dba_clients", 4))

        self.model_replacement = bool(self._client_settings.get("model_replacement", False))
        self.mr_scale_mode = str(self._client_settings.get("mr_scale_mode", "weight")).lower()
        self.mr_max_scale = float(self._client_settings.get("mr_max_scale", 1e3))

        self.use_semantic = bool(self._client_settings.get("use_semantic_backdoor", False))
        self.semantic_trigger_type = self._client_settings.get("semantic_trigger_type", "green_car")
        self.semantic_source_class = int(self._client_settings.get("semantic_source_class", 0))
        self.semantic_intensity = float(self._client_settings.get("semantic_intensity", 0.3))

        self.initial_global_model: Optional[ModelState] = None

        # ── Neurotoxin (Zhang et al., ICML 2022) ──
        # Masks the poisoned update to only affect bottom-(1-k)% of params
        # by gradient magnitude, making the backdoor more durable.
        self.use_neurotoxin = bool(self._client_settings.get("use_neurotoxin", False))
        self.gradmask_ratio = float(self._client_settings.get("gradmask_ratio", 0.95))

        # ── A3FL (Zhang et al., NeurIPS 2023) ──
        # Adversarially adaptive trigger: patch-based trigger optimised against
        # adversarial model copies that simulate backdoor unlearning.
        self.use_a3fl = bool(self._client_settings.get("use_a3fl", False))
        self.a3fl_trigger_lr = float(self._client_settings.get("a3fl_trigger_lr", 0.02))
        self.a3fl_outer_epochs = int(self._client_settings.get("a3fl_outer_epochs", 10))
        self.a3fl_adv_epochs = int(self._client_settings.get("a3fl_adv_epochs", 5))
        self.a3fl_adv_model_count = int(self._client_settings.get("a3fl_adv_model_count", 2))
        self.a3fl_adv_interval = int(self._client_settings.get("a3fl_adv_interval", 5))
        self.a3fl_noise_loss_lambda = float(self._client_settings.get("a3fl_noise_loss_lambda", 1.0))
        self.a3fl_trigger_size = int(self._client_settings.get("a3fl_trigger_size", 4))
        # Persistent trigger & mask (initialized lazily on first use)
        self._a3fl_trigger: Optional[torch.Tensor] = None
        self._a3fl_mask: Optional[torch.Tensor] = None

        # ── PGD universal perturbation trigger ──
        # Simple ℓ∞-bounded universal additive perturbation via PGD.
        self.use_pgd_trigger = bool(self._client_settings.get("use_pgd_trigger", False))
        self.pgd_eps = float(self._client_settings.get("pgd_eps", 8.0 / 255.0))
        self.pgd_step_size = float(self._client_settings.get("pgd_step_size", 2.0 / 255.0))
        self.pgd_steps = int(self._client_settings.get("pgd_steps", 10))
        self._pgd_trigger: Optional[torch.Tensor] = None

        # ── Constrain-and-Scale (Bagdasaryan et al., AISTATS 2020) ──
        # Projects update onto ℓ2 norm ball then scales to dominate aggregation.
        # Unlike raw Model Replacement, the norm projection lets poisoned updates
        # pass norm-based defenses (Multi-Krum, FLAME, FLTrust).
        self.use_constrain_and_scale = bool(self._client_settings.get("constrain_and_scale", False))
        self.cas_norm_bound = float(self._client_settings.get("cas_norm_bound", 2.0))
        self.cas_scale = float(self._client_settings.get("cas_scale", -1))  # -1 = auto (n/m)
        self.cas_noise_sigma = float(self._client_settings.get("cas_noise_sigma", 0.0))

        # ── LP Attack (Zhuang et al., 2023) ──
        # "Backdoor FL by Poisoning Backdoor-Critical Layers"
        # Trains benign + malicious reference models from global, uses FLS
        # (Forward Layer Substitution) to score layers, then BLS (Backward
        # Layer Substitution) to greedily select the minimal critical set.
        # Final update: benign weights for non-critical, malicious for critical.
        self.use_lp_attack = bool(self._client_settings.get("use_lp_attack", False))
        self.lp_tau = float(self._client_settings.get("lp_tau", 0.8))        # BLS threshold: τ * BSR
        self.lp_benign_epochs = int(self._client_settings.get("lp_benign_epochs", 5))  # epochs for benign ref model
        self._lp_attack_layers: Optional[List[str]] = None  # cache from previous round

        # ── Adaptive Attack ──
        self.use_adaptive_cosine_attack = bool(
            self._client_settings.get("use_adaptive_cosine_attack", False)
        )

        dataset_name = str(self._client_settings.get("dataset_name", "")).lower()
        model_name = str(self._client_settings.get("model", "")).lower()

        self.aca_benign_epochs = int(
            self._client_settings.get("aca_benign_epochs", self.local_epochs)
        )
        self.aca_malicious_epochs = int(
            self._client_settings.get("aca_malicious_epochs", self.local_epochs)
        )
        self.aca_use_malicious_lr = bool(
            self._client_settings.get("aca_use_malicious_lr", True)
        )

        if dataset_name in ["cifar10", "cifar100"] or model_name in ["resnet18"]:
            self.aca_lambda = 5.0
        elif dataset_name in ["fmnist", "mnist"] or model_name in ["simple_cnn"]:
            self.aca_lambda = 1.0
        else:
            self.aca_lambda = float(self._client_settings.get("aca_lambda", 2.0))

        # ── Adaptive Attack + Critical-Layer Replacement ──
        self.use_adaptive_cosine_clr_attack = bool(
            self._client_settings.get("use_adaptive_cosine_clr_attack", False)
        )

        if dataset_name in ["cifar10", "cifar100"] or model_name in ["resnet18"]:
            self.aca_topk = 4
        elif dataset_name in ["fmnist", "mnist"] or model_name in ["simple_cnn"]:
            self.aca_topk = 2
        else:
            self.aca_topk = int(self._client_settings.get("aca_topk", 4))

        # ── Override LR and local epochs for malicious clients ──
        # If mal_learning_rate / mal_local_epochs are set in config, use them.
        # Otherwise fall back to the base (benign) values.
        self.mal_lr = float(self._client_settings.get("mal_learning_rate", self.lr))
        self.mal_local_epochs = int(self._client_settings.get("mal_local_epochs", self.local_epochs))

        # Apply malicious LR: override the base class lr and rebuild optimizer
        if self.mal_lr != self.lr:
            self.lr = self.mal_lr
            self.reset_optimizer()

        # Apply malicious local epochs
        if self.mal_local_epochs != self.local_epochs:
            self.local_epochs = self.mal_local_epochs

        self._print_attack_config()

    def set_initial_global_model(self, global_state_dict: Optional[ModelState]) -> None:
        """Store the initial global model for this round.
        Pass None to explicitly clear the stored global model (e.g. between rounds).
        """
        if global_state_dict is None:
            self.initial_global_model = None
            return
        self.initial_global_model = {
            k: v.detach().clone().cpu() for k, v in global_state_dict.items()
        }

    def _compute_neurotoxin_mask(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Neurotoxin gradient masking (Zhang et al., ICML 2022).

        Compute gradients on clean data, identify the top heavy-hitter
        coordinates, and build a mask that blocks only those.

        gradmask_ratio = fraction of params to KEEP (bottom by gradient magnitude).
          e.g. 0.95 → keep bottom 95%, block top 5% heavy hitters.
        Matches the official implementation: topk(-grads, int(length * ratio)).
        """
        self.model.train()
        self.optimizer.zero_grad()
        for x, y in self._loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            loss = self.criterion(self.model(x), y)
            loss.backward()

        # Flatten all gradient magnitudes
        grad_list = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_list.append(param.grad.detach().abs().flatten())

        if not grad_list:
            return None

        flat_grad = torch.cat(grad_list)
        n_total = flat_grad.numel()

        # Block only the top (1 - gradmask_ratio) heavy hitters
        n_block = max(1, int(n_total * (1.0 - self.gradmask_ratio)))
        threshold = torch.topk(flat_grad, n_block).values[-1]

        # mask=1 for params BELOW threshold (kept), mask=0 for top heavy hitters
        mask = {}
        n_blocked = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                m = (param.grad.detach().abs() < threshold).float()
                mask[name] = m
                n_blocked += (1 - m).sum().item()

        self.optimizer.zero_grad()

        pct_kept = 100.0 * (1.0 - n_blocked / n_total) if n_total > 0 else 0
        print(f"[MaliciousClient {self.client_id}] Neurotoxin mask: "
              f"gradmask_ratio={self.gradmask_ratio:.2f}, "
              f"keeping {pct_kept:.1f}% of params, "
              f"blocking top {100.0 - pct_kept:.1f}% heavy hitters")
        return mask

    def train(self) -> None:
        """Train locally with optional backdoor poisoning."""
        self.model.train()

        # ── A3FL: optimise patch trigger with adversarial models ──
        if self.use_a3fl and self.attack_type == "backdoor":
            # Lazy init: create trigger & mask on first use, persist across rounds
            if self._a3fl_trigger is None:
                sample_x, _ = next(iter(self._loader))
                C, H, W = sample_x.shape[1:]
                self._a3fl_trigger = torch.ones(1, C, H, W) * 0.5
                self._a3fl_mask = torch.zeros(1, C, H, W)
                sz = self.a3fl_trigger_size
                self._a3fl_mask[:, :, 2:2+sz, 2:2+sz] = 1.0

            self._a3fl_trigger = optimise_a3fl_trigger(
                self.model, self._loader,
                target_label=self.backdoor_target,
                trigger=self._a3fl_trigger,
                mask=self._a3fl_mask,
                trigger_lr=self.a3fl_trigger_lr,
                outer_epochs=self.a3fl_outer_epochs,
                adv_epochs=self.a3fl_adv_epochs,
                adv_model_count=self.a3fl_adv_model_count,
                adv_interval=self.a3fl_adv_interval,
                noise_loss_lambda=self.a3fl_noise_loss_lambda,
                device=self.device,
            )
            print(f"[MaliciousClient {self.client_id}] A3FL trigger optimised "
                  f"(lr={self.a3fl_trigger_lr}, outer={self.a3fl_outer_epochs}, "
                  f"adv_models={self.a3fl_adv_model_count})")

        # ── PGD: optimise universal perturbation trigger ──
        if self.use_pgd_trigger and self.attack_type == "backdoor":
            self._pgd_trigger = optimise_pgd_trigger(
                self.model, self._loader,
                target_label=self.backdoor_target,
                eps=self.pgd_eps,
                step_size=self.pgd_step_size,
                steps=self.pgd_steps,
                device=self.device,
            )
            print(f"[MaliciousClient {self.client_id}] PGD trigger optimised "
                  f"(eps={self.pgd_eps:.4f}, steps={self.pgd_steps})")

        # ── Neurotoxin: compute gradient mask BEFORE poisoned training ──
        neurotoxin_mask = None
        if self.use_neurotoxin and self.initial_global_model is not None:
            neurotoxin_mask = self._compute_neurotoxin_mask()

        for _ in range(self.local_epochs):
            for x, y in self._loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if self.attack_type == "backdoor":
                    self._poison_batch(x, y)

                self.optimizer.zero_grad()
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                loss.backward()

                # ── Neurotoxin: zero out gradients for top-k% params ──
                if neurotoxin_mask is not None:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and name in neurotoxin_mask:
                            param.grad.mul_(neurotoxin_mask[name])
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # ← ADD
                self.optimizer.step()
                

    def _poison_batch(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """In-place poison a subset of the batch."""
        batch_size = x.size(0)
        k = int(self.poison_ratio * batch_size)
        if k <= 0:
            return

        if self.use_semantic:
            source_class_mask = (y == self.semantic_source_class)
            source_indices = torch.where(source_class_mask)[0]

            if len(source_indices) == 0:
                return

            k = min(k, len(source_indices))
            idxs = source_indices[torch.randperm(len(source_indices))[:k]].tolist()
        else:
            idxs = random.sample(range(batch_size), k)

        if self.use_a3fl and self._a3fl_trigger is not None:
            # A3FL: apply patch trigger via mask blending
            x[idxs] = add_a3fl_trigger(x[idxs], self._a3fl_trigger, self._a3fl_mask)
        elif self.use_pgd_trigger and self._pgd_trigger is not None:
            # PGD: apply universal additive perturbation
            x[idxs] = add_pgd_trigger(x[idxs], self._pgd_trigger)
        elif self.use_semantic:
            from utils.backdoor_utils import add_semantic_trigger
            x[idxs] = add_semantic_trigger(
                x[idxs],
                trigger_type=self.semantic_trigger_type,
                source_class=self.semantic_source_class,
                intensity=self.semantic_intensity,
            )
        else:
            x[idxs] = add_trigger(
                x[idxs],
                trigger_type=self.trigger_type,
                position=self.trigger_position,
                color=self.backdoor_color,
                is_grayscale=(x.size(1) == 1),
                use_dba=self.use_dba,
                client_id=self.client_id,
                num_mal=self.num_dba_clients,
                pixels_per_client=self.pixels_per_client,
            )

        y[idxs] = torch.tensor(self.backdoor_target, device=y.device).expand(len(idxs))

    def _print_attack_config(self) -> None:
        """Print the attack configuration for this client."""
        print(f"\n{'=' * 60}")
        print(f"⚠️  MALICIOUS CLIENT {self.client_id} - Attack Configuration")
        print(f"{'=' * 60}")
        print(f"  Backdoor Type: {self.attack_type}")
        print(f"  Poison Ratio: {self.poison_ratio:.1%}")
        print(f"  Backdoor Target: {self.backdoor_target}")

        # Show training config differences from benign
        benign_lr = float(self._client_settings.get("learning_rate", 0.01))
        benign_ep = int(self._client_settings.get("local_epochs", 1))
        if self.lr != benign_lr or self.local_epochs != benign_ep:
            print(f"\n  Training Config (differs from benign):")
            print(f"    Benign:    LR={benign_lr}, Epochs={benign_ep}")
            print(f"    Malicious: LR={self.lr}, Epochs={self.local_epochs}")
        else:
            print(f"  Training Config: LR={self.lr}, Epochs={self.local_epochs} (same as benign)")

        attack_modes = ["Backdoor"]
        if self.use_a3fl:
            attack_modes.append(f"A3FL(lr={self.a3fl_trigger_lr}, outer={self.a3fl_outer_epochs})")
        if self.use_pgd_trigger:
            attack_modes.append(f"PGD(eps={self.pgd_eps:.4f}, steps={self.pgd_steps})")
        if self.use_neurotoxin:
            attack_modes.append(f"Neurotoxin(k={self.gradmask_ratio:.2f})")
        if self.model_replacement:
            attack_modes.append(f"MR-{self.mr_scale_mode}")
        if self.use_constrain_and_scale:
            attack_modes.append(f"CaS(norm={self.cas_norm_bound:.1f})")
        if self.use_lp_attack:
            attack_modes.append(f"LP(τ={self.lp_tau}, benign_ep={self.lp_benign_epochs})")
        if self.use_adaptive_cosine_attack:
            attack_modes.append(
            f"ACA(λ={self.aca_lambda}, ben_ep={self.aca_benign_epochs}, mal_ep={self.aca_malicious_epochs})"
        )
        if self.use_adaptive_cosine_clr_attack:
            attack_modes.append(
            f"Adaptive(λ={self.aca_lambda}, k={self.aca_topk}, ben_ep={self.aca_benign_epochs}, mal_ep={self.aca_malicious_epochs})"
        )

        print(f"  Attack Pipeline: {' -> '.join(attack_modes)}")

        if self.model_replacement:
            print(f"\n  MR Configuration:")
            print(f"    Scale Mode: {self.mr_scale_mode}")
            print(f"    Max Scale: {self.mr_max_scale}")

        if self.use_constrain_and_scale:
            print(f"\n  Constrain-and-Scale Configuration:")
            print(f"    Norm Bound: {self.cas_norm_bound}")
            print(f"    Scale: {'auto' if self.cas_scale <= 0 else self.cas_scale}")
            print(f"    Noise Sigma: {self.cas_noise_sigma}")

        if self.use_semantic:
            print(f"\n  Semantic Backdoor:")
            print(f"    Trigger: {self.semantic_trigger_type}")
            print(f"    Source Class: {self.semantic_source_class} -> Target: {self.backdoor_target}")

        if self.use_a3fl:
            print(f"\n  A3FL Configuration:")
            print(f"    Trigger LR: {self.a3fl_trigger_lr}")
            print(f"    Outer Epochs: {self.a3fl_outer_epochs}")
            print(f"    Adv Model Count: {self.a3fl_adv_model_count}")
            print(f"    Adv Epochs: {self.a3fl_adv_epochs}")
            print(f"    Trigger Size: {self.a3fl_trigger_size}x{self.a3fl_trigger_size}")

        if self.use_pgd_trigger:
            print(f"\n  PGD Trigger Configuration:")
            print(f"    Epsilon: {self.pgd_eps:.4f}")
            print(f"    Step Size: {self.pgd_step_size:.4f}")
            print(f"    Steps: {self.pgd_steps}")
        
        if self.use_adaptive_cosine_clr_attack:
            print(f"\n  Adaptive Critical Layer Replacement Attack Configuration:")
            print(f"    Lambda: {self.aca_lambda}")
            print(f"    Top-k Layers: {self.aca_topk}")
            print(f"    Benign Epochs: {self.aca_benign_epochs}")
            print(f"    Malicious Epochs: {self.aca_malicious_epochs}")
        
        if self.use_adaptive_cosine_attack:
            print(f"\n  Adaptive Attack Configuration:")
            print(f"    Lambda: {self.aca_lambda}")
            print(f"    Benign Epochs: {self.aca_benign_epochs}")
            print(f"    Malicious Epochs: {self.aca_malicious_epochs}")

        print(f"{'=' * 60}\n")

    def get_model_update(self) -> ModelState:
        """
        Return full model parameters with optional transformations.

        Priority (mutually exclusive, first match wins):
          1. Model Replacement scaling
          2. Constrain-and-Scale (norm projection + scaling)
          3. LP Attack (layer poisoning) — can combine with Neurotoxin
          4. Neurotoxin gradient mask (applied during training, no post-hoc)
          5. Raw update (no post-processing)
        """
        if self.model_replacement:
            if self.initial_global_model is None:
                print(f"[MaliciousClient {self.client_id}] WARNING: Cannot run MR: initial_global_model not set")
                return super().get_model_update()
            if self.use_neurotoxin:
                print(f"[MaliciousClient {self.client_id}] WARNING: Neurotoxin disabled: model_replacement takes priority")
            return self._get_scaled_update()

        if self.use_constrain_and_scale:
            if self.initial_global_model is None:
                print(f"[MaliciousClient {self.client_id}] WARNING: Cannot run CaS: initial_global_model not set")
                return super().get_model_update()
            return self._get_constrain_and_scale_update()
        
        if self.use_adaptive_cosine_clr_attack:
            if self.initial_global_model is None:
                print(f"[MaliciousClient {self.client_id}] WARNING: Cannot run ACA-CLR: initial_global_model not set")
                return super().get_model_update()
            return self._get_adaptive_cosine_clr_update()
        
        if self.use_adaptive_cosine_attack:
            if self.initial_global_model is None:
                print(f"[MaliciousClient {self.client_id}] WARNING: Cannot run ACA: initial_global_model not set")
                return super().get_model_update()
            return self._get_adaptive_cosine_align_update()

        if self.use_lp_attack:
            if self.initial_global_model is None:
                print(f"[MaliciousClient {self.client_id}] WARNING: Cannot run LP: initial_global_model not set")
                return super().get_model_update()
            return self._get_lp_attack_update()

        # Neurotoxin is applied during training (gradient masking), so no
        # post-hoc processing needed here.
        return super().get_model_update()

    def _get_scaled_update(self) -> ModelState:
        """Apply MR scaling to current model."""
        if self.mr_scale_mode == "weight":
            if self.total_weight is None or self.adv_weight is None:
                raise ValueError("MR(weight) requires total_weight and adv_weight.")
            if float(self.adv_weight) <= 0.0:
                raise ValueError("adv_weight must be > 0.")
            scale = float(self.total_weight) / float(self.adv_weight)
        elif self.mr_scale_mode == "count":
            if self.num_clients is None or self.num_adversaries is None:
                raise ValueError("MR(count) requires num_clients and num_adversaries.")
            if int(self.num_adversaries) <= 0:
                raise ValueError("num_adversaries must be > 0.")
            scale = float(self.num_clients) / float(self.num_adversaries)
        else:
            raise ValueError(f"Unknown mr_scale_mode='{self.mr_scale_mode}'")

        if scale > float(self.mr_max_scale):
            scale = float(self.mr_max_scale)

        current = {
            k: v.detach().clone().cpu()
            for k, v in self.model.state_dict().items()
        }
        if self.initial_global_model is None:
            raise ValueError("initial_global_model not set")

        scaled_state: ModelState = {}
        for k, cur in current.items():
            init = self.initial_global_model.get(k)
            if init is None:
                raise KeyError(f"Missing key '{k}' in initial_global_model")
            delta = cur - init
            scaled_state[k] = (init + delta * scale).detach().clone()

        delta_norm = self._compute_update_norm(current, self.initial_global_model)
        scaled_delta_norm = delta_norm * scale

        print(
            f"[MaliciousClient {self.client_id}] MR: "
            f"scale={scale:.3f}, ||Δ||={delta_norm:.4f} → ||Δ_scaled||={scaled_delta_norm:.4f}"
        )

        return {k: v.clone() for k, v in scaled_state.items()}

    def _compute_update_norm(self, current: ModelState, original: ModelState) -> float:
        """Compute L2 norm of update."""
        total = 0.0
        for k, v_curr in current.items():
            if k not in original:
                continue
            v_orig = original[k]
            delta = v_curr.detach() - v_orig.detach()
            total += torch.sum(delta ** 2).item()
        return total ** 0.5

    # ── Constrain-and-Scale (Bagdasaryan et al., AISTATS 2020) ──────────
    def _get_constrain_and_scale_update(self) -> ModelState:
        """
        Constrain-and-Scale attack:
          1. Compute delta = W_local - W_global
          2. Project delta onto ℓ₂ norm ball of radius *cas_norm_bound*
          3. Scale delta by *cas_scale* (auto = n_clients / n_adversaries)
          4. Optionally add Gaussian noise for extra stealth

        Unlike raw Model Replacement, the norm projection lets the poisoned
        update pass norm-based defenses (Multi-Krum, FLAME, FLTrust) before
        the scaling amplifies it during aggregation.
        """
        current = {
            k: v.detach().clone().cpu()
            for k, v in self.model.state_dict().items()
        }
        if self.initial_global_model is None:
            raise ValueError("initial_global_model not set")

        # Step 1: compute delta
        delta: ModelState = {}
        for k in current:
            delta[k] = current[k].float() - self.initial_global_model[k].float()

        # Step 2: compute ℓ₂ norm and project
        raw_norm = sum(torch.sum(d ** 2).item() for d in delta.values()) ** 0.5
        if raw_norm > self.cas_norm_bound and raw_norm > 0:
            proj_scale = self.cas_norm_bound / raw_norm
            delta = {k: d * proj_scale for k, d in delta.items()}
            projected_norm = self.cas_norm_bound
        else:
            projected_norm = raw_norm

        # Step 3: scale
        if self.cas_scale <= 0:
            # Auto: n_clients / n_adversaries
            nc = self.num_clients or 20
            na = self.num_adversaries or 1
            gamma = float(nc) / float(na)
        else:
            gamma = self.cas_scale

        delta = {k: d * gamma for k, d in delta.items()}

        # Step 4: optional noise injection
        if self.cas_noise_sigma > 0:
            delta = {
                k: d + torch.randn_like(d) * self.cas_noise_sigma
                for k, d in delta.items()
            }

        # Reconstruct
        result: ModelState = {}
        for k in current:
            result[k] = (self.initial_global_model[k].float() + delta[k]).to(current[k].dtype)

        print(
            f"[MaliciousClient {self.client_id}] CaS: "
            f"||delta||={raw_norm:.4f} -> proj={projected_norm:.4f} -> "
            f"scaled={projected_norm * gamma:.4f} (gamma={gamma:.1f})"
        )

        return result
    
    # ── Adaptive Attack ──────────
    def _train_clean_reference_model(self) -> ModelState:

        benign_model = copy.deepcopy(self.model)
        benign_model.load_state_dict(
            {k: v.to(self.device) for k, v in self.initial_global_model.items()},
            strict=False,
        )
        benign_model.train()

        momentum = float(self._client_settings.get("momentum", 0.0))
        weight_decay = float(self._client_settings.get("weight_decay", 0.0))
        benign_optimizer = optim.SGD(
            benign_model.parameters(),
            lr=self.lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.aca_benign_epochs):
            for x, y in self._loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                benign_optimizer.zero_grad()
                logits = benign_model(x)
                loss = criterion(logits, y)
                loss.backward()
                benign_optimizer.step()

        return {
            k: v.detach().clone().cpu()
            for k, v in benign_model.state_dict().items()
        }

    def _adaptive_cosine_align_loss(
        self,
        bad_model: nn.Module,
        benign_state: ModelState,
        global_state: ModelState,
        eps: float = 1e-12,
    ) -> torch.Tensor:

        reg_loss = None
        valid_layer_cnt = 0

        named_params = dict(bad_model.named_parameters())

        for name, param in named_params.items():
            if name not in benign_state or name not in global_state:
                continue

            bad_delta = param - global_state[name].to(self.device, dtype=param.dtype)
            benign_delta = (
                benign_state[name].to(self.device, dtype=param.dtype)
                - global_state[name].to(self.device, dtype=param.dtype)
            )

            bad_flat = bad_delta.view(-1)
            benign_flat = benign_delta.view(-1)

            bad_norm = torch.norm(bad_flat, p=2)
            benign_norm = torch.norm(benign_flat, p=2)

            if bad_norm.item() < eps or benign_norm.item() < eps:
                continue

            cos_sim = torch.dot(bad_flat, benign_flat) / (bad_norm * benign_norm + eps)
            term = (cos_sim - 1.0) ** 2

            reg_loss = term if reg_loss is None else reg_loss + term
            valid_layer_cnt += 1

        if valid_layer_cnt == 0:
            return torch.tensor(0.0, device=self.device)

        return reg_loss / valid_layer_cnt
    
    def _compute_update_cosine_similarity(
        self,
        state_a: ModelState,
        state_b: ModelState,
        global_state: ModelState,
        eps: float = 1e-12,
    ) -> float:

        vec_a = []
        vec_b = []

        named_params = dict(self.model.named_parameters())

        for name in named_params.keys():
            if name not in state_a or name not in state_b or name not in global_state:
                continue

            delta_a = (state_a[name] - global_state[name]).float().view(-1)
            delta_b = (state_b[name] - global_state[name]).float().view(-1)

            vec_a.append(delta_a)
            vec_b.append(delta_b)

        if len(vec_a) == 0:
            return 0.0

        vec_a = torch.cat(vec_a, dim=0)
        vec_b = torch.cat(vec_b, dim=0)

        norm_a = torch.norm(vec_a, p=2)
        norm_b = torch.norm(vec_b, p=2)

        if norm_a.item() < eps or norm_b.item() < eps:
            return 0.0

        cos_sim = torch.dot(vec_a, vec_b) / (norm_a * norm_b + eps)
        return float(cos_sim.item())

    def _train_adaptive_malicious_reference_model(self, benign_state: ModelState) -> ModelState:

        bad_model = copy.deepcopy(self.model)
        bad_model.load_state_dict(
            {k: v.to(self.device) for k, v in self.initial_global_model.items()},
            strict=False,
        )
        bad_model.train()

        momentum = float(self._client_settings.get("momentum", 0.0))
        weight_decay = float(self._client_settings.get("weight_decay", 0.0))
        bad_lr = self.mal_lr if self.aca_use_malicious_lr else self.lr

        bad_optimizer = optim.SGD(
            bad_model.parameters(),
            lr=bad_lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        cls_meter = 0.0
        reg_meter = 0.0
        step_cnt = 0

        for _ in range(self.aca_malicious_epochs):
            for x, y in self._loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                # poison a copy, keep style aligned with existing _poison_batch()
                x_poison = x.clone()
                y_poison = y.clone()
                self._poison_batch(x_poison, y_poison)

                bad_optimizer.zero_grad()
                logits = bad_model(x_poison)

                cls_loss = criterion(logits, y_poison)
                reg_loss = self._adaptive_cosine_align_loss(
                    bad_model=bad_model,
                    benign_state=benign_state,
                    global_state=self.initial_global_model,
                )
                loss = cls_loss + self.aca_lambda * reg_loss

                loss.backward()
                bad_optimizer.step()

                cls_meter += float(cls_loss.item())
                reg_meter += float(reg_loss.item())
                step_cnt += 1

        if step_cnt > 0:
            print(
                f"[MaliciousClient {self.client_id}] ACA: "
                f"avg_cls={cls_meter / step_cnt:.4f}, "
                f"avg_reg={reg_meter / step_cnt:.4f}, "
                f"lambda={self.aca_lambda}"
            )

        return {
            k: v.detach().clone().cpu()
            for k, v in bad_model.state_dict().items()
        }

    def _get_adaptive_cosine_align_update(self) -> ModelState:
        """
        Adaptive attack for cosine-similarity-based defenses:
          1. Train a clean reference model from global weights on clean data
          2. Retrain a malicious model from the same global weights on poisoned data
             with layer-wise cosine-alignment regularization
          3. Return the malicious reference model as the uploaded update/state
        """
        benign_state = self._train_clean_reference_model()
        malicious_state = self._train_adaptive_malicious_reference_model(benign_state)

        # Log update norms for debugging
        cos_sim = self._compute_update_cosine_similarity(
            state_a=malicious_state,
            state_b=benign_state,
            global_state=self.initial_global_model,
        )
        print(
            f"[MaliciousClient {self.client_id}] ACA: "
            f"cos(Δ_mal, Δ_benign)={cos_sim:.6f}"
        )

        return malicious_state
    
    def _get_conv_linear_weight_keys(self) -> List[str]:
        """
        Return only weight parameter keys belonging to Conv2d / Linear modules.
        Example:
            conv1.weight, layer1.0.conv2.weight, fc.weight, ...
        """
        allowed_keys = []

        for module_name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                full_name = f"{module_name}.weight" if module_name else "weight"
                allowed_keys.append(full_name)

        return allowed_keys
    
    def _get_layerwise_update_cosine_scores(
        self,
        malicious_state: ModelState,
        benign_state: ModelState,
        global_state: ModelState,
        eps: float = 1e-12,
    ) -> List[tuple]:
        scores = []
        param_keys = self._get_conv_linear_weight_keys()

        for name in param_keys:
            if name not in malicious_state or name not in benign_state or name not in global_state:
                continue

            mal_delta = (malicious_state[name] - global_state[name]).float().view(-1)
            benign_delta = (benign_state[name] - global_state[name]).float().view(-1)

            mal_norm = torch.norm(mal_delta, p=2)
            benign_norm = torch.norm(benign_delta, p=2)

            if mal_norm.item() < eps or benign_norm.item() < eps:
                continue

            cos_sim = torch.dot(mal_delta, benign_delta) / (mal_norm * benign_norm + eps)
            scores.append((name, float(cos_sim.item())))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def _get_adaptive_cosine_clr_update(self) -> ModelState:
        """
        ACA-CLR attack:
          1. Train a clean reference model from global weights on clean data
          2. Retrain a malicious model from the same global weights on poisoned data
             with cosine-alignment regularization
          3. Compute layer-wise cosine similarity between malicious/benign updates
          4. Select top-k most aligned layers
          5. Return a composite model:
                benign_state for non-selected layers
                malicious_state for selected top-k layers
        """
        benign_state = self._train_clean_reference_model()
        malicious_state = self._train_adaptive_malicious_reference_model(benign_state)

        layer_scores = self._get_layerwise_update_cosine_scores(
            malicious_state=malicious_state,
            benign_state=benign_state,
            global_state=self.initial_global_model,
        )

        if len(layer_scores) == 0:
            print(f"[MaliciousClient {self.client_id}] ACA-CLR: no valid layers, fallback to malicious_state")
            return malicious_state

        topk = min(self.aca_topk, len(layer_scores))
        selected_layers = [name for name, _ in layer_scores[:topk]]

        result = {}
        for k in benign_state:
            if k in selected_layers:
                result[k] = malicious_state[k].clone()
            else:
                result[k] = benign_state[k].clone()

        topk_scores = [(name, round(score, 6)) for name, score in layer_scores[:topk]]
        print(
            f"[MaliciousClient {self.client_id}] ACA-CLR: "
            f"selected top-{topk} layers by cosine similarity"
        )
        print(f"  Top-k layers: {topk_scores}")

        return result

    # ── LP Attack (Zhuang et al., 2023) ─────────────────────────────

    def _lp_train_benign_model(self) -> ModelState:
        """Train a benign reference model from global weights on local clean data."""
        import copy as _copy
        benign_model = _copy.deepcopy(self.model)
        benign_model.load_state_dict(
            {k: v.to(self.device) for k, v in self.initial_global_model.items()}
        )
        benign_model.train()
        momentum = float(self._client_settings.get("momentum", 0.0))
        weight_decay = float(self._client_settings.get("weight_decay", 0.0))
        benign_opt = optim.SGD(
            benign_model.parameters(), lr=self.lr,
            momentum=momentum, weight_decay=weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        for _ in range(self.lp_benign_epochs):
            for x, y in self._loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                benign_opt.zero_grad()
                criterion(benign_model(x), y).backward()
                benign_opt.step()
        return {k: v.detach().clone().cpu() for k, v in benign_model.state_dict().items()}

    @torch.no_grad()
    def _lp_measure_bsr(self, state: ModelState) -> float:
        """Measure Backdoor Success Rate of a state dict on local data with triggers."""
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in state.items()}, strict=False
        )
        self.model.eval()

        from utils.backdoor_utils import apply_trigger_batch
        correct, total = 0, 0
        for x, y in self._loader:
            x = x.to(self.device, non_blocking=True)
            x_triggered = apply_trigger_batch(
                x, trigger_type=self.trigger_type,
                position=self.trigger_position, color=self.backdoor_color,
                use_dba=False, client_id=self.client_id,
            )
            preds = self.model(x_triggered).argmax(dim=1)
            correct += (preds == self.backdoor_target).sum().item()
            total += x.size(0)

        self.model.load_state_dict(original_state)
        self.model.train()
        return correct / max(total, 1)

    def _lp_fls(self, malicious_state: ModelState, benign_state: ModelState,
                bsr_full: float) -> List[tuple]:
        """
        Forward Layer Substitution (FLS).
        For each named_parameter key, replace it in the malicious model with
        the benign version and measure BSR drop.
        Returns list of (key, score) sorted by score ascending (most negative = most critical).
        """
        param_keys = [k for k, _ in self.model.named_parameters()]
        scores = []
        for key in param_keys:
            hybrid = dict(malicious_state)
            if key in benign_state:
                hybrid[key] = benign_state[key]
            bsr_replaced = self._lp_measure_bsr(hybrid)
            # score = BSR_replaced - BSR_full (negative means BSR dropped → layer is critical)
            scores.append((key, bsr_replaced - bsr_full))
        # Sort ascending: most negative first (most critical)
        scores.sort(key=lambda x: x[1])
        return scores

    def _lp_bls(self, scores: List[tuple], benign_state: ModelState,
                malicious_state: ModelState, bsr_full: float) -> List[str]:
        """
        Backward Layer Substitution (BLS).
        Greedily add the most critical layers (from FLS ranking) to a benign
        base until the composite model reaches τ * BSR_full.
        Returns the list of critical layer keys.
        """
        target_bsr = self.lp_tau * bsr_full
        attack_layers = []

        for n in range(1, len(scores) + 1):
            # Take top-n most critical layers
            candidate_keys = [k for k, _ in scores[:n]]
            # Build composite: benign base + malicious critical layers
            composite = dict(benign_state)
            for k in candidate_keys:
                composite[k] = malicious_state[k]
            composite_bsr = self._lp_measure_bsr(composite)
            if composite_bsr >= target_bsr:
                attack_layers = candidate_keys
                break
        else:
            # Could not reach target; use all layers ranked as critical (negative score)
            attack_layers = [k for k, s in scores if s < 0]

        return attack_layers

    def _get_lp_attack_update(self) -> ModelState:
        """
        LP Attack (Zhuang et al., 2023) — full pipeline:
          1. Train a benign reference model from global on clean local data
          2. The malicious model is already trained (self.model) with backdoor
          3. FLS: score each layer by BSR drop when replaced with benign version
          4. BLS: greedily select minimal critical set reaching τ * BSR
          5. Final update: benign weights for non-critical, malicious for critical
          6. Cache attack_layers; reuse next round if they still achieve τ * BSR
        """
        malicious_state = {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}
        global_state = self.initial_global_model

        # Measure BSR of full malicious model
        bsr_full = self._lp_measure_bsr(malicious_state)

        # Train benign reference model
        benign_state = self._lp_train_benign_model()

        # Skip optimisation: if cached attack_layers still work, reuse them
        if self._lp_attack_layers:
            composite = dict(benign_state)
            for k in self._lp_attack_layers:
                if k in malicious_state:
                    composite[k] = malicious_state[k]
            cached_bsr = self._lp_measure_bsr(composite)
            if cached_bsr >= self.lp_tau * bsr_full and bsr_full > 0:
                attack_layers = self._lp_attack_layers
                print(
                    f"[MaliciousClient {self.client_id}] LP Attack: "
                    f"BSR={bsr_full:.3f}, reusing {len(attack_layers)} cached layers "
                    f"(composite BSR={cached_bsr:.3f} >= τ*BSR={self.lp_tau * bsr_full:.3f})"
                )
                result: ModelState = {}
                for k in malicious_state:
                    result[k] = malicious_state[k].clone() if k in attack_layers else benign_state.get(k, global_state[k]).clone()
                return result

        # FLS: score all layers
        scores = self._lp_fls(malicious_state, benign_state, bsr_full)

        # BLS: greedily select critical layers
        if bsr_full > 0:
            attack_layers = self._lp_bls(scores, benign_state, malicious_state, bsr_full)
        else:
            attack_layers = []

        # Cache for next round
        self._lp_attack_layers = attack_layers if attack_layers else None

        # Build final update
        if not attack_layers:
            print(
                f"[MaliciousClient {self.client_id}] LP Attack: "
                f"BSR={bsr_full:.3f}, no critical layers — sending full poisoned update"
            )
            return malicious_state

        result = {}
        for k in malicious_state:
            if k in attack_layers:
                result[k] = malicious_state[k].clone()
            else:
                result[k] = benign_state.get(k, global_state[k]).clone()

        param_keys = [k for k, _ in self.model.named_parameters()]
        n_total = len(param_keys)
        print(
            f"[MaliciousClient {self.client_id}] LP Attack: "
            f"BSR={bsr_full:.3f}, poisoning {len(attack_layers)}/{n_total} layers"
        )
        print(f"  Critical layers: {attack_layers}")

        return result

