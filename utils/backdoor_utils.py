"""
Back‑door helper utilities
-------------------------

This module gives you identical behaviour on every platform:

* `add_trigger`            – static or DBA trigger, px‑exact splitting
* `add_semantic_trigger`   – edge‑case / semantic backdoor (colour transforms)
* `add_a3fl_trigger`       – A3FL patch trigger with mask (Zhang et al., NeurIPS 2023)
* `add_pgd_trigger`        – simple PGD universal perturbation trigger
* `build_trigger_loader`   – construct a DataLoader for BA evaluation
* `apply_trigger_batch`    – convenience wrapper for batch tensors
"""

# utils/backdoor_utils.py
from __future__ import annotations

import random
from typing import Tuple, Sequence, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ──────────────────────────────────────────────────────────────────────────────
# 1. add_trigger  ── paint a pattern onto one *or* many images
# ──────────────────────────────────────────────────────────────────────────────
def add_trigger(
    images: torch.Tensor,                         # [B,C,H,W]  or  [C,H,W]
    *,
    trigger_type: str = "3x3",                 # '1pixel' | '4pixel' | '3x3'
    position: str = "bottom_right",                # 'top_left' … 'bottom_right'
    color: Tuple[float, ...] = (1.0, 0.0, 0.0),   # RGB or (val,) for grey
    is_grayscale: bool = False,
    use_dba: bool = False,
    client_id: int | None = None,
    num_mal: int | None = None,
    pixels_per_client: int | None = 3             # if None → legacy sharding
) -> torch.Tensor:
    """
    Paint a (static or DBA) trigger onto *images* **in‑place**.

    When *use_dba* is `True` every malicious client receives exactly
    `pixels_per_client` coordinates selected *cyclically* from the pattern.

    Parameters
    ----------
    images               : `Tensor`  shape [C,H,W] **or** [B,C,H,W]
    trigger_type         : str       pattern key ('1pixel' / '4pixel' / '3x3')
    position             : str       anchor corner
    color                : tuple     RGB or single value
    is_grayscale         : bool      treat image as single‑channel even if C==3
    use_dba              : bool      enable distributed backdoor attack
    client_id, num_mal   : int       DBA bookkeeping
    pixels_per_client    : int|None  #pixels per attacker (None → legacy mode)
    """
    # ── pattern coordinates (0‑origin, <=3) ─────────────────────────────
    patterns: dict[str, List[Tuple[int, int]]] = {
        "1pixel": [(1, 1)],
        "4pixel": [(1, 1), (2, 2), (3, 3), (1, 3)],
        "3x3":    [(i, j) for i in range(3) for j in range(3)],
    }
    if trigger_type not in patterns:
        raise ValueError(f"Unknown trigger_type '{trigger_type}'")
    full_coords = patterns[trigger_type]

    # ── DBA sharding  ───────────────────────────────────────────────────
    if use_dba:
        if client_id is None or num_mal is None:
            raise ValueError("use_dba=True requires client_id and num_mal")

        if pixels_per_client is None:                    # legacy modulo shard
            coords = [c for i, c in enumerate(full_coords)
                      if i % num_mal == client_id % num_mal]
        else:                                            # exactly N pix / cl
            coords = [full_coords[(client_id * pixels_per_client + i)
                                   % len(full_coords)]
                      for i in range(pixels_per_client)]
    else:
        coords = full_coords

    if not coords:                                       # sanity fallback
        coords = full_coords

    # ── anchor offset (auto‑fits any image size) ────────────────────────
    anchors = {
        "top_left":     lambda H, W, mx, my: (0, 0),
        "top_right":    lambda H, W, mx, my: (0, W - my - 1),
        "bottom_left":  lambda H, W, mx, my: (H - mx - 1, 0),
        "bottom_right": lambda H, W, mx, my: (H - mx - 1, W - my - 1),
    }
    if position not in anchors:
        raise ValueError(f"Unknown position '{position}'")

    # ensure batch dimension
    batch = images.unsqueeze(0) if images.dim() == 3 else images
    B, C, H, W = batch.shape
    max_x = max(c[0] for c in full_coords)
    max_y = max(c[1] for c in full_coords)
    dx, dy = anchors[position](H, W, max_x, max_y)

    # ── paint pixels ────────────────────────────────────────────────────
    for img in batch:                                    # iterate over B
        for x, y in coords:
            px, py = dx + x, dy + y                      # absolute coords
            if is_grayscale or img.size(0) == 1:
                img[0, px, py] = color[0]
            else:
                img[0, px, py], img[1, px, py], img[2, px, py] = color

    return batch if images.dim() == 4 else batch.squeeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# 2. apply_trigger_batch  ── vectorised convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────
def apply_trigger_batch(imgs: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Apply `add_trigger` to every image in *imgs* (batched).

    All keyword arguments are forwarded unmodified.
    Legacy keys like `trigger_value` are silently ignored.
    """
    kwargs.pop("trigger_value", None)     # strip deprecated arg
    for i in range(imgs.size(0)):
        add_trigger(imgs[i], **kwargs)
    return imgs


# ──────────────────────────────────────────────────────────────────────────────
# 3. build_trigger_loader  ── create a DataLoader for BA evaluation
# ──────────────────────────────────────────────────────────────────────────────
def build_trigger_loader(
    base_dataset,
    trigger_source_classes: Sequence[int],
    trigger_target_label:   int,
    batch_size:             int,
    *,
    trigger_type:      str = "3x3",
    trigger_position:  str = "bottom_right",
    color: Tuple[float, ...] = (1.0, 0.0, 0.0),
    is_grayscale:      bool = False,
    use_dba:           bool = False,
    num_mal:           int = 4,
    pixels_per_client: int = 3,
) -> DataLoader:
    """
    Return a DataLoader in which *every* sample:

    1. originates from `trigger_source_classes`,
    2. is relabelled to `trigger_target_label`,
    3. carries the trigger (DBA shards if required).

    Useful for measuring back‑door accuracy (attack success rate).
    """
    xs, ys = [], []
    for img, lbl in base_dataset:
        if lbl in trigger_source_classes:
            img = img.clone()

            if use_dba:
                for cid in range(num_mal):
                    add_trigger(
                        img,
                        trigger_type=trigger_type,
                        position=trigger_position,
                        color=color,
                        is_grayscale=is_grayscale,
                        use_dba=True,
                        client_id=cid,
                        num_mal=num_mal,
                        pixels_per_client=pixels_per_client,
                    )
            else:
                add_trigger(
                    img,
                    trigger_type=trigger_type,
                    position=trigger_position,
                    color=color,
                    is_grayscale=is_grayscale,
                    use_dba=False,
                )

            xs.append(img)
            ys.append(torch.tensor(trigger_target_label))

    ds = TensorDataset(torch.stack(xs), torch.stack(ys))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ──────────────────────────────────────────────────────────────────────────────
# 4. add_semantic_trigger ── edge‑case / semantic backdoor (Wang et al., 2020)
# ──────────────────────────────────────────────────────────────────────────────
def add_semantic_trigger(
    images: torch.Tensor,
    *,
    trigger_type: str = "green_car",
    source_class: int = 0,
    intensity: float = 0.3,
) -> torch.Tensor:
    """
    Apply a *semantic* trigger – a subtle, natural‑looking colour / texture
    transform instead of a pixel‑patch pattern.

    Reference: "Attack of the Tails" (Wang et al., ICLR 2020) and
    "How to Backdoor Federated Learning" (Bagdasaryan et al., AISTATS 2020).

    Supported trigger_types
    -----------------------
    green_car   – additive green‑channel tint  (mimics green‑coloured objects)
    stripe      – faint horizontal stripe overlay
    brightness  – uniform brightness shift
    colour_shift – hue rotation via channel permutation + blending

    Parameters
    ----------
    images         : [C,H,W] or [B,C,H,W]  – modified **in‑place**
    trigger_type   : str   – semantic trigger variant
    source_class   : int   – (informational; selection done by caller)
    intensity      : float – blend factor in [0, 1]
    """
    batch = images.unsqueeze(0) if images.dim() == 3 else images
    B, C, H, W = batch.shape

    if trigger_type == "green_car":
        # Boost green channel, slightly suppress red & blue
        batch[:, 1, :, :] = batch[:, 1, :, :] * (1.0 - intensity) + intensity
        if C >= 3:
            batch[:, 0, :, :] *= (1.0 - 0.3 * intensity)
            batch[:, 2, :, :] *= (1.0 - 0.3 * intensity)

    elif trigger_type == "stripe":
        # Faint horizontal stripes every 4 rows
        stripe_mask = torch.zeros(1, 1, H, W, device=batch.device)
        stripe_mask[:, :, ::4, :] = 1.0
        batch.mul_(1.0 - intensity * stripe_mask).add_(intensity * stripe_mask)

    elif trigger_type == "brightness":
        # Uniform brightness increase
        batch.mul_(1.0 - intensity).add_(intensity)

    elif trigger_type == "colour_shift":
        # Rotate channels (R→G→B→R) and blend with original
        if C >= 3:
            shifted = batch[:, [1, 2, 0], :, :].clone()
            batch.mul_(1.0 - intensity).add_(shifted * intensity)
        else:
            # Grayscale fallback: invert and blend
            batch.mul_(1.0 - intensity).add_((1.0 - batch) * intensity)

    else:
        raise ValueError(f"Unknown semantic trigger_type '{trigger_type}'")

    batch.clamp_(0.0, 1.0)
    return batch if images.dim() == 4 else batch.squeeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# 5. A3FL trigger ── patch‑based adversarially adaptive trigger (NeurIPS 2023)
#    Zhang et al., "A3FL: Adversarially Adaptive Backdoor Attacks to FL"
# ──────────────────────────────────────────────────────────────────────────────
def add_a3fl_trigger(
    images: torch.Tensor,
    trigger: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Apply A3FL patch trigger via mask blending: x' = trigger*mask + (1-mask)*x.

    Parameters
    ----------
    images  : [C,H,W] or [B,C,H,W]
    trigger : [1,C,H,W] – learned trigger pattern
    mask    : [1,C,H,W] – binary mask (1 = trigger region, 0 = clean)
    """
    batch = images.unsqueeze(0) if images.dim() == 3 else images
    t = trigger.to(batch.device)
    m = mask.to(batch.device)
    result = t * m + (1 - m) * batch
    result.clamp_(0.0, 1.0)
    if images.dim() == 3:
        return result.squeeze(0)
    # In-place update for batch
    images.copy_(result)
    return images


def _build_a3fl_adversarial_model(
    model: nn.Module,
    data_loader: DataLoader,
    trigger: torch.Tensor,
    mask: torch.Tensor,
    *,
    adv_epochs: int = 5,
    adv_lr: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> tuple:
    """
    Train an adversarial copy of the model that "unlearns" the backdoor.

    The adv model is trained on triggered inputs with their TRUE labels,
    simulating the global model's natural tendency to overwrite the backdoor.

    Returns (adv_model, cosine_similarity_weight).
    """
    import copy
    adv_model = copy.deepcopy(model).to(device)
    adv_model.train()
    ce_loss = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(adv_model.parameters(), lr=adv_lr, momentum=0.9, weight_decay=5e-4)

    t = trigger.to(device)
    m = mask.to(device)

    for _ in range(adv_epochs):
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Apply trigger but keep TRUE labels → "unlearn" the backdoor
            inputs = t * m + (1 - m) * inputs
            loss = ce_loss(adv_model(inputs), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Compute cosine similarity weight between adv and original model conv grads
    cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
    sim_sum, sim_count = 0.0, 0
    adv_params = dict(adv_model.named_parameters())
    orig_params = dict(model.named_parameters())
    for name in adv_params:
        if "conv" in name and adv_params[name].grad is not None and orig_params[name].grad is not None:
            sim_count += 1
            sim_sum += cos_sim(
                adv_params[name].grad.reshape(-1),
                orig_params[name].grad.reshape(-1),
            ).item()

    adv_w = sim_sum / max(sim_count, 1)
    return adv_model, adv_w


def optimise_a3fl_trigger(
    model: nn.Module,
    data_loader: DataLoader,
    target_label: int,
    trigger: torch.Tensor,
    mask: torch.Tensor,
    *,
    trigger_lr: float = 0.02,
    outer_epochs: int = 10,
    adv_epochs: int = 5,
    adv_model_count: int = 2,
    adv_interval: int = 5,
    noise_loss_lambda: float = 1.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    A3FL adversarially adaptive trigger optimisation (Zhang et al., NeurIPS 2023).

    Core idea: optimise the trigger to fool not only the current model but also
    adversarial copies that have been trained to "unlearn" the trigger.

    Parameters
    ----------
    model            : current local model
    data_loader      : training data loader
    target_label     : backdoor target class
    trigger          : [1,C,H,W] current trigger (persistent across rounds)
    mask             : [1,C,H,W] binary trigger mask
    trigger_lr       : PGD step size (α)
    outer_epochs     : number of outer optimisation loops (K)
    adv_epochs       : epochs to train each adversarial model
    adv_model_count  : number of adversarial model copies per refresh
    adv_interval     : refresh adversarial models every N outer epochs
    noise_loss_lambda: weight for adversarial model loss component
    device           : torch device

    Returns
    -------
    trigger : [1,C,H,W] – updated trigger pattern (clamped to [-2, 2])
    """
    device = torch.device(device) if isinstance(device, str) else device
    model.eval()

    t = trigger.clone().to(device)
    m = mask.to(device)
    ce_loss = nn.CrossEntropyLoss()

    adv_models = []
    adv_ws = []

    for outer in range(outer_epochs):
        # Periodically refresh adversarial models
        if outer % adv_interval == 0 and outer != 0:
            # Clean up old models
            for am in adv_models:
                del am
            adv_models, adv_ws = [], []
            for _ in range(adv_model_count):
                am, aw = _build_a3fl_adversarial_model(
                    model, data_loader, t, m,
                    adv_epochs=adv_epochs, device=device,
                )
                adv_models.append(am)
                adv_ws.append(aw)

        for x_batch, _ in data_loader:
            t.requires_grad_(True)
            x_batch = x_batch.to(device)
            bs = x_batch.size(0)
            target = torch.full((bs,), target_label, dtype=torch.long, device=device)

            # Apply trigger via mask
            x_triggered = t * m + (1 - m) * x_batch

            # Loss on current model
            loss = ce_loss(model(x_triggered), target)

            # Add adversarial model losses (core A3FL contribution)
            if adv_models:
                for am, aw in zip(adv_models, adv_ws):
                    am_loss = ce_loss(am(x_triggered), target)
                    loss = loss + noise_loss_lambda * aw * am_loss / len(adv_models)

            loss.backward()

            # PGD step
            with torch.no_grad():
                t = t - trigger_lr * t.grad.sign()
                t = t.detach()
                t = torch.clamp(t, min=-2, max=2)

    # Clean up
    for am in adv_models:
        del am

    model.train()
    return t.detach().cpu()


# ──────────────────────────────────────────────────────────────────────────────
# 6. PGD universal perturbation trigger (simple baseline)
#    ℓ∞-bounded universal additive perturbation optimised via PGD.
# ──────────────────────────────────────────────────────────────────────────────
def add_pgd_trigger(
    images: torch.Tensor,
    trigger_delta: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a PGD universal additive perturbation to images.

    Parameters
    ----------
    images        : [C,H,W] or [B,C,H,W]
    trigger_delta : [1,C,H,W] – additive perturbation
    """
    batch = images.unsqueeze(0) if images.dim() == 3 else images
    delta = trigger_delta.to(batch.device)
    if delta.dim() == 3:
        delta = delta.unsqueeze(0)
    batch.add_(delta).clamp_(0.0, 1.0)
    return batch if images.dim() == 4 else batch.squeeze(0)


def optimise_pgd_trigger(
    model: nn.Module,
    data_loader: DataLoader,
    target_label: int,
    *,
    eps: float = 8.0 / 255.0,
    step_size: float = 2.0 / 255.0,
    steps: int = 10,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Simple PGD universal perturbation trigger.

    Solves:  min_δ  E_{(x,y)∈D}[ CE( f(x + δ), y_target ) ]
             s.t.   ||δ||∞ ≤ eps

    Returns
    -------
    trigger_delta : Tensor [1, C, H, W] – ℓ∞-bounded universal perturbation
    """
    device = torch.device(device) if isinstance(device, str) else device
    model.eval()

    sample_x, _ = next(iter(data_loader))
    C, H, W = sample_x.shape[1:]
    delta = torch.zeros(1, C, H, W, device=device, requires_grad=True)
    criterion = nn.CrossEntropyLoss()

    for step in range(steps):
        for x_batch, _ in data_loader:
            x_batch = x_batch.to(device)
            bs = x_batch.size(0)
            target = torch.full((bs,), target_label, dtype=torch.long, device=device)

            x_adv = (x_batch + delta).clamp(0.0, 1.0)
            loss = criterion(model(x_adv), target)
            loss.backward()

        with torch.no_grad():
            delta.data -= step_size * delta.grad.sign()
            delta.data.clamp_(-eps, eps)
            delta.grad.zero_()

    model.train()
    return delta.detach().clone()



