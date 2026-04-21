"""
Utility functions for visualizing datasets and client distributions.
"""
import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from typing import List, Union, Dict
from torch.utils.data import Dataset
from torchvision.utils import save_image, make_grid

from utils.backdoor_utils import apply_trigger_batch


def plot_individual_distributions(
        client_distributions: List[Union[List[int], Dict[int, int]]],
        num_classes: int,
        columns: int = 5,
        fig_title: str = "Client Class Distributions",
        partition_method: str = "IID",
        alpha: Union[float, str] = "N/A"
) -> None:
    """
    Plot each client's distribution in a separate subplot and save the figure.

    Args:
        client_distributions: List of client class distributions
        num_classes: Total number of classes
        columns: Number of subplots per row
        fig_title: Title for the entire figure
        partition_method: Data partitioning method
        alpha: Dirichlet alpha parameter
    """
    num_clients = len(client_distributions)
    rows = math.ceil(num_clients / columns)

    full_title = f"{fig_title} ({partition_method}, alpha={alpha})"

    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    fig.suptitle(full_title, fontsize=16)

    if rows * columns == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, distribution in enumerate(client_distributions):
        ax = axes[i]
        if isinstance(distribution, dict):
            counts = [distribution.get(c, 0) for c in range(num_classes)]
        else:
            counts = distribution

        ax.bar(range(num_classes), counts, color='skyblue')
        ax.set_title(f"Client {i} Class Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_xticks(range(num_classes))

    for j in range(num_clients, rows * columns):
        axes[j].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_filename = f"results/Individual_class_distribution_{timestamp}.png"

    plt.savefig(save_filename)
    print(f"Saved class distribution plot as: {save_filename}")
    plt.show()
    plt.close()


def plot_and_save_class_distribution(
    client_datasets: List[Dataset],
    num_classes: int,
    title: str = "Class Distribution Across Clients",
    filename: str = "results/class_distribution.png"
) -> None:
    """
    Plot, save, and print the class distribution for each client as a stacked bar chart.

    Args:
        client_datasets: List of datasets (one per client)
        num_classes: Total number of classes in the dataset
        title: Title of the plot
        filename: Base path to save the plot
    """
    os.makedirs(os.path.dirname(filename) or "results", exist_ok=True)

    num_clients = len(client_datasets)
    client_distributions = []

    for i, dataset in enumerate(client_datasets):
        labels = []
        for item in dataset:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                _, label = item
            else:
                label = item
            labels.append(label)

        class_counts = Counter(labels)
        distribution = [class_counts.get(c, 0) for c in range(num_classes)]
        client_distributions.append(distribution)

    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = np.arange(num_clients)
    bar_width = 0.8
    bottom_stack = np.zeros(num_clients)

    for class_id in range(num_classes):
        counts = [dist[class_id] for dist in client_distributions]
        ax.bar(
            x_positions,
            counts,
            bar_width,
            bottom=bottom_stack,
            label=f"Class {class_id}"
        )
        bottom_stack += np.array(counts)

    ax.set_xlabel("Clients")
    ax.set_ylabel("Number of Samples")
    ax.set_title(title)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{i}" for i in range(num_clients)])
    ax.legend()
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = os.path.join(
        os.path.dirname(filename) or "results",
        f"class_distribution_{timestamp}.png"
    )

    plt.savefig(unique_filename)
    print(f"Saved class distribution plot as: {unique_filename}")
    plt.show()
    plt.close()


def plot_evaluation_metrics(
        accuracies: List[float],
        backdoor_accuracies: List[float],
        *,
        defense_name: str = "FedAvg",
        dataset_name: str = "cifar10",
        model_name: str = "resnet18",
        attack_type: str = "badnet",
        poison_data_ratio: float = 0.0,
        malicious_ratio: float = 0.0,
        num_clients: int = 20,
        num_malicious: int = 0,
        title: str = "",           # legacy — overridden by auto-generated title
) -> None:
    """
    Publication-quality dual-axis plot of Main Task Accuracy (MTA) and
    Attack Success Rate (ASR) over federated rounds.

    Produces a PDF + PNG saved to results/ with a descriptive filename:
      {dataset}_{model}_{defense}_{timestamp}_{attack}_{mal_ratio}malicious_{pdr}poisondata.pdf

    Args:
        accuracies:             MTA per round (%).
        backdoor_accuracies:    ASR per round (%).
        defense_name:           e.g. "FedSurrogate", "FLAME", "FedAvg".
        dataset_name:           e.g. "cifar10", "cifar100", "fmnist".
        model_name:             e.g. "resnet18", "simple_cnn".
        attack_type:            e.g. "badnet", "dba", "adaptive".
        poison_data_ratio:      Fraction of poisoned data (0–1).
        malicious_ratio:        Fraction of malicious clients (0–1).
        num_clients:            Total number of clients.
        num_malicious:          Number of malicious clients.
    """
    import matplotlib
    matplotlib.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif"],
        "font.size":         12,
        "axes.labelsize":    14,
        "axes.titlesize":    13,
        "legend.fontsize":   11,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype":      42,      # TrueType — required by most venues
        "ps.fonttype":       42,
    })

    rounds = list(range(1, len(accuracies) + 1))

    # ── Colour palette ──
    C_MTA = "#2166AC"     # steel blue
    C_ASR = "#B2182B"     # crimson red

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # ── MTA line (left y-axis) ──
    ln1 = ax1.plot(rounds, accuracies,
                   color=C_MTA, linewidth=2.0,
                   marker='o', markersize=3.5, markevery=max(1, len(rounds) // 20),
                   label="Main Task Accuracy (MTA)", zorder=3)
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Main Task Accuracy (%)", color=C_MTA)
    ax1.tick_params(axis='y', colors=C_MTA)
    ax1.set_ylim(-2, 102)
    ax1.set_xlim(1, len(rounds))

    # ── ASR line (right y-axis) ──
    ax2 = ax1.twinx()
    ln2 = ax2.plot(rounds, backdoor_accuracies,
                   color=C_ASR, linewidth=2.0, linestyle='--',
                   marker='s', markersize=3.5, markevery=max(1, len(rounds) // 20),
                   label="Attack Success Rate (ASR)", zorder=3)
    ax2.set_ylabel("Attack Success Rate (%)", color=C_ASR)
    ax2.tick_params(axis='y', colors=C_ASR)
    ax2.set_ylim(-2, 102)

    # ── Grid ──
    ax1.grid(True, which='major', linestyle=':', linewidth=0.5, alpha=0.6)
    ax1.set_axisbelow(True)

    # ── Combined legend ──
    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels,
               loc='center right', framealpha=0.9, edgecolor='0.8',
               fancybox=False)

    # ── Summary stats annotation ──
    final_mta = accuracies[-1] if accuracies else 0
    peak_mta  = max(accuracies) if accuracies else 0
    final_asr = backdoor_accuracies[-1] if backdoor_accuracies else 0
    peak_asr  = max(backdoor_accuracies) if backdoor_accuracies else 0

    stats_text = (f"Final MTA: {final_mta:.1f}%  |  Peak MTA: {peak_mta:.1f}%\n"
                  f"Final ASR: {final_asr:.1f}%  |  Peak ASR: {peak_asr:.1f}%")
    ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='0.8', alpha=0.9))

    # ── Title ──
    # Normalise display names
    ds_display = {
        "cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
        "mnist": "MNIST", "fmnist": "Fashion-MNIST",
        "fashion_mnist": "Fashion-MNIST", "svhn": "SVHN",
    }.get(dataset_name.lower(), dataset_name.upper())

    mdl_display = {
        "resnet18": "ResNet-18", "resnet8": "ResNet-8",
        "simple_cnn": "CNN", "cifar10model": "CIFARNet",
    }.get(model_name.lower(), model_name)

    atk_display = {
        "badnet": "BadNet", "dba": "DBA",
        "adaptive": "Adaptive", "none": "No Attack",
        "backdoor": "BadNet",
    }.get(attack_type.lower(), attack_type)

    def_display = defense_name.replace("_", " ")

    mal_pct = f"{malicious_ratio:.0%}" if malicious_ratio > 0 else f"{num_malicious}/{num_clients}"
    pdr_pct = f"{poison_data_ratio:.0%}" if poison_data_ratio > 0 else "0%"

    plot_title = (f"{def_display}  ·  {ds_display} / {mdl_display}  ·  "
                  f"{atk_display}  ·  PDR {pdr_pct}  ·  "
                  f"Malicious {mal_pct}")
    fig.suptitle(plot_title, fontsize=13, fontweight='bold', y=1.01)

    fig.tight_layout()

    # ── Save ──
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build descriptive filename matching existing convention
    ds_tag  = dataset_name.lower().replace("fashion_mnist", "fashion_mnist").replace("fmnist", "fashion_mnist")
    mdl_tag = model_name.lower().replace("simple_cnn", "cnn")
    def_tag = defense_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    atk_tag = attack_type.lower()
    mal_tag = f"{malicious_ratio:.1f}" if malicious_ratio > 0 else f"{num_malicious / max(num_clients, 1):.1f}"
    pdr_tag = f"{poison_data_ratio}"

    base = f"{ds_tag}_{mdl_tag}_{def_tag}_{timestamp}_{atk_tag}_{mal_tag}malicious_{pdr_tag}poisondata"

    for ext in ("pdf", "png"):
        fpath = os.path.join("results", f"{base}.{ext}")
        fig.savefig(fpath, format=ext)

    # ── Save raw data ──
    txt_path = os.path.join("results", f"accuracy_file_{base}.txt")
    with open(txt_path, 'w') as f:
        f.write(f"# {plot_title}\n")
        f.write(f"# Round  MTA(%)  ASR(%)\n")
        for r, (a, b) in enumerate(zip(accuracies, backdoor_accuracies), 1):
            f.write(f"{r}\t{a:.2f}\t{b:.2f}\n")

    print(f"✅ Saved: results/{base}.pdf  &  .png  &  accuracy_file_*.txt")

    plt.show()
    plt.close(fig)


@torch.no_grad()
def save_backdoored_images(
    dataset,
    trigger_kwargs: dict,
    save_dir: str = "results/vis",
    classes: int = None,
    samples_per_class: int = 5,
    file_prefix: str = "bd",
):
    """
    Save poisoned images for every class.

    Args:
        dataset: Torch Dataset returning (img, label)
        trigger_kwargs: Arguments for apply_trigger_batch
        save_dir: Output folder
        classes: Number of classes (auto-infer if None)
        samples_per_class: Number of examples per class
        file_prefix: Prefix for each PNG file
    """
    os.makedirs(save_dir, exist_ok=True)

    if classes is None:
        labels = [int(lbl) for _, lbl in dataset]
        classes = max(labels) + 1

    global_grid = []
    for cls in range(classes):
        imgs = [img.clone() for img, lbl in dataset if int(lbl) == cls][:samples_per_class]
        if not imgs:
            continue
        batch = torch.stack(imgs)
        poisoned = apply_trigger_batch(batch, **trigger_kwargs)

        grid = make_grid(poisoned, nrow=samples_per_class, pad_value=1.0)
        save_image(grid, os.path.join(save_dir, f"{file_prefix}_{cls}.png"))
        global_grid.append(grid)

    if global_grid:
        cols = min(5, len(global_grid))
        rows = math.ceil(len(global_grid) / cols)
        montage = make_grid(
            global_grid,
            nrow=cols, pad_value=1.0, padding=4
        )
        save_image(montage, os.path.join(save_dir, f"{file_prefix}_montage.png"))
        print(f"Backdoored images written to {save_dir}")
    else:
        print("No class had enough samples to visualize a trigger")


def analyze_model_class_dominance(
    model,
    bias_thresh: float = 5.0,
    weight_thresh: float = 5.0,
    dead_thresh: float = 1e-3
):
    """
    Analyze model's final layer to detect class dominance and dead classes.

    Args:
        model: PyTorch model with a Linear classification layer
        bias_thresh: Threshold for max/median bias ratio
        weight_thresh: Threshold for max/median weight norm ratio
        dead_thresh: Threshold to identify dead classes

    Returns:
        dict: Contains dominance scores, dead classes, and warnings
    """
    final_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            final_layer = module
            break
    if final_layer is None:
        raise ValueError("No Linear layer found in the model")

    weights = final_layer.weight.data
    biases = final_layer.bias.data

    bias_ratio = torch.max(biases) / (torch.median(biases.abs()) + 1e-10)
    weight_norms = torch.norm(weights, dim=1)
    weight_ratio = torch.max(weight_norms) / (torch.median(weight_norms) + 1e-10)

    dead_classes = []
    for i, (b, w) in enumerate(zip(biases, weights)):
        if torch.abs(b) < dead_thresh and torch.norm(w) < dead_thresh:
            dead_classes.append(i)

    is_bias_dominant = bias_ratio > bias_thresh
    is_weight_dominant = weight_ratio > weight_thresh
    is_anomalous = is_bias_dominant or is_weight_dominant

    print(f"Bias dominance ratio (max/median): {bias_ratio:.2f}")
    print(f"Weight dominance ratio (max/median): {weight_ratio:.2f}")
    print(f"Dead classes: {dead_classes if dead_classes else 'None'}")

    if is_anomalous:
        dominant_class = torch.argmax(biases).item()
        print(f"ANOMALOUS: Class {dominant_class} dominates (bias/weight ratio > threshold)")
    else:
        print("Normal: No single class dominates")

    return {
        "bias_ratio": bias_ratio.item(),
        "weight_ratio": weight_ratio.item(),
        "dead_classes": dead_classes,
        "is_anomalous": is_anomalous,
        "dominant_class": torch.argmax(biases).item() if is_anomalous else None,
    }
