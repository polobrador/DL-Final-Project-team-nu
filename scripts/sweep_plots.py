import os

import matplotlib.pyplot as plt


def plot_gamma_sweep(sweep_results, sweep_history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    gammas = list(sweep_results.keys())
    peak_aucs = list(sweep_results.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gammas, peak_aucs, marker="o", linestyle="-", color="purple", linewidth=2, markersize=8)
    ax.set_title("Focal Loss Gamma vs Peak Validation AUC")
    ax.set_xlabel("Focal Loss Gamma")
    ax.set_ylabel("Peak Macro ROC-AUC")
    ax.grid(True, linestyle="--", alpha=0.7)
    for i, txt in enumerate(peak_aucs):
        ax.annotate(f"{txt:.4f}", (gammas[i], peak_aucs[i]), textcoords="offset points", xytext=(0, 10), ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gamma_sweep_summary.png"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    for gamma, history in sweep_history.items():
        epochs = range(1, len(history["val_aucs"]) + 1)
        ax.plot(epochs, history["val_aucs"], marker="o", label=f"Gamma {gamma}")
    ax.set_title("Validation AUC Trajectory per Gamma")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Macro ROC-AUC")
    ax.legend(title="Gamma Values")
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gamma_sweep_trajectories.png"), dpi=300)
    plt.close()
    print(f"Gamma sweep plots saved to {save_dir}")


def plot_weight_decay_sweep(sweep_results, sweep_history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    wds = list(sweep_results.keys())
    peak_aucs = list(sweep_results.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wds, peak_aucs, marker="o", linestyle="-", color="teal", linewidth=2, markersize=8)
    ax.set_xscale("log")
    ax.set_title("Weight Decay vs Peak Validation AUC")
    ax.set_xlabel("Weight Decay (log scale)")
    ax.set_ylabel("Peak Macro ROC-AUC")
    ax.grid(True, linestyle="--", alpha=0.7)
    for i, txt in enumerate(peak_aucs):
        ax.annotate(f"{txt:.4f}", (wds[i], peak_aucs[i]), textcoords="offset points", xytext=(0, 10), ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "weight_decay_sweep_summary.png"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    for wd, history in sweep_history.items():
        epochs = range(1, len(history["val_aucs"]) + 1)
        ax.plot(epochs, history["val_aucs"], marker="o", label=f"wd={wd}")
    ax.set_title("Validation AUC Trajectory per Weight Decay")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Macro ROC-AUC")
    ax.legend(title="Weight Decay")
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "weight_decay_trajectories.png"), dpi=300)
    plt.close()
    print(f"Weight decay sweep plots saved to {save_dir}")


def plot_lr_sweep(sweep_results, sweep_history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    lrs = list(sweep_results.keys())
    peak_aucs = list(sweep_results.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lrs, peak_aucs, marker="o", linestyle="-", color="firebrick", linewidth=2, markersize=8)
    ax.set_xscale("log")
    ax.set_title("Learning Rate vs Peak Validation AUC")
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel("Peak Macro ROC-AUC")
    ax.grid(True, linestyle="--", alpha=0.7)
    for i, txt in enumerate(peak_aucs):
        ax.annotate(f"{txt:.4f}", (lrs[i], peak_aucs[i]), textcoords="offset points", xytext=(0, 10), ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lr_sweep_summary.png"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    for lr, history in sweep_history.items():
        epochs = range(1, len(history["val_aucs"]) + 1)
        ax.plot(epochs, history["val_aucs"], marker="o", label=f"LR={lr}")
    ax.set_title("Validation AUC Trajectory per Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Macro ROC-AUC")
    ax.legend(title="Learning Rates")
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lr_trajectories.png"), dpi=300)
    plt.close()
    print(f"LR sweep plots saved to {save_dir}")


def plot_context_sweep(sweep_results, sweep_history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    contexts = [str(k) for k in sweep_results.keys()]
    peak_aucs = list(sweep_results.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(contexts, peak_aucs, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_title("Temporal Context Window vs Peak Validation AUC")
    ax.set_xlabel("Context size (2.5s windows)")
    ax.set_ylabel("Peak Macro ROC-AUC")
    min_auc = min(peak_aucs) - 0.01
    ax.set_ylim([max(0.0, min_auc), max(peak_aucs) + 0.005])
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.001, f"{yval:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "context_sweep_summary.png"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    for ctx, history in sweep_history.items():
        epochs = range(1, len(history["val_aucs"]) + 1)
        ax.plot(epochs, history["val_aucs"], marker="o", label=f"{ctx} windows ({ctx * 1536} dims)")
    ax.set_title("Validation AUC Trajectory per Context Size")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Macro ROC-AUC")
    ax.legend(title="Context")
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "context_trajectories.png"), dpi=300)
    plt.close()
    print(f"Context sweep plots saved to {save_dir}")
