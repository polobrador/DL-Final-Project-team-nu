import os

import matplotlib.pyplot as plt


def plot_and_save_learning_curves(train_losses, val_losses, val_aucs, fold_num, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, train_losses, label="Train Loss", marker="o", color="blue")
    axes[0].plot(epochs, val_losses, label="Validation Loss", marker="o", color="orange")
    axes[0].set_title(f"Fold {fold_num} - Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.7)
    axes[1].plot(epochs, val_aucs, label="Validation AUC", marker="o", color="green")
    axes[1].set_title(f"Fold {fold_num} - AUC over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro ROC-AUC")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"learning_curves_fold{fold_num}.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Learning curves saved to {save_path}")
