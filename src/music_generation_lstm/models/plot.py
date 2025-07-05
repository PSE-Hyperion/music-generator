import os

from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import History  # type: ignore

from ..config import PLOT_DIR, PLOT_TRAINING, SAVE_PLOT_TRAINING

# The docker container is missing a backend to display plot results (plt.show wont work)
# This means we can add this, to allow the plots to be displayed upon training completion, if enabled,
# or we keep this as is (should suffice)


def plot_training(history: History, model_name: str):
    if PLOT_TRAINING:
        dir_path = os.path.join(PLOT_DIR, f"training_{model_name}")
        os.makedirs(dir_path, exist_ok=False)
        _plot_training_history(history, model_name, dir_path)
        _plot_training_metrics_separate(history, model_name, dir_path)


def _plot_training_history(history, model_name: str, dir_path: str):
    """
    Plot training history showing loss and accuracy for all 6 feature outputs.
    """

    feature_names = ["bar", "position", "pitch", "duration", "velocity", "tempo"]

    # Create subplots: 4 rows x 3 columns (loss and accuracy for each feature)
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(f"Training History - {model_name}", fontsize=16, fontweight="bold")

    # Plot loss for each feature (first 2 rows)
    for i, feature in enumerate(feature_names):
        row = i // 3
        col = i % 3

        loss_key = f"{feature}_output_loss"
        val_loss_key = f"val_{feature}_output_loss"

        if loss_key in history.history:
            axes[row, col].plot(history.history[loss_key], label="Training Loss", color="blue")
            if val_loss_key in history.history:
                axes[row, col].plot(history.history[val_loss_key], label="Validation Loss", color="red")

            axes[row, col].set_title(f"{feature.title()} Loss")
            axes[row, col].set_xlabel("Epoch")
            axes[row, col].set_ylabel("Loss")
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)

    # Plot accuracy for each feature (last 2 rows)
    for i, feature in enumerate(feature_names):
        row = (i // 3) + 2  # Offset by 2 rows for accuracy plots
        col = i % 3

        acc_key = f"{feature}_output_accuracy"
        val_acc_key = f"val_{feature}_output_accuracy"

        if acc_key in history.history:
            axes[row, col].plot(history.history[acc_key], label="Training Accuracy", color="green")
            if val_acc_key in history.history:
                axes[row, col].plot(history.history[val_acc_key], label="Validation Accuracy", color="orange")

            axes[row, col].set_title(f"{feature.title()} Accuracy")
            axes[row, col].set_xlabel("Epoch")
            axes[row, col].set_ylabel("Accuracy")
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        else:
            # Hide subplot if no accuracy data
            axes[row, col].set_visible(False)

    plt.tight_layout()

    # Optional save
    if SAVE_PLOT_TRAINING:
        file_path = os.path.join(dir_path, f"training_history_{model_name}.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Training history plot saved to: {file_path}")

    # plt.show()


def _plot_training_metrics_separate(history: History, model_name: str, dir_path: str):
    """
    Alternative version: Plot loss and accuracy in separate figures for better readability.
    """

    feature_names = ["bar", "position", "pitch", "duration", "velocity", "tempo"]

    # Seperate loss plot
    plt.figure(figsize=(15, 10))

    for i, feature in enumerate(feature_names):
        plt.subplot(2, 3, i + 1)

        loss_key = f"{feature}_output_loss"
        val_loss_key = f"val_{feature}_output_loss"

        if loss_key in history.history:
            plt.plot(history.history[loss_key], label="Training Loss", color="blue")
            if val_loss_key in history.history:
                plt.plot(history.history[val_loss_key], label="Validation Loss", color="red")

            plt.title(f"{feature.title()} Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)

    plt.suptitle(f"Training Loss History - {model_name}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Optional save
    if SAVE_PLOT_TRAINING:
        file_path = os.path.join(dir_path, f"loss_{model_name}.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Loss plot saved to: {file_path}")

    # plt.show()

    # Seperate accuracy plot
    has_accuracy = any(f"{feature}_output_accuracy" in history.history for feature in feature_names)

    if has_accuracy:
        plt.figure(figsize=(15, 10))

        for i, feature in enumerate(feature_names):
            plt.subplot(2, 3, i + 1)

            acc_key = f"{feature}_output_accuracy"
            val_acc_key = f"val_{feature}_output_accuracy"

            if acc_key in history.history:
                plt.plot(history.history[acc_key], label="Training Accuracy", color="green")
                if val_acc_key in history.history:
                    plt.plot(history.history[val_acc_key], label="Validation Accuracy", color="orange")

                plt.title(f"{feature.title()} Accuracy")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.grid(True, alpha=0.3)

        plt.suptitle(f"Training Accuracy History - {model_name}", fontsize=16, fontweight="bold")
        plt.tight_layout()

    # Optional save
    if SAVE_PLOT_TRAINING:
        file_path = os.path.join(dir_path, f"accuracy_{model_name}.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Accuracy plot saved to: {file_path}")

    # plt.show()
