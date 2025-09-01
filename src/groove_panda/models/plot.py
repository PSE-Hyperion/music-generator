import logging
import os
import shutil

from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import History  # type: ignore

from groove_panda import directories
from groove_panda.config import Config

config = Config()
logger = logging.getLogger(__name__)


def plot_training(history: History, model_name: str) -> None:
    if config.plot_training:
        plot_path = os.path.join(directories.MODELS_DIR, model_name, f"training_{model_name}")

        # If the dir exists, it's an old version. Delete it so the new version can be saved.
        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        os.makedirs(plot_path, exist_ok=False)
        _plot_training_history(history, model_name, plot_path)
        _plot_training_metrics_separate(history, model_name, plot_path)
        _plot_training_validation(history, model_name, plot_path)


def _plot_training_history(history, model_name: str, dir_path: str) -> None:
    """
    Plot training history showing loss and accuracy for all 6 feature outputs.
    """

    feature_names = [feature.name for feature in config.features]

    # Create subplots: 4 rows x 3 columns (loss and accuracy for each feature)
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(f"Training History - {model_name}", fontsize=16, fontweight="bold")

    # Plot loss for each feature (first 2 rows)
    for i, feature in enumerate(feature_names):
        row = i // 3
        col = i % 3

        loss_key = f"output_{feature}_loss"
        val_loss_key = f"val_output_{feature}_loss"

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

        acc_key = f"output_{feature}_accuracy"
        val_acc_key = f"val_output_{feature}_accuracy"

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
    if config.save_plot_training:
        file_path = os.path.join(dir_path, f"training_history_{model_name}.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        logger.info("Training history plot saved to: %s", file_path)

    # plt.show()


def _plot_training_metrics_separate(history: History, model_name: str, dir_path: str) -> None:
    """
    Alternative version: Plot loss and accuracy in separate figures for better readability.
    """

    feature_names = [feature.name for feature in config.features]

    # Seperate loss plot
    plt.figure(figsize=(15, 10))

    for i, feature in enumerate(feature_names):
        plt.subplot(2, 3, i + 1)

        loss_key = f"output_{feature}_loss"
        val_loss_key = f"val_output_{feature}_loss"

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
    if config.save_plot_training:
        file_path = os.path.join(dir_path, f"loss_{model_name}.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        logger.info("Loss plot saved to: %s", file_path)

    # plt.show()

    # Seperate accuracy plot
    has_accuracy = any(f"output_{feature}_accuracy" in history.history for feature in feature_names)

    if has_accuracy:
        plt.figure(figsize=(15, 10))

        for i, feature in enumerate(feature_names):
            plt.subplot(2, 3, i + 1)

            acc_key = f"output_{feature}_accuracy"
            val_acc_key = f"val_output_{feature}_accuracy"

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
    if config.save_plot_training:
        file_path = os.path.join(dir_path, f"accuracy_{model_name}.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        logger.info("Accuracy plot saved to: %s", file_path)

    # plt.show()


def _plot_training_validation(history: History, model_name: str, dir_path: str) -> None:
    """
    Plot only validation loss and accuracy in separate figures for better readability.
    """

    feature_names = [feature.name for feature in config.features]

    # Validation loss plot
    plt.figure(figsize=(15, 10))

    for i, feature in enumerate(feature_names):
        plt.subplot(2, 3, i + 1)

        val_loss_key = f"val_output_{feature}_loss"

        if val_loss_key in history.history:
            plt.plot(history.history[val_loss_key], label="Validation Loss", color="red")

            plt.title(f"{feature.title()} Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)

    plt.suptitle(f"Validation Loss History - {model_name}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Optional save
    if config.save_plot_training:
        file_path = os.path.join(dir_path, f"val_loss_{model_name}.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        logger.info("Validation loss plot saved to: %s", file_path)

    # plt.show()

    # Validation accuracy plot
    has_val_accuracy = any(f"val_output_{feature}_accuracy" in history.history for feature in feature_names)

    if has_val_accuracy:
        plt.figure(figsize=(15, 10))

        for i, feature in enumerate(feature_names):
            plt.subplot(2, 3, i + 1)

            val_acc_key = f"val_output_{feature}_accuracy"

            if val_acc_key in history.history:
                plt.plot(history.history[val_acc_key], label="Validation Accuracy", color="orange")

                plt.title(f"{feature.title()} Validation Accuracy")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.grid(True, alpha=0.3)

        plt.suptitle(f"Validation Accuracy History - {model_name}", fontsize=16, fontweight="bold")
        plt.tight_layout()

        # Optional save
        if config.save_plot_training:
            file_path = os.path.join(dir_path, f"val_accuracy_{model_name}.png")
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            logger.info("Validation accuracy plot saved to: %s", file_path)
