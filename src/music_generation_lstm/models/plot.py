import matplotlib.pyplot as plt
import os

from keras.api.callbacks import History
from config import PLOT_DIR, PLOT_TRAINING

# The docker container is missing a backend to display plot results (plt.show wont work)

def plot_training(history : History, model_name : str):
    dir_path = os.path.join(PLOT_DIR, f"training_{model_name}")
    os.makedirs(dir_path, exist_ok=False)
    _plot_training_history(history, model_name, dir_path)
    _plot_training_metrics_separate(history, model_name, dir_path)

def _plot_training_history(history : History, model_name : str, dir_path : str):
    """
    Plot training history showing loss and accuracy for all 6 feature outputs.

    Args:
        history: Keras History object from model.fit()
        model_name: Name of the model for the plot title
        save_path: Optional path to save the plot image
    """

    feature_names = ["bar", "position", "pitch", "duration", "velocity", "tempo"]

    # Create subplots: 2 rows (loss, accuracy) x 3 columns (3 features per row)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Training History - {model_name}', fontsize=16, fontweight='bold')

    # Plot loss for each feature
    for i, feature in enumerate(feature_names):
        row = i // 3
        col = i % 3

        loss_key = f'{feature}_loss'
        val_loss_key = f'val_{feature}_loss'

        if loss_key in history.history:
            axes[row, col].plot(history.history[loss_key], label=f'Training Loss', color='blue')
            if val_loss_key in history.history:
                axes[row, col].plot(history.history[val_loss_key], label=f'Validation Loss', color='red')

            axes[row, col].set_title(f'{feature.title()} Loss')
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('Loss')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()

    # Optional save
    if PLOT_TRAINING:
        file_path = os.path.join(dir_path, f"training_history_{model_name}.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {file_path}")

    #plt.show()

def _plot_training_metrics_separate(history : History, model_name: str, dir_path : str):
    """
    Alternative version: Plot loss and accuracy in separate figures for better readability.

    Args:
        history: Keras History object from model.fit()
        model_name: Name of the model for the plot title
        save_path: Optional path to save the plot image (will create two files)
    """

    feature_names = ["bar", "position", "pitch", "duration", "velocity", "tempo"]

    # Plot 1: All losses
    plt.figure(figsize=(15, 10))

    for i, feature in enumerate(feature_names):
        plt.subplot(2, 3, i + 1)

        loss_key = f'{feature}_loss'
        val_loss_key = f'val_{feature}_loss'

        if loss_key in history.history:
            plt.plot(history.history[loss_key], label='Training Loss', color='blue')
            if val_loss_key in history.history:
                plt.plot(history.history[val_loss_key], label='Validation Loss', color='red')

            plt.title(f'{feature.title()} Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)

    plt.suptitle(f'Training Loss History - {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Optional save
    if PLOT_TRAINING:
        file_path = os.path.join(dir_path, f"loss_{model_name}.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved to: {file_path}")

    #plt.show()

    # Plot 2: All accuracies (if available)
    has_accuracy = any(f'{feature}_accuracy' in history.history for feature in feature_names)

    if has_accuracy:
        plt.figure(figsize=(15, 10))

        for i, feature in enumerate(feature_names):
            plt.subplot(2, 3, i + 1)

            acc_key = f'{feature}_accuracy'
            val_acc_key = f'val_{feature}_accuracy'

            if acc_key in history.history:
                plt.plot(history.history[acc_key], label='Training Accuracy', color='green')
                if val_acc_key in history.history:
                    plt.plot(history.history[val_acc_key], label='Validation Accuracy', color='orange')

                plt.title(f'{feature.title()} Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True, alpha=0.3)

        plt.suptitle(f'Training Accuracy History - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()


    # Optional save
    if PLOT_TRAINING:
        file_path = os.path.join(dir_path, f"accuracy_{model_name}.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy plot saved to: {file_path}")

    #plt.show()
