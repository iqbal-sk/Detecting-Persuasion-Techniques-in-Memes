import os
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from transformers.models.auto.image_processing_auto import model_type

from config.logger import get_logger
from config import config

logger = get_logger(__name__)

def create_plots(metrics):
    """
    Generates and saves combined plots for training metrics in a single image arranged in a row.

    Args:
        metrics (dict): A dictionary containing lists of metrics.
                        Expected keys: 'Epoch', 'Training Loss', 'Validation Loss',
                                       'Precision', 'Recall', 'F1-Score'
        plot_dir (str): Directory where the plot will be saved.
    """
    try:
        # Ensure the plot directory exists
        model_plot_dir = config.results.plot_dir

        os.makedirs(model_plot_dir, exist_ok=True)
        logger.info(f"Plot directory '{model_plot_dir}' is ready.")

        # Extract metrics
        epochs = metrics.get('Epoch', [])
        training_loss = metrics.get('Training Loss', [])
        validation_loss = metrics.get('Validation Loss', [])
        precision = metrics.get('Precision', [])
        recall = metrics.get('Recall', [])
        f1_score = metrics.get('F1-Score', [])

        if not epochs:
            logger.warning("No epochs data available for plotting.")
            return

        # Set Seaborn style
        sns.set(style="whitegrid")

        # Generate a timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Create a figure with subplots in a row
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 6))  # Adjust figsize for width

        # Plot Training and Validation Loss on the first subplot
        axes[0].plot(epochs, training_loss, label='Training Loss', color='blue')
        axes[0].plot(epochs, validation_loss, label='Validation Loss', color='orange')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss Over Epochs')
        axes[0].legend()
        axes[0].grid(True)

        # Plot Precision, Recall, and F1-Score on the second subplot
        axes[1].plot(epochs, precision, label='Precision', color='green')
        axes[1].plot(epochs, recall, label='Recall', color='red')
        axes[1].plot(epochs, f1_score, label='F1-Score', color='purple')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Precision, Recall, and F1-Score Over Epochs')
        axes[1].legend()
        axes[1].grid(True)

        # Plot F1-Score Distribution on the third subplot
        sns.histplot(f1_score, bins=20, kde=True, color='magenta', ax=axes[2])
        axes[2].set_xlabel('F1-Score')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('F1-Score Distribution Over Epochs')
        axes[2].grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the combined figure
        combined_plot_path = os.path.join(model_plot_dir, f'training_metrics_{timestamp}.png')
        plt.savefig(combined_plot_path)
        plt.close()
        logger.info(f"Combined plot saved to '{combined_plot_path}'.")

    except Exception as e:
        logger.error(f"An error occurred while creating plots: {e}")
        raise
