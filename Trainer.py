import torch
import os
from datetime import datetime

from torch.utils.data import DataLoader
from config.config import TrainingConfig
from config.logger import get_logger
from plotting import create_plots
from src.evaluation.evaluation import Evaluation
from config import config

logger = get_logger(__name__)


class Trainer:
    def __init__(self, task, model: torch.nn.Module, training_config: TrainingConfig, training_dataset, validation_dataset,
                 device: torch.device):
        self.task = task
        self.model = model
        self.training_config = training_config
        self.hyperparameters = training_config.hyperparameters

        self.num_epochs = self.hyperparameters.num_epochs
        self.learning_rate = self.hyperparameters.learning_rate
        self.beta1 = self.hyperparameters.beta1
        self.batch_size = self.hyperparameters.batch_size
        self.train_dataset = training_dataset
        self.val_dataset = validation_dataset
        self.evaluator = Evaluation()
        self.device = device

    def train(self):
        """
            Trains the given model using the provided datasets and configuration.

            Args:
                model (torch.nn.Module): The neural network model to train.
                training_dataset (torch.utils.data.Dataset): The dataset for  f training.
                validation_dataset (torch.utils.data.Dataset): The dataset for validation.
                train_config (dict): Configuration dictionary containing hyperparameters.
            """

        logger.info("Starting training process.")



        logger.debug(
            f"Training hyperparameters: num_epochs={self.num_epochs}, learning_rate={self.learning_rate}, beta1={self.beta1}, batch_size={self.batch_size}")

        # Initialize DataLoaders
        logger.info("Initializing DataLoaders.")
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        logger.debug(f"Number of training batches: {len(train_dataloader)}")
        logger.debug(f"Number of validation batches: {len(validation_dataloader)}")

        self.model.to(self.device)
        logger.debug("Model moved to device.")

        # Initialize optimizer
        logger.info("Initializing optimizer.")
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, 0.999)
        )
        logger.debug("Optimizer initialized with Adam.")

        # Initialize Hierarchical Loss
        logger.info("Initializing HierarchicalLoss.")
        Hierarchical_Loss = self.evaluator.get_loss_obj()
        logger.debug("HierarchicalLoss object created.")

        metrics = {
            'Epoch': [],
            'Training Loss': [],
            'Validation Loss': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': []
        }

        # Training Loop
        logger.info("Beginning training loop.")

        for epoch in range(self.num_epochs):
            # logger.info(f"Epoch {epoch + 1}/{num_epochs} started.")
            self.model.train()
            running_loss = 0.0

            for batch_idx, batch in enumerate(train_dataloader, 1):
                logger.debug(f"Processing batch {batch_idx}/{len(train_dataloader)}.")

                # Extract targets and embeddings
                y_1 = batch.get('level_1_target').to(self.device)
                y_2 = batch.get('level_2_target').to(self.device)
                y_3 = batch.get('level_3_target').to(self.device)
                y_4 = batch.get('level_4_target').to(self.device)
                y_5 = batch.get('level_5_target').to(self.device)

                embeddings = batch.get('text_features').to(self.device)

                if self.task == 'subtask2a':
                    image_features = batch.get('image_features').to(self.device)

                logger.debug(f"Batch {batch_idx}: Targets and embeddings moved to device.")

                optimizer.zero_grad()
                logger.debug(f"Batch {batch_idx}: Optimizer gradients zeroed.")


                # Forward pass
                if self.task == 'subtask1':
                    pred_1, pred_2, pred_3, pred_4, pred_5 = self.model(embeddings)
                else:
                    pred_1, pred_2, pred_3, pred_4, pred_5 = self.model(embeddings, image_features)
                logger.debug(f"Batch {batch_idx}: Forward pass completed.")

                # Calculate losses
                dloss = Hierarchical_Loss.calculate_dloss([pred_1, pred_2, pred_3, pred_4, pred_5],
                                                          [y_1, y_2, y_3, y_4, y_5])
                lloss = Hierarchical_Loss.calculate_lloss([pred_1, pred_2, pred_3, pred_4, pred_5],
                                                          [y_1, y_2, y_3, y_4, y_5])
                logger.debug(f"Batch {batch_idx}: dloss={dloss.item()}, lloss={lloss.item()}")

                # Total loss
                total_loss = lloss + dloss
                logger.debug(f"Batch {batch_idx}: Total loss={total_loss.item()}")

                # Backward pass and optimization
                total_loss.backward()
                optimizer.step()
                logger.debug(f"Batch {batch_idx}: Backward pass and optimizer step completed.")

                running_loss += total_loss.detach().item()
                logger.debug(f"Batch {batch_idx}: Running loss={running_loss}")

            # Average loss for the epoch
            epoch_loss = running_loss / len(train_dataloader)
            # logger.info(f"Epoch {epoch + 1}: Average Training Loss={epoch_loss:.4f}")

            # Validation
            logger.debug(f"Epoch {epoch + 1}: Starting validation.")
            prec_h, rec_h, f1_h, validation_loss = self.evaluator.evaluate_model(self.model, validation_dataloader,
                                                                             validation=True)
            # logger.info(
            #     f"Epoch {epoch + 1}: Average Training Loss={epoch_loss:.4f}, Validation Loss={validation_loss:.4f}, Precision={prec_h:.4f}, Recall={rec_h:.4f}, F1-Score={f1_h:.4f}")

            # Store metrics in-memory
            metrics['Epoch'].append(epoch + 1)
            metrics['Training Loss'].append(epoch_loss)
            metrics['Validation Loss'].append(validation_loss)
            metrics['Precision'].append(prec_h)
            metrics['Recall'].append(rec_h)
            metrics['F1-Score'].append(f1_h)

            # Periodic logging every 100 epochs
            if (epoch + 1) % 50 == 0:
                logger.info(
                    f"Epoch {epoch + 1}: Average Training Loss={epoch_loss:.4f}, Validation Loss={validation_loss:.4f}, Precision={prec_h:.4f}, Recall={rec_h:.4f}, F1-Score={f1_h:.4f}")

        # Generate and save plots
        create_plots(metrics)
        logger.info(f"Plots generated and saved.")

        model_save_dir = self.training_config.save_model_to

        os.makedirs(model_save_dir, exist_ok=True)

        # Append timestamp to the filename and add '.pth' extension
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        base_filename = config.training.model
        model_filename = f"{base_filename}_{timestamp}.pth"

        full_model_save_path = os.path.join(model_save_dir, model_filename)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': self.num_epochs,
            'metrics': metrics
        }, full_model_save_path)

        logger.info(f"Model saved to {full_model_save_path}")

        logger.info("Training process completed.")
