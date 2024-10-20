import os
import json
import torch

from src.utils.label_decoding import (id2label_1, id2label_subtask_2a,
                                      hierarchy_1, hierarchy_subtask_2a,
                                      persuasion_techniques_1, persuasion_techniques_2a)

from src.utils.HierarchicalLoss import HierarchicalLoss
from src.utils.utils import get_labels
from src.evaluation.evaluation_utils import evaluate_h

from config import config, DEVICE
from config.logger import get_logger

logger = get_logger(__name__)

class Evaluation:
    def __init__(self):
        self.task = config.task
        logger.info(f"Initializing Evaluation for task: {self.task}")

        if self.task == 'subtask1':
            logger.debug("Configuring HierarchicalLoss for subtask1")
            self.HL = HierarchicalLoss(id2label=id2label_1, hierarchical_labels=hierarchy_1,
                                       persuasion_techniques=persuasion_techniques_1, device=DEVICE)
            self.id2leaf_label = id2label_1
            logger.info("HierarchicalLoss configured for subtask1")
        else:
            logger.debug("Configuring HierarchicalLoss for subtask2a")
            self.HL = HierarchicalLoss(id2label=id2label_subtask_2a, hierarchical_labels=hierarchy_subtask_2a,
                                       persuasion_techniques=persuasion_techniques_2a, device=DEVICE)
            self.id2leaf_label = id2label_subtask_2a
            logger.info("HierarchicalLoss configured for subtask2a")

    def get_loss_obj(self):
        return self.HL

    """
    ToDo: Can we eliminate format and validation from function parameters?
    Is it good? """

    def evaluate_model(self, model, dataloader, format=None, validation=False):
        logger.debug("Starting model evaluation")
        model.eval()
        predictions = []

        pred_file_path = config.evaluation.prediction_output_path
        gold_file_path = config.evaluation.dataset_file
        threshold = config.evaluation.hyperparameters.threshold

        total_loss = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader, 1):
                if not isinstance(batch['id'], list):
                    ids = batch['id'].detach().cpu().numpy().tolist()
                else:
                    ids = batch['id']

                text_features = batch['text_features'].to(DEVICE)

                if self.task == 'subtask2a':
                    image_features = batch['image_features'].to(DEVICE)

                try:
                    if self.task == 'subtask1':
                        pred_1, pred_2, pred_3, pred_4, pred_5 = model(text_features)
                    else:
                        pred_1, pred_2, pred_3, pred_4, pred_5 = model(text_features, image_features)
                except Exception as e:
                    logger.error(f"Model prediction failed at batch {batch_idx + 1}: {e}")
                    continue

                if validation:
                    y_1 = batch['level_1_target'].to(DEVICE)
                    y_2 = batch['level_2_target'].to(DEVICE)
                    y_3 = batch['level_3_target'].to(DEVICE)
                    y_4 = batch['level_4_target'].to(DEVICE)
                    y_5 = batch['level_5_target'].to(DEVICE)

                    try:
                        dloss = self.HL.calculate_dloss([pred_1, pred_2, pred_3, pred_4, pred_5],
                                                       [y_1, y_2, y_3, y_4, y_5])
                        lloss = self.HL.calculate_lloss([pred_1, pred_2, pred_3, pred_4, pred_5],
                                                       [y_1, y_2, y_3, y_4, y_5])
                        total_loss += (dloss + lloss).detach().cpu().item()
                        logger.debug(f"Batch {batch_idx + 1}: DLoss={dloss.item()}, LLoss={lloss.item()}")
                    except Exception as e:
                        logger.error(f"Loss calculation failed at batch {batch_idx + 1}: {e}")

                pred_3 = (pred_3.cpu().detach().numpy() > threshold).astype(int)
                pred_4 = (pred_4.cpu().detach().numpy() > threshold).astype(int)
                pred_5 = (pred_5.cpu().detach().numpy() > threshold).astype(int)

                try:
                    batch_predictions = get_labels(self.id2leaf_label, ids, pred_3, pred_4, pred_5, format)
                    predictions += batch_predictions
                    logger.debug(f"Batch {batch_idx + 1}: Predictions extracted successfully")
                except Exception as e:
                    logger.error(f"Label decoding failed at batch {batch_idx + 1}: {e}")

            # Writing JSON data
            try:
                os.makedirs(os.path.dirname(pred_file_path), exist_ok=True)
                with open(pred_file_path, 'w') as f:
                    json.dump(predictions, f, indent=4)
                logger.debug(f"Predictions saved to {pred_file_path}")
            except Exception as e:
                logger.error(f"Failed to save predictions to {pred_file_path}: {e}")

            if gold_file_path is None:
                logger.warning("Gold file path is None. Evaluation metrics will not be computed.")
                return

            try:
                prec_h, rec_h, f1_h = evaluate_h(pred_file_path, gold_file_path)
                logger.debug(f"Evaluation Metrics - Precision: {prec_h}, Recall: {rec_h}, F1 Score: {f1_h}")
            except Exception as e:
                logger.error(f"Failed to evaluate predictions against gold data: {e}")
                return

            if validation:
                average_loss = total_loss / len(dataloader)
                # logger.debug(f"Validation Metrics - Precision: {prec_h}, Recall: {rec_h}, F1 Score: {f1_h}, Average Loss: {average_loss}")
                return prec_h, rec_h, f1_h, average_loss

            return prec_h, rec_h, f1_h
