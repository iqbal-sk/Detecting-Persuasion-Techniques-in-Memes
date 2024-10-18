import numpy as np
import torch
import math
import torch.nn as nn
from config.logger import get_logger


logger = get_logger(__name__)


class HierarchicalLoss:
    """Logics to calculate the loss of the model.
    """

    def __init__(self, persuasion_techniques, hierarchical_labels, id2label,
                 device, total_levels=5, alpha=1,
                 beta=1, threshold=0.6):
        """Param init.
        """
        logger.info("Initializing HierarchicalLoss.")
        self.total_levels = total_levels
        self.alpha = alpha
        self.beta = beta
        self.p_loss = np.linspace(2, 1.1, 5)
        self.device = device
        logger.debug(f"Total levels: {self.total_levels}, Alpha: {self.alpha}, Beta: {self.beta}, P_loss: {self.p_loss}")
        
        try:
            self.level_one_labels, self.level_two_labels = persuasion_techniques['Level 1'], persuasion_techniques['Level 2']
            self.level_three_labels = persuasion_techniques['Level 3']
            self.level_four_labels = persuasion_techniques['Level 4']
            self.level_five_labels = persuasion_techniques['Level 5']
            logger.debug("Assigned persuasion techniques labels for all levels.")
        except KeyError as e:
            logger.error(f"Missing persuasion technique labels: {e}")
            raise
        
        self.hierarchical_labels = hierarchical_labels
        self.id2label = id2label
        self.threshold = threshold
        logger.info("HierarchicalLoss initialization complete.")

    # def words_to_indices(self):
    #     '''Convert the classes from words to indices.
    #     '''
    #     numeric_hierarchy = {}
    #     for k, v in self.hierarchical_labels.items():
    #         numeric_hierarchy[self.level_one_labels.index(k)] = [self.level_two_labels.index(i) for i in v]
    #
    #     return numeric_hierarchy

    def check_hierarchy(self, current_level, previous_level, level):
        """Check if the predicted class at level l is a children of the class predicted at level l-1 for the entire batch.
        """
        # logger.debug(f"Checking hierarchy between Level {level} and Level {level - 1}.")
        level_key = f'Level {level}'

        dl_array = []

        for i in range(len(current_level)):
            dl = 0
            current_labels = current_level[i]
            previous_labels = previous_level[i]

            allowed_parents = []
            for label in current_labels:
                parents = self.hierarchical_labels.get(level_key, {}).get(label, [])
                allowed_parents.extend(parents)
                # logger.debug(f"Label '{label}' at {level_key} has allowed parents: {parents}")

            allowed_parents = set(allowed_parents)
            # logger.debug(f"Allowed parents for Level {level}: {allowed_parents}")

            for parent_label in previous_labels:
                if parent_label not in allowed_parents:
                    dl += 1
                    # logger.debug(f"Parent label '{parent_label}' not allowed for Level {level}.")

            loss_value = math.log(1 + math.exp(dl))
            dl_array.append(loss_value)
            # logger.debug(f"Instance {i}: dl={dl}, log(1+exp(dl))={loss_value}")

        dloss_tensor = torch.FloatTensor(dl_array).to(self.device)
        # logger.debug("Completed hierarchy check.")
        return dloss_tensor

    def calculate_lloss(self, predictions, true_labels):
        """Calculates the layer loss.
        """
        logger.debug("Calculating layer loss (lloss).")
        lloss = 0
        cross_entropy_loss_fn = nn.CrossEntropyLoss()
        binary_cross_entropy_loss_fn = nn.BCELoss()
        for l in range(self.total_levels):
            if l == 0 or l == 1:
                loss = cross_entropy_loss_fn(predictions[l], true_labels[l])
                # logger.debug(f"Level {l+1}: CrossEntropyLoss={loss.item()}")
                lloss += loss
            else:
                loss = binary_cross_entropy_loss_fn(predictions[l], true_labels[l])
                # logger.debug(f"Level {l+1}: BCELoss={loss.item()}")
                lloss += loss

        total_lloss = self.alpha * lloss
        logger.debug(f"Total layer loss (lloss): {total_lloss.item()}")
        return total_lloss

    def decode_labels(self, id_labels, level):
        # logger.debug(f"Decoding labels for Level {level}.")
        id2label_level = self.id2label.get(f'Level {level}', {})
        decoded_labels = []
        for id_label in id_labels:
            labels = [id2label_level[id] for id in id_label if id in id2label_level]
            decoded_labels.append(labels)
            # logger.debug(f"Decoded labels for ID {id_label}: {labels}")
        return decoded_labels

    def calculate_dloss(self, predictions, true_labels):
        """Calculate the dependence loss.
        """
        logger.debug("Calculating dependence loss (dloss).")
        dloss = 0
        for cur_level in range(1, self.total_levels):
            prev_level = cur_level - 1
            # logger.debug(f"Calculating dloss between Level {prev_level + 1} and Level {cur_level + 1}.")

            if cur_level == 1:
                current_lvl_pred = [[instance.argmax().item()] for instance in predictions[cur_level]]
                # logger.debug(f"Level {cur_level + 1} predictions (argmax): {current_lvl_pred}")
            else:
                current_lvl_pred = [torch.nonzero(instance > self.threshold).squeeze().tolist() for instance in predictions[cur_level]]
                # logger.debug(f"Level {cur_level + 1} predictions (thresholded): {current_lvl_pred}")

            if prev_level == 0 or prev_level == 1:
                prev_lvl_pred = [[instance.argmax().item()] for instance in predictions[prev_level]]
                # logger.debug(f"Level {prev_level + 1} predictions (argmax): {prev_lvl_pred}")
            else:
                prev_lvl_pred = [torch.nonzero(instance > self.threshold).squeeze().tolist() for instance in predictions[prev_level]]
                # logger.debug(f"Level {prev_level + 1} predictions (thresholded): {prev_lvl_pred}")

            current_lvl_pred = [[item] if isinstance(item, int) else item for item in current_lvl_pred]
            prev_lvl_pred = [[item] if isinstance(item, int) else item for item in prev_lvl_pred]
            # logger.debug(f"Formatted Level {cur_level + 1} predictions: {current_lvl_pred}")
            # logger.debug(f"Formatted Level {prev_level + 1} predictions: {prev_lvl_pred}")

            current_lvl_labels = self.decode_labels(current_lvl_pred, cur_level + 1)
            prev_lvl_labels = self.decode_labels(prev_lvl_pred, prev_level + 1)
            # logger.debug(f"Decoded Level {cur_level + 1} labels: {current_lvl_labels}")
            # logger.debug(f"Decoded Level {prev_level + 1} labels: {prev_lvl_labels}")

            D_l = self.check_hierarchy(current_lvl_labels, prev_lvl_labels, cur_level)
            # logger.debug(f"D_l for Level {cur_level + 1}: {D_l}")

            current_lvl_scores = predictions[cur_level]
            prev_lvl_scores = predictions[prev_level]

            if cur_level == 1:
                predicted_classes = torch.argmax(current_lvl_scores, dim=1)
                current_lvl_pred_one_hot = torch.nn.functional.one_hot(predicted_classes, num_classes=current_lvl_scores.shape[1]).float()
                # logger.debug(f"Level {cur_level + 1} one-hot predictions: {current_lvl_pred_one_hot}")
            else:
                current_lvl_pred_one_hot = (predictions[cur_level] > self.threshold).float()
                # logger.debug(f"Level {cur_level + 1} binary predictions: {current_lvl_pred_one_hot}")

            if prev_level == 0 or prev_level == 1:
                predicted_classes = torch.argmax(prev_lvl_scores, dim=1)
                prev_lvl_pred_one_hot = torch.nn.functional.one_hot(predicted_classes, num_classes=prev_lvl_scores.shape[1]).float()
                # logger.debug(f"Level {prev_level + 1} one-hot predictions: {prev_lvl_pred_one_hot}")
            else:
                prev_lvl_pred_one_hot = (predictions[prev_level] > self.threshold).float()
                # logger.debug(f"Level {prev_level + 1} binary predictions: {prev_lvl_pred_one_hot}")

            l_prev = torch.where(prev_lvl_pred_one_hot == true_labels[prev_level], 
                                 torch.zeros_like(prev_lvl_pred_one_hot), 
                                 torch.ones_like(prev_lvl_pred_one_hot))
            l_prev = l_prev.sum(dim=1).log().to(self.device)
            # logger.debug(f"l_prev: {l_prev}")

            l_curr = torch.where(current_lvl_pred_one_hot == true_labels[cur_level], 
                                 torch.zeros_like(current_lvl_pred_one_hot), 
                                 torch.ones_like(current_lvl_pred_one_hot))
            l_curr = l_curr.sum(dim=1).log().to(self.device)
            # logger.debug(f"l_curr: {l_curr}")

            term = torch.pow(torch.tensor(self.p_loss[prev_level], dtype=torch.float).to(self.device),
                             D_l + l_prev) * \
                   torch.pow(torch.tensor(self.p_loss[cur_level], dtype=torch.float).to(self.device),
                             D_l + l_curr)
            # logger.debug(f"Term for dloss: {term}")

            dloss += torch.sum(term, dim=-1)
            # logger.debug(f"Accumulated dloss: {dloss}")

        total_dloss = self.beta * dloss
        logger.debug(f"Total dependence loss (dloss): {total_dloss.item()}")
        return total_dloss
