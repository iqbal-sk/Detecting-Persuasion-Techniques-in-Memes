'''Hierarchical Loss Network
'''
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
class HierarchicalLoss:
    '''Logics to calculate the loss of the model.
    '''

    def __init__(self, persuasion_techniques, hierarchical_labels, id2label,
                 device=torch.device('cuda'), total_level=5, alpha=1,
                 beta=1, threshold=0.6):
        '''Param init.
        '''
        self.total_level = total_level
        self.alpha = alpha
        self.beta = beta
        self.p_loss = np.linspace(2, 1.1, 5)
        self.device = device
        self.level_one_labels, self.level_two_labels = persuasion_techniques['Level 1'], persuasion_techniques[
            'Level 2']
        self.level_three_labels = persuasion_techniques['Level 3']
        self.level_four_labels = persuasion_techniques['Level 4']
        self.level_five_labels = persuasion_techniques['Level 5']
        self.hierarchical_labels = hierarchical_labels
        # self.numeric_hierarchy = self.words_to_indices()
        self.id2label = id2label

        self.threshold = threshold

    # def words_to_indices(self):
    #     '''Convert the classes from words to indices.
    #     '''
    #     numeric_hierarchy = {}
    #     for k, v in self.hierarchical_labels.items():
    #         numeric_hierarchy[self.level_one_labels.index(k)] = [self.level_two_labels.index(i) for i in v]
    #
    #     return numeric_hierarchy

    def check_hierarchy(self, current_level, previous_level, level):
        '''Check if the predicted class at level l is a children of the class predicted at level l-1 for the entire batch.
        '''

        level = f'Level {level}'

        # check using the dictionary whether the current level's prediction belongs to the superclass (prediction from the prev layer)

        dl_array = []

        for i in range(len(current_level)):
            dl = 0
            current_labels = current_level[i]
            previous_labels = previous_level[i]

            allowed_parents = []

            for label in current_labels:
                allowed_parents += self.hierarchical_labels[level][label]

            allowed_parents = set(allowed_parents)
            for parent_label in previous_labels:
                if not parent_label in allowed_parents:
                    dl += 1

            dl_array.append(math.log(1 + math.exp(dl)))

        # bool_tensor = [not previous_level[i] in self.hierarchical_labels[level][previous_level[i].item()] for i in range(previous_level.size()[0])]

        return torch.FloatTensor(dl_array).to(self.device)

    def calculate_lloss(self, predictions, true_labels):
        '''Calculates the layer loss.
        '''

        lloss = 0
        cross_entropy_loss_fn = nn.CrossEntropyLoss()
        binary_cross_entropy_loss_fn = nn.BCELoss()
        for l in range(self.total_level):
            if l == 0 or l == 1:
                lloss += cross_entropy_loss_fn(predictions[l], true_labels[l])
            else:
                lloss += binary_cross_entropy_loss_fn(predictions[l], true_labels[l])

        return self.alpha * lloss

    def decode_labels(self, id_labels, level):
        id2label = self.id2label[f'Level {level}']
        # print(id2label)
        decoded_labels = []
        for id_label in id_labels:
            # print(id_label)
            decoded_labels.append([id2label[id] for id in id_label])

        return decoded_labels

    def calculate_dloss(self, predictions, true_labels):
        '''Calculate the dependence loss.
        '''

        dloss = 0
        for cur_level in range(1, self.total_level):
            prev_level = cur_level - 1

            if cur_level == 1:
                current_lvl_pred = [[instance.argmax().item()] for instance in predictions[cur_level]]
            else:
                current_lvl_pred = [torch.nonzero(instance > self.threshold).squeeze().tolist() for instance in predictions[cur_level]]

            if prev_level == 0 or prev_level == 1:
                prev_lvl_pred = [[instance.argmax().item()] for instance in predictions[prev_level]]
            else:
                prev_lvl_pred = [torch.nonzero(instance > self.threshold).squeeze().tolist() for instance in predictions[prev_level]]

            current_lvl_pred = [[item] if isinstance(item, int) else item for item in current_lvl_pred]
            prev_lvl_pred = [[item] if isinstance(item, int) else item for item in prev_lvl_pred]

            # print(current_lvl_pred)

            current_lvl_labels = self.decode_labels(current_lvl_pred, cur_level + 1)
            prev_lvl_labels = self.decode_labels(prev_lvl_pred, prev_level + 1)

            # current_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l]), dim=1)
            # prev_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l-1]), dim=1)

            D_l = self.check_hierarchy(current_lvl_labels, prev_lvl_labels, cur_level)

            current_lvl_scores = predictions[cur_level]
            prev_lvl_scores = predictions[prev_level]

            if cur_level == 1:
                predicted_classes = torch.argmax(current_lvl_scores, dim=1)
                current_lvl_pred = torch.nn.functional.one_hot(predicted_classes, num_classes=current_lvl_scores.shape[1])
            else:
                current_lvl_pred = (predictions[cur_level] > self.threshold)

            if prev_level == 0 or prev_level == 1:
                predicted_classes = torch.argmax(prev_lvl_scores, dim=1)
                prev_lvl_pred = torch.nn.functional.one_hot(predicted_classes, num_classes=prev_lvl_scores.shape[1])
            else:
                prev_lvl_pred = (predictions[prev_level] > self.threshold)

            l_prev = torch.where(prev_lvl_pred == true_labels[prev_level], torch.FloatTensor([0]).to(self.device),
                                 torch.FloatTensor([1]).to(self.device))
            l_prev = l_prev.sum(dim=1).log()

            l_curr = torch.where(current_lvl_pred == true_labels[cur_level], torch.FloatTensor([0]).to(self.device),
                                 torch.FloatTensor([1]).to(self.device))
            l_curr = l_curr.sum(dim=1).log()

            # print(l_prev.shape)
            # print(D_l.shape)

            dloss += torch.sum(torch.pow(self.p_loss[prev_level], D_l + l_prev) * torch.pow(self.p_loss[cur_level], D_l + l_curr), dim=-1)

        return self.beta * dloss