import torch
from torch.utils.data import Dataset
import pickle


from config import config
from src.utils.label_decoding import indexed_persuasion_techniques_1, indexed_persuasion_techniques_2a


class TextOnlyDataSet(Dataset):
    def __init__(self, df, features_file):
        super(TextOnlyDataSet, self).__init__()
        self.data_df = df

        if config.task == 'subtask1':
            self.labels_at_level = indexed_persuasion_techniques_1
        else:
            self.labels_at_level = indexed_persuasion_techniques_2a

        self.features_file = features_file
        self.features_dict = None

        with open(self.features_file, 'rb') as f:
            self.features_dict = pickle.load(f)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        id = self.data_df.iloc[idx]['id']
        text = self.data_df.iloc[idx]['cleaned_text']

        level_1_target = self.encode(self.data_df.iloc[idx]['Level 1'], 1)
        level_2_target = self.encode(self.data_df.iloc[idx]['Level 2'], 2)
        level_3_target = self.encode(self.data_df.iloc[idx]['Level 3'], 3)
        level_4_target = self.encode(self.data_df.iloc[idx]['Level 4'], 4)
        level_5_target = self.encode(self.data_df.iloc[idx]['Level 5'], 5)

        return {'id': id,
                'text': text,
                'text_features': self.features_dict[id],
                'level_1_target': level_1_target,
                'level_2_target': level_2_target,
                'level_3_target': level_3_target,
                'level_4_target': level_4_target,
                'level_5_target': level_5_target}

    def encode(self, labels, level):
        level_ = f'Level {level}'

        target = torch.zeros(len(self.labels_at_level[level_]) + 1)

        for label in labels:
            label_idx = self.labels_at_level[level_][label]
            target[label_idx] = 1

        if len(labels) == 0:
            target[-1] = 1

        return target


class TextOnlyTestDataSet(Dataset):
    def __init__(self, df):
        super(TextOnlyTestDataSet, self).__init__()
        self.data_df = df
        self.features_file = config.feature_extractor.features_outfile
        self.features_dict = None

        with open(self.features_file, 'rb') as f:
            self.features_dict = pickle.load(f)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        id = self.data_df.iloc[idx]['id']
        text = self.data_df.iloc[idx]['cleaned_text']

        return {'id': id,
                'text': text,
                'text_features': self.features_dict[id]}


"""
ToDo: move text_features_file, image_features_file to the config.py
Can do later, First Deal with unimodal[TextOnlyDataSet]"""

class MemeDataSet(Dataset):
    def __init__(self, df, text_features_file, image_features_file):
        super(MemeDataSet, self).__init__()
        self.data_df = df

        if config.task == 'subtask1':
            self.labels_at_level = indexed_persuasion_techniques_1
        else:
            self.labels_at_level = indexed_persuasion_techniques_2a


        self.image_features = None
        self.text_features = None

        with open(image_features_file, 'rb') as f:
            self.image_features = pickle.load(f)

        with open(text_features_file, 'rb') as f:
            self.text_features = pickle.load(f)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        id = self.data_df.iloc[idx]['id']
        text = self.data_df.iloc[idx]['cleaned_text']
        image_name = self.data_df.iloc[idx]['image']

        level_1_target = self.encode(self.data_df.iloc[idx]['Level 1'], 1)
        level_2_target = self.encode(self.data_df.iloc[idx]['Level 2'], 2)
        level_3_target = self.encode(self.data_df.iloc[idx]['Level 3'], 3)
        level_4_target = self.encode(self.data_df.iloc[idx]['Level 4'], 4)
        level_5_target = self.encode(self.data_df.iloc[idx]['Level 5'], 5)

        image_features = self.image_features[image_name]
        text_features = self.text_features[id]

        return {
            'id': id,
            'text': text,
            'image_features': image_features,
            'text_features': text_features,
            'level_1_target': level_1_target,
            'level_2_target': level_2_target,
            'level_3_target': level_3_target,
            'level_4_target': level_4_target,
            'level_5_target': level_5_target
        }

    def encode(self, labels, level):
        level_ = f'Level {level}'
        target = torch.zeros(len(self.labels_at_level[level_]) + 1)

        for label in labels:
            label_idx = self.labels_at_level[level_][label]
            target[label_idx] = 1

        if len(labels) == 0:
            target[-1] = 1

        return target


"""
ToDo: move text_features_file, image_features_file to the config.py
Can do later, First Deal with unimodal[TextOnlyDataSet]"""

class MemeTestDataSet(Dataset):
    def __init__(self, df, text_features_file, image_features_file):
        super(MemeTestDataSet, self).__init__()
        self.data_df = df

        self.image_features = None
        self.text_features = None

        with open(image_features_file, 'rb') as f:
            self.image_features = pickle.load(f)

        with open(text_features_file, 'rb') as f:
            self.text_features = pickle.load(f)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        id = self.data_df.iloc[idx]['id']
        text = self.data_df.iloc[idx]['cleaned_text']
        image_name = self.data_df.iloc[idx]['image']
        text_features = self.text_features[id]

        return {'id': id,
                'text': text,
                'text_features': text_features,
                'image_features': self.image_features[image_name]}