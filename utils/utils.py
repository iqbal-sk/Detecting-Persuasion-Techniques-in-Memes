import torch
import numpy as np
import pandas as pd
import re


def create_indexed_dictionaries(techniques):
    indexed_dictionaries = {}
    for level, items in techniques.items():
        indexed_dict = {label: index for index, label in enumerate(items)}
        indexed_dictionaries[level] = indexed_dict
    return indexed_dictionaries


def replace_newlines_with_fullstop(text):
    text = re.sub(r'(\\n)+', '. ', text)

    # print(text)
    # print()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove any remaining backslashes
    text = text.replace("\\", " ")

    # print(text)
    # print()

    # Remove digits - You can comment this line if you want to keep numbers
    # text = re.sub(r'\d+', '', text)

    # Optionally, remove punctuation
    # text = text.translate(str.maketrans('', '', punctuation))

    # Convert text to lowercase
    # text = text.lower()

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # print(text)
    # print()

    return text

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    # Fallback to CPU if CUDA is not available
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device

def process_test_json(file_path):
    data_df = pd.read_json(file_path)
    data_df['cleaned_text'] = data_df['text'].apply(replace_newlines_with_fullstop)
    if 'link' in data_df.columns:
        data_df.drop(columns=['link'], inplace=True)

    return data_df
def get_labels(id2label, ids, pred_3, pred_4, pred_5, format):
    predictions = []

    pred_3 = [np.where(row == 1)[0].tolist() for row in pred_3]
    pred_4 = [np.where(row == 1)[0].tolist() for row in pred_4]
    pred_5 = [np.where(row == 1)[0].tolist() for row in pred_5]

    # print(f'pred_3: {pred_3}')
    # print(f'pred_4: {pred_4}')
    # print(f'pred 5: {pred_5}')

    def decode_labels(id2label, preds, level):
        decoded_labels = []
        level = f'Level {level}'

        for pred in preds:
            labels = []
            for id in pred:
                if id in id2label[level].keys():
                    labels.append(id2label[level][id])
            decoded_labels.append(labels)

        return decoded_labels

    lvl3_decoded_labels = decode_labels(id2label, pred_3, 3)
    lvl4_decoded_labels = decode_labels(id2label, pred_4, 4)
    lvl5_decoded_labels = decode_labels(id2label, pred_5, 5)

    for i in range(len(ids)):
        id = ids[i]
        labels = lvl3_decoded_labels[i] + lvl4_decoded_labels[i] + lvl5_decoded_labels[i]
        labels = list(set(labels))

        if format is None:
            predictions.append({'id': str(id), 'labels': labels})
        else:
            forma = '{:0' + str(format) + 'd}'
            predictions.append({'id': forma.format(id), 'labels': labels })

    return predictions


def process_json(file_path, techniques_to_level, hierarchy):
    data_df = pd.read_json(file_path)
    data_df['cleaned_text'] = data_df['text'].apply(replace_newlines_with_fullstop)
    if 'link' in data_df.columns:
        data_df.drop(columns=['link'], inplace=True)

    for level in ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']:
        data_df[level] = pd.Series([[] for _ in range(len(data_df))], index=data_df.index)

    for index, row in data_df.iterrows():
        for label in row['labels']:
            # Normalize the label to match the case of your dictionary keys
            # Find the level of the current label
            levels = techniques_to_level.get(label)
            # Append the label to the appropriate column if the level is found
            if len(levels):
                for level in levels:
                    data_df.at[index, level].append(label)

    for index, row in data_df.iterrows():
        for label in row['Level 5']:
            row['Level 4'].append(hierarchy['Level 4'][label][0])

        for label in row['Level 4']:
            row['Level 3'].append(hierarchy['Level 3'][label][0])

        for label in row['Level 3']:
            row['Level 2'].append(hierarchy['Level 2'][label][0])

        for label in row['Level 2']:
            row['Level 1'].append(hierarchy['Level 1'][label][0])

    cols = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
    for col in cols:
        data_df[col] = data_df[col].apply(set).apply(list)
        data_df[col] = data_df[col].apply(set).apply(list)

    return data_df