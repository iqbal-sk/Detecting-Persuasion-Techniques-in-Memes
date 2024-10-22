import torch
import numpy as np
import pandas as pd
import re
from config.logger import get_logger

logger = get_logger(__name__)


def create_indexed_dictionaries(techniques):
    logger.debug("Creating indexed dictionaries from techniques.")
    indexed_dictionaries = {}
    for level, items in techniques.items():
        indexed_dict = {label: index for index, label in enumerate(items)}
        indexed_dictionaries[level] = indexed_dict
        logger.debug(f"Indexed dictionary created for {level}: {indexed_dict}")
    logger.info("Indexed dictionaries creation complete.")
    return indexed_dictionaries


def replace_newlines_with_fullstop(text):
    # logger.debug("Replacing newlines with full stops in text.")
    text = re.sub(r'(\\n)+', '. ', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # logger.debug("Removed URLs from text.")

    # Remove any remaining backslashes
    text = text.replace("\\", " ")
    # logger.debug("Removed remaining backslashes from text.")

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # logger.debug("Removed extra spaces from text.")

    return text


def get_device():
    logger.info("Determining the computation device.")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA device: {cuda_name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Metal Performance Shaders (MPS) device.")
    else:
        device = torch.device("cpu")
        logger.info("CUDA and MPS not available. Using CPU.")

    return device


def get_labels(id2label, ids, pred_3, pred_4, pred_5, format=None):
    logger.debug("Decoding labels from predictions.")
    predictions = []

    pred_3 = [np.where(row == 1)[0].tolist() for row in pred_3]
    pred_4 = [np.where(row == 1)[0].tolist() for row in pred_4]
    pred_5 = [np.where(row == 1)[0].tolist() for row in pred_5]

    def decode_labels(id2label, preds, level):
        # logger.debug(f"Decoding labels for Level {level}.")
        decoded_labels = []
        level_key = f'Level {level}'

        for pred in preds:
            labels = []
            for id in pred:
                label = id2label.get(level_key, {}).get(id)
                if label:
                    labels.append(label)
            decoded_labels.append(labels)
            # logger.debug(f"Decoded labels for ID: {pred} -> {labels}")

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
            # logger.debug(f"Prediction for ID {id}: {labels}")
        else:
            forma = '{:0' + str(format) + 'd}'
            formatted_id = forma.format(id)
            predictions.append({'id': formatted_id, 'labels': labels})
            # logger.debug(f"Prediction for formatted ID {formatted_id}: {labels}")

    logger.debug("Label decoding complete.")
    return predictions

def process_json_openai(file_path):
    logger.info(f"Processing JSON file for OpenAI feature extraction: {file_path}")

    try:
        data_df = pd.read_json(file_path)
        logger.debug("JSON file loaded into DataFrame successfully.")
    except ValueError as e:
        logger.error(f"Failed to load JSON file {file_path}: {e}")
        raise

    data_df['cleaned_text'] = data_df['text'].apply(replace_newlines_with_fullstop)
    logger.debug("Cleaned text column created.")

    if 'link' in data_df.columns:
        data_df.drop(columns=['link'], inplace=True)
        logger.debug("Dropped 'link' column from DataFrame.")

    return data_df

def process_json(file_path, techniques_to_level, hierarchy, is_testdata):
    logger.info(f"Processing JSON file: {file_path}")
    try:
        data_df = pd.read_json(file_path)
        logger.debug("JSON file loaded into DataFrame successfully.")
    except ValueError as e:
        logger.error(f"Failed to load JSON file {file_path}: {e}")
        raise

    data_df['cleaned_text'] = data_df['text'].apply(replace_newlines_with_fullstop)
    logger.debug("Cleaned text column created.")

    if 'link' in data_df.columns:
        data_df.drop(columns=['link'], inplace=True)
        logger.debug("Dropped 'link' column from DataFrame.")

    if is_testdata:
        logger.info("Test data detected. Skipping label processing.")
        return data_df

    # Initialize empty lists for each level
    for level in ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']:
        data_df[level] = pd.Series([[] for _ in range(len(data_df))], index=data_df.index)
        # logger.debug(f"Initialized empty list for {level}.")

    # Assign labels to appropriate levels
    logger.info("Assigning labels to hierarchical levels.")
    for index, row in data_df.iterrows():
        for label in row['labels']:
            levels = techniques_to_level.get(label, [])
            if levels:
                for level in levels:
                    data_df.at[index, level].append(label)
                    # logger.debug(f"Appended label '{label}' to {level} for index {index}.")

    # Propagate labels from lower to higher levels based on hierarchy
    logger.info("Propagating labels up the hierarchy.")
    for index, row in data_df.iterrows():
        for label in row['Level 5']:
            parent_label = hierarchy['Level 4'].get(label, [None])[0]
            if parent_label:
                row['Level 4'].append(parent_label)
                # logger.debug(f"Appended parent label '{parent_label}' to Level 4 for index {index}.")

        for label in row['Level 4']:
            parent_label = hierarchy['Level 3'].get(label, [None])[0]
            if parent_label:
                row['Level 3'].append(parent_label)
                # logger.debug(f"Appended parent label '{parent_label}' to Level 3 for index {index}.")

        for label in row['Level 3']:
            parent_label = hierarchy['Level 2'].get(label, [None])[0]
            if parent_label:
                row['Level 2'].append(parent_label)
                # logger.debug(f"Appended parent label '{parent_label}' to Level 2 for index {index}.")

        for label in row['Level 2']:
            parent_label = hierarchy['Level 1'].get(label, [None])[0]
            if parent_label:
                row['Level 1'].append(parent_label)
                # logger.debug(f"Appended parent label '{parent_label}' to Level 1 for index {index}.")

    # Remove duplicate labels and convert lists to unique lists
    cols = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
    for col in cols:
        data_df[col] = data_df[col].apply(lambda x: list(set(x)))
        logger.debug(f"Converted {col} lists to unique lists.")

    logger.info("JSON processing complete.")
    return data_df