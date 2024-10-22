import os
import torch
import pickle
import numpy as np

from openai import OpenAI
from src.utils.utils import process_json, process_json_openai
from src.utils.label_decoding import (techniques_to_level_1, hierarchy_1,
                                      techniques_to_level_2a, hierarchy_subtask_2a)

from config.logger import get_logger
from config import config

logger = get_logger(__name__)

def text_features(file_path, tokenizer, text_model, output_file_path, device,
                 is_testdata=False, subtask='subtask1'):

    text_model.to(device)
    logger.info(f"Textual extraction model moved to {device}")

    text_model.eval()

    features_dict = {}
    logger.info(f"Starting text feature 2 for file: {file_path} with subtask: {subtask}")

    if subtask == 'subtask1':
        logger.debug("Processing data for subtask1")
        data = process_json(file_path, techniques_to_level_1, hierarchy_1, is_testdata)
    else:
        logger.debug("Processing data for subtask2a")
        data = process_json(file_path, techniques_to_level_2a, hierarchy_subtask_2a, is_testdata)

    batch_size = 256

    ids = data['id'].tolist()
    texts = data['cleaned_text'].tolist()

    step = 0

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_texts = texts[i:i + batch_size]

        # Tokenize the batch of texts
        encoded_input = tokenizer(
            batch_texts,
            return_tensors='pt',
            add_special_tokens=True,
            max_length=128,
            truncation=True,
            padding='max_length'
        ).to(device)

        # Compute embeddings without computing gradients
        with torch.no_grad():
            embeddings = text_model(**encoded_input)

        # Extract the embeddings for the [CLS] token (assuming you want the first token)
        # embeddings.last_hidden_state shape: [batch_size, sequence_length, hidden_size]
        batch_embeddings = embeddings.last_hidden_state[:, 0, :].detach().cpu().numpy()

        # Store the embeddings in the features dictionary
        for id, embedding in zip(batch_ids, batch_embeddings):
            features_dict[id] = embedding

        # Update the step counter
        step += len(batch_ids)

        # Logging progress
        if step % batch_size == 0:
            logger.info(f'Completed {step} steps')

    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(f'{output_file_path}', 'wb') as f:
            pickle.dump(features_dict, f)
        logger.info(f"Features extracted and stored in {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to save features to {output_file_path}: {e}")


def openai_embeddings(file_path, model, output_file_path, task):

    logger.info(f"Starting OpenAI embeddings extraction for task: {task}, model: {model}, file: {file_path}")

    try:
        client = OpenAI(api_key=config.openai.api_key)
        logger.info("OpenAI client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return

    features_dict = {}

    try:
        data = process_json_openai(file_path)
        logger.debug(f"Data processed successfully from {file_path}")
    except Exception as e:
        logger.error(f"Failed to process JSON from {file_path}: {e}")
        return

    step = 0

    for id, text in zip(data['id'], data['cleaned_text']):

        try:
            embedding_response = client.embeddings.create(input=[text], model=model)
            features_dict[id] = np.array(embedding_response.data[0].embedding, dtype=np.float32)
            logger.debug(f"Embedding created for ID: {id}")
        except Exception as e:
            logger.error(f'Exception for ID: {id} and text: "{text}". Error: {e}')

            if model == 'text-embedding-3-large':
                features_dict[id] = np.zeros(3072, dtype=np.float32)
            else:
                features_dict[id] = np.zeros(1536, dtype=np.float32)

        step += 1

        if step % 100 == 0:
            logger.info(f'Completed {step} steps')

    try:
        with open(f'{output_file_path}', 'wb') as f:
            pickle.dump(features_dict, f)
        logger.info(f"Features extracted and stored in {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to save features to {output_file_path}: {e}")

    return features_dict