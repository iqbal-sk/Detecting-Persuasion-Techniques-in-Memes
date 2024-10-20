from config import config
from src.FeatureExtractor.Textual import text_features, openai_embeddings
from src.models.factory import ModelFactory
from src.datasets.Datasets import (TextOnlyDataSet, TextOnlyTestDataSet)
from src.utils.utils import process_json, get_device
from src.utils.label_decoding import (techniques_to_level_1, hierarchy_1)

from Trainer import Trainer
from config.logger import get_logger
from config import DEVICE


logger = get_logger(__name__)


if __name__ == "__main__":

    ### ------------------------------------ Feature Extraction ------------------------------------

    task = config.task
    training_model_name = config.training.model

    train_file = config.training.dataset_file
    val_file = config.evaluation.dataset_file

    train_features_file = config.feature_extractor.train_text_features_outfile
    val_features_file = config.feature_extractor.val_text_features_outfile

    tokenizer, text_model = None, None
    feature_extractor_device = get_device()

    if training_model_name == 'mBERT':
        from transformers import BertTokenizer, BertModel

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        text_model = BertModel.from_pretrained("bert-base-multilingual-uncased")

    elif training_model_name == 'XLMRoBERTa':
        from transformers import XLMRobertaTokenizer, XLMRobertaModel

        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        text_model = XLMRobertaModel.from_pretrained("xlm-roberta-large")

    elif training_model_name == 'XLNet':
        from transformers import XLNetModel, XLNetTokenizer

        tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
        text_model = XLNetModel.from_pretrained("xlnet-large-cased")

    elif training_model_name == 'OpenAiSmall':

        openai_embeddings(train_file, 'text-embedding-3-small', train_features_file, task)
        openai_embeddings(val_file, 'text-embedding-3-small', val_features_file, task)

    elif training_model_name == 'OpenAiLarge':

        openai_embeddings(train_file, 'text-embedding-3-large', train_features_file, task)
        openai_embeddings(val_file, 'text-embedding-3-large', val_features_file, task)


    if training_model_name not in ['OpenAiSmall', 'OpenAiLarge']:
        text_features(train_file, tokenizer, text_model, train_features_file, device=feature_extractor_device,
                      is_testdata=False, subtask=task)

        text_features(val_file, tokenizer, text_model, val_features_file, device=feature_extractor_device,
                      is_testdata=False, subtask=task)

    ### ------------------------------------ Preparing Datasets ------------------------------------

    train_data = process_json(train_file, techniques_to_level_1, hierarchy_1, is_testdata=False)
    validation_data = process_json(val_file, techniques_to_level_1, hierarchy_1, is_testdata=False)

    training_dataset = TextOnlyDataSet(train_data, train_features_file)
    validation_dataset = TextOnlyDataSet(validation_data, val_features_file)

    ### ------------------------------------ Model Training ------------------------------------

    model = ModelFactory.get_model(config.training.model)

    trainer = Trainer(task,
                      model,
                      training_config=config.training,
                      training_dataset=training_dataset,
                      validation_dataset=validation_dataset,
                      device=DEVICE)

    trainer.train()