from config import config
from src.FeatureExtractor.Images import extract_image_features
from src.FeatureExtractor.Textual import text_features, openai_embeddings
from src.models.factory import ModelFactory
from src.datasets.Datasets import (MemeDataSet, MemeTestDataSet)
from src.utils.utils import process_json, get_device
from src.utils.label_decoding import (techniques_to_level_2a, hierarchy_subtask_2a)

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

    train_text_features_file = config.feature_extractor.train_text_features_outfile
    val_text_features_file = config.feature_extractor.val_text_features_outfile

    train_image_features_file = config.feature_extractor.train_image_features_outfile
    val_image_features_file = config.feature_extractor.val_image_features_outfile


    tokenizer, text_model = None, None
    feature_extractor_device = get_device()

    if config.feature_extractor.text_model == 'mBERT':
        from transformers import BertTokenizer, BertModel

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        text_model = BertModel.from_pretrained("bert-base-multilingual-uncased")
    elif config.feature_extractor.text_model == 'OpenAiSmall':
        pass
        openai_embeddings(train_file, 'text-embedding-3-small', train_text_features_file, task)
        openai_embeddings(val_file, 'text-embedding-3-small', val_text_features_file, task)
    elif config.feature_extractor.text_model == 'OpenAiLarge':
        pass
        openai_embeddings(train_file, 'text-embedding-3-large', train_text_features_file, task)
        openai_embeddings(val_file, 'text-embedding-3-large', val_text_features_file, task)

    if config.feature_extractor.text_model not in ['OpenAiSmall', 'OpenAiLarge']:
        text_features(train_file, tokenizer, text_model, train_text_features_file, device=feature_extractor_device,
                      is_testdata=False, subtask=task)

        text_features(val_file, tokenizer, text_model, val_text_features_file, device=feature_extractor_device,
                      is_testdata=False, subtask=task)


    extract_image_features(model=config.feature_extractor.image_model,
                           directory=config.training.images_directory,
                           output_file_path=train_image_features_file,
                           device=feature_extractor_device)

    extract_image_features(model=config.feature_extractor.image_model,
                           directory=config.evaluation.images_directory,
                           output_file_path=val_image_features_file,
                           device=feature_extractor_device)


    ### ------------------------------------ Preparing Datasets ------------------------------------

    train_data = process_json(train_file, techniques_to_level_2a, hierarchy_subtask_2a, is_testdata=False)
    validation_data = process_json(val_file, techniques_to_level_2a, hierarchy_subtask_2a, is_testdata=False)

    training_dataset = MemeDataSet(train_data, train_text_features_file, train_image_features_file)
    validation_dataset = MemeDataSet(validation_data, val_text_features_file, val_image_features_file)

    ### ------------------------------------ Model Training ------------------------------------

    text_features_size, image_features_size = None, None

    if config.feature_extractor.text_model == 'OpenAiSmall':
        text_features_size = 1536
    elif config.feature_extractor.text_model == 'OpenAiLarge':
        text_features_size = 3072
    elif config.feature_extractor.text_model == 'mBERT':
        text_features_size = 768

    if config.feature_extractor.image_model == 'CLIP':
        image_features_size = 512
    elif config.feature_extractor.image_model == 'ResNet50':
        image_features_size = 2048

    model = ModelFactory.get_model(config.training.model, img_feature_size=image_features_size,
                                   text_feature_size=text_features_size)

    trainer = Trainer(task, model,
                      training_config=config.training,
                      training_dataset=training_dataset,
                      validation_dataset=validation_dataset,
                      device=DEVICE)

    trainer.train()