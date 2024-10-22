from .base_model import BaseModel

import src.models as models
from config.logger import get_logger

logger = get_logger(__name__)


class ModelFactory:

    @staticmethod
    def get_model(model_name: str, **kwargs) -> BaseModel:
        """
        Dynamically retrieves and instantiates a model class based on the model_name.

        :param model_name: Name of the model class to instantiate.

        :return: An instance of the specified model.
        :raises ValueError: If the model cannot be found or instantiated.
        """
        try:
            # Retrieve the model class from the models package
            model_class = getattr(models, model_name)
            logger.info(f"Retrieved '{model_name}' class from models package.")
        except AttributeError as e:
            logger.error(f"Model '{model_name}' not found in models package: {e}")
            raise ValueError(f"Model '{model_name}' is not recognized.") from e

        try:
            model_instance = model_class(**kwargs)
            logger.info(f"Instantiated model '{model_name}'")
            return model_instance
        except Exception as e:
            logger.error(f"Error instantiating model '{model_name}': {e}")
            raise ValueError(f"Could not instantiate model '{model_name}'.") from e
