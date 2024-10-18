import yaml
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import os
import sys
from jinja2 import Environment, FileSystemLoader

@dataclass
class TrainingHyperparameters:
    alpha: float
    batch_size: int
    beta: float
    beta1: float
    learning_rate: float
    threshold: float
    dropout_rate: float
    num_epochs: int

@dataclass
class TrainingConfig:
    dataset_file: str
    images_directory: str
    model: str
    save_model_to: str
    hyperparameters: TrainingHyperparameters

@dataclass
class FeatureExtractorConfig:
    text_model: str
    image_model: str
    train_text_features_outfile: str
    val_text_features_outfile: str
    train_image_features_outfile: Optional[str] = None
    val_image_features_outfile: Optional[str] = None

@dataclass
class EvaluationHyperparameters:
    metric: str
    threshold: float

@dataclass
class EvaluationConfig:
    dataset_file: str
    images_directory: str
    hyperparameters: EvaluationHyperparameters
    prediction_output_path: str

@dataclass
class ResultsConfig:
    metrics_file: str
    plot_dir: str

@dataclass
class OpenAIConfig:
    api_key: str

@dataclass
class Config:
    task: str
    training: TrainingConfig
    feature_extractor: FeatureExtractorConfig
    evaluation: EvaluationConfig
    results: ResultsConfig
    openai: OpenAIConfig

class ConfigLoader:
    _config: Optional[Config] = None

    @classmethod
    def load_config(cls, config_path: str = 'config.j2') -> Config:
        """
        Load configuration from a Jinja2-templated YAML file and return a Config object.
        Implements singleton pattern to ensure configuration is loaded only once.
        """


        if cls._config is not None:
            return cls._config

        try:
            path = Path(config_path)

            if not path.is_file():
                sys.exit(1)

            # Set up Jinja2 environment
            env = Environment(
                loader=FileSystemLoader(path.parent),
                autoescape=False  # YAML files don't need autoescaping
            )

            # Add environment variable function to Jinja2 environment
            env.globals['env'] = os.getenv

            # Load the template
            template = env.get_template(path.name)

            # Prepare context for rendering (e.g., environment variables)
            context = os.environ.copy()

            # Render the template
            rendered_config = template.render(**context)

            # Load the rendered YAML
            cfg = yaml.safe_load(rendered_config)

            # Parse each section into corresponding dataclasses
            training_hyperparams = TrainingHyperparameters(**cfg['training']['hyperparameters'])
            training_cfg = TrainingConfig(
                dataset_file=cfg['training']['dataset_file'],
                images_directory=cfg['training'].get('images_directory', None),
                model=cfg['training']['model'],
                hyperparameters=training_hyperparams,
                save_model_to=cfg['training']['save_model_to']
            )

            feature_extractor_cfg = FeatureExtractorConfig(
                text_model=cfg['feature_extractor']['text_model'],
                image_model=cfg['feature_extractor'].get('image_model', None),
                train_text_features_outfile=cfg['feature_extractor']['train_text_features_outfile'],
                val_text_features_outfile=cfg['feature_extractor']['val_text_features_outfile'],
                train_image_features_outfile=cfg['feature_extractor'].get('train_image_features_outfile', None),
                val_image_features_outfile=cfg['feature_extractor'].get('val_image_features_outfile', None),
            )

            evaluation_hyperparams = EvaluationHyperparameters(**cfg['evaluation']['hyperparameters'])
            evaluation_cfg = EvaluationConfig(
                dataset_file=cfg['evaluation']['dataset_file'],
                images_directory=cfg['evaluation'].get('images_directory', None),
                hyperparameters=evaluation_hyperparams,
                prediction_output_path=cfg['evaluation']['prediction_output_path']
            )

            results_config = ResultsConfig(**cfg['results'])
            openai_cfg = OpenAIConfig(api_key=cfg['openai']['api_key'])

            cls._config = Config(
                task=cfg['task'],
                training=training_cfg,
                feature_extractor=feature_extractor_cfg,
                evaluation=evaluation_cfg,
                results=results_config,
                openai=openai_cfg
            )

            return cls._config

        except FileNotFoundError:
            print(f"Configuration file {config_path} not found.")
            sys.exit(1)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
            sys.exit(1)
        except TypeError as exc:
            print(f"Configuration structure error: {exc}")
            sys.exit(1)
        except KeyError as exc:
            print(f"Missing configuration key: {exc.args[0]}")
            sys.exit(1)
        except Exception as exc:
            print(f"An unexpected error occurred: {exc}")
            sys.exit(1)
