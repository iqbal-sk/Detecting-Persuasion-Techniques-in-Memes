import logging
import logging.config
import yaml
import os
from pathlib import Path


def setup_logging(default_path='logging_config.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    """
    Setup logging configuration from a YAML file.

    Args:
        default_path (str): Path to the default logging configuration file.
        default_level (int): Default logging level.
        env_key (str): Environment variable key to look for logging config path.
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if Path(path).is_file():
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        # Ensure log directories exist
        log_handlers = config.get('handlers', {})
        for handler in log_handlers.values():
            if 'filename' in handler:
                log_path = Path(handler['filename'])
                if not log_path.parent.exists():
                    log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.config.dictConfig(config)
    else:
        # Ensure default log directory exists
        Path('logs').mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=default_level)
        print(f"Warning: Logging configuration file '{path}' not found. Using basic configuration.")


def get_logger(name):
    # Define the path to the logging configuration file
    config_path = Path('logging_config.yaml')

    # Load the logging configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f.read())

    # Ensure the log directory exists
    log_file = config['handlers']['file']['filename']
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Configure logging
    logging.config.dictConfig(config)

    # Return a logger with the specified name
    return logging.getLogger(name)


# Initialize logging when the module is imported
setup_logging()
