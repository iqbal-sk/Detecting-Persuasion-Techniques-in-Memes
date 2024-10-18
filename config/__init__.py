from .config import ConfigLoader


# Load configuration when the config module is imported
config = ConfigLoader.load_config()

__all__ = [
    "config",
]