from torch import device
from .config import ConfigLoader
from src.utils.utils import get_device


# Load configuration when the config module is imported
config = ConfigLoader.load_config()
# DEVICE = get_device()
DEVICE = device('cpu')

__all__ = [
    "config",
    "DEVICE"
]